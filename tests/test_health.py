import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from model_council.health import HealthCache, _safe_error, probe_models
from model_council.inventory import ModelSpec


class ProbeModelsTests(unittest.TestCase):
    def test_marks_successes_and_failures_without_dropping_models(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek"),
        ]

        def invoke(model, prompt, role, effort):
            if model.provider == "deepseek":
                raise RuntimeError("invalid credential")
            return "HEALTH_OK"

        result = probe_models(models, invoke=invoke, max_workers=2)

        self.assertEqual([model.healthy for model in result.models], [True, False])
        self.assertEqual(result.diagnostics["openai-codex:gpt-5.6-sol"], "ok")
        self.assertIn("invalid credential", result.diagnostics["deepseek:deepseek-v4-pro"])

    def test_rejects_response_that_only_mentions_health_ok(self):
        model = ModelSpec("provider", "model", "family")

        for output in (
            "ERROR: could not produce HEALTH_OK",
            "ERROR: request degraded\nHEALTH_OK",
        ):
            with self.subTest(output=output):
                result = probe_models(
                    [model],
                    invoke=lambda *_, value=output: value,
                    max_workers=1,
                )

                self.assertFalse(result.models[0].healthy)
                self.assertIn("unexpected", result.diagnostics[model.key])

    def test_serializes_all_hermes_cli_health_probes(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic"),
        ]
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        def invoke(model, prompt, role, effort):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return "HEALTH_OK"

        result = probe_models(models, invoke=invoke, max_workers=2)

        self.assertTrue(all(model.healthy for model in result.models))
        self.assertEqual(max_active, 1)

    def test_redacts_health_diagnostics_with_the_invoker_policy(self):
        samples = {
            "Bearer bare-health-secret": "bare-health-secret",
            "client_secret=health-secret-value": "health-secret-value",
            "https://user:url-health-secret@example.com/path": "url-health-secret",
        }

        for text, secret in samples.items():
            with self.subTest(text=text):
                cleaned = _safe_error(RuntimeError(text))
                self.assertNotIn(secret, cleaned)
                self.assertIn("[REDACTED]", cleaned)

    def test_serializes_across_concurrent_probe_batches(self):
        state_lock = threading.Lock()
        start = threading.Barrier(2)
        active = 0
        max_active = 0
        errors = []

        def invoke(model, prompt, role, effort):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return "HEALTH_OK"

        def run_batch(model):
            try:
                start.wait(timeout=1)
                probe_models([model], invoke=invoke, max_workers=1)
            except Exception as exc:  # noqa: BLE001 - capture thread assertion failures
                errors.append(exc)

        threads = [
            threading.Thread(
                target=run_batch,
                args=(ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),),
            ),
            threading.Thread(
                target=run_batch,
                args=(ModelSpec("deepseek", "deepseek-v4-pro", "deepseek"),),
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(max_active, 1)

    def _mktempdir(self):
        path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(path), ignore_errors=True)
        return path

    def test_health_cache_reuses_fresh_results_without_storing_diagnostics(self):
        model = ModelSpec("provider", "model", "family")
        directory = self._mktempdir()
        cache = HealthCache(
            Path(directory) / "cache.json",
            ttl_seconds=900,
            failure_ttl_seconds=120,
        )
        failed = ModelSpec("provider", "failed", "family")
        cache.store({model.key: True, failed.key: False}, now=1000)
        self.assertEqual(
            cache.load([model, failed], now=1001),
            {model.key: True, failed.key: False},
        )
        self.assertEqual(cache.load([model, failed], now=1121), {model.key: True})
        self.assertEqual(cache.load([model, failed], now=2000), {})
        self.assertNotIn("diagnostic", cache.path.read_text(encoding="utf-8"))

    def test_cache_update_preserves_existing_entry_timestamp(self):
        first = ModelSpec("provider", "first", "family")
        second = ModelSpec("provider", "second", "family")
        directory = self._mktempdir()
        cache = HealthCache(Path(directory) / "cache.json", ttl_seconds=900)
        cache.store({first.key: True}, now=1000)
        cache.store({second.key: True}, now=1100)

        self.assertEqual(
            cache.load([first, second], now=1101),
            {first.key: True, second.key: True},
        )
        self.assertEqual(
            cache.load([first, second], now=1901),
            {second.key: True},
        )

    def test_concurrent_cache_writes_preserve_all_entries(self):
        directory = self._mktempdir()
        path = Path(directory) / "cache.json"
        models = [
            ModelSpec("provider", f"model-{index}", "family")
            for index in range(20)
        ]
        start = threading.Barrier(len(models))
        errors = []

        def write(model):
            try:
                start.wait(timeout=2)
                HealthCache(path).store({model.key: True}, now=1000)
            except Exception as exc:  # noqa: BLE001 - capture thread failures
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(model,)) for model in models]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual(errors, [])
        self.assertEqual(
            HealthCache(path).load(models, now=1001),
            {model.key: True for model in models},
        )


if __name__ == "__main__":
    unittest.main()

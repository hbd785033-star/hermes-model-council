import threading
import time
import unittest

from model_council.health import _safe_error, probe_models
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
            except Exception as exc:
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


if __name__ == "__main__":
    unittest.main()

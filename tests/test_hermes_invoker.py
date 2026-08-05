import subprocess
import unittest

from model_council.hermes_invoker import HermesInvoker, _redact
from model_council.inventory import ModelSpec


class HermesInvokerTests(unittest.TestCase):
    def test_invokes_isolated_tool_free_hermes_session_without_shell(self):
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, stdout="  MODEL_OK\n", stderr="")

        invoker = HermesInvoker(executable="hermes", timeout=42, run_command=fake_run)
        output = invoker(
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
            "analyze this",
            "advisor-1",
            "high",
        )

        self.assertEqual(output, "MODEL_OK")
        self.assertEqual(captured["command"][0:2], ["hermes", "chat"])
        self.assertIn("--ignore-rules", captured["command"])
        self.assertIn("--safe-mode", captured["command"])
        self.assertIn("--source", captured["command"])
        source_index = captured["command"].index("--source")
        self.assertEqual(captured["command"][source_index + 1], "model-council")
        self.assertIn("-t", captured["command"])
        self.assertIn("", captured["command"])
        self.assertFalse(captured["kwargs"].get("shell", False))
        self.assertEqual(captured["kwargs"]["timeout"], 42)

    def test_redacts_common_secret_shapes(self):
        samples = {
            "Authorization: Bearer bearer-secret-value": "bearer-secret-value",
            "Bearer bare-bearer-secret": "bare-bearer-secret",
            "sk-abcdefgh12345678": "sk-abcdefgh12345678",
            "OPENROUTER_API_KEY=supersecretvalue": "supersecretvalue",
            "client_secret=anothersecretvalue": "anothersecretvalue",
            "https://user:urlsecret@example.com/path": "urlsecret",
        }

        for text, secret in samples.items():
            with self.subTest(text=text):
                cleaned = _redact(text)
                self.assertNotIn(secret, cleaned)
                self.assertIn("[REDACTED]", cleaned)


if __name__ == "__main__":
    unittest.main()

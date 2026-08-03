import unittest

from model_council.inventory import discover_models


class DiscoverModelsTests(unittest.TestCase):
    def test_filters_virtual_and_unavailable_providers(self):
        payload = {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "providers": [
                {"slug": "moa", "authenticated": True, "models": ["default"]},
                {
                    "slug": "openai-codex",
                    "authenticated": True,
                    "models": ["gpt-5.6-sol", "gpt-5.6-luna"],
                    "capabilities": {"gpt-5.6-sol": {"fast": True, "reasoning": True}},
                },
                {"slug": "deepseek", "authenticated": False, "models": ["deepseek-v4-pro"]},
            ],
        }

        models = discover_models(payload=payload)

        self.assertEqual(
            [(model.provider, model.model) for model in models],
            [("openai-codex", "gpt-5.6-sol"), ("openai-codex", "gpt-5.6-luna")],
        )
        self.assertTrue(models[0].is_current)
        self.assertTrue(models[0].reasoning)
        self.assertEqual(models[0].family, "openai")


if __name__ == "__main__":
    unittest.main()

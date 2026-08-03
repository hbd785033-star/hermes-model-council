import unittest

from model_council.analysis import analyze_task


class AnalyzeTaskTests(unittest.TestCase):
    def test_classifies_security_sensitive_code_task(self):
        profile = analyze_task(
            "Review and fix the production authentication code, run tests, and check security risks"
        )

        self.assertEqual(profile.kind, "code")
        self.assertTrue(profile.needs_tools)
        self.assertGreaterEqual(profile.risk, 4)
        self.assertGreaterEqual(profile.complexity, 3)
        self.assertTrue(profile.benefits_from_diversity)

    def test_chinese_production_refactor_is_complex(self):
        profile = analyze_task("为生产认证系统设计安全重构方案并实现测试")

        self.assertEqual(profile.kind, "code")
        self.assertGreaterEqual(profile.complexity, 3)
        self.assertGreaterEqual(profile.risk, 4)


if __name__ == "__main__":
    unittest.main()

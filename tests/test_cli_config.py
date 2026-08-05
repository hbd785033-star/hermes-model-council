import tempfile
import unittest
from pathlib import Path

from model_council.cli import _save_config_with_rollback


class ConfigTransactionTests(unittest.TestCase):
    def test_restores_backup_when_save_raises_after_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.yaml"
            backup = Path(directory) / "config.yaml.backup"
            source.write_text("original", encoding="utf-8")
            backup.write_text("original", encoding="utf-8")
            checks = 0

            def save_config(config):
                source.write_text("partial", encoding="utf-8")
                raise RuntimeError("disk write failed")

            def check_config():
                nonlocal checks
                checks += 1

            with self.assertRaisesRegex(RuntimeError, "restored from backup"):
                _save_config_with_rollback(
                    {"moa": {}},
                    save_config=save_config,
                    source=source,
                    backup=backup,
                    check_config=check_config,
                )

            self.assertEqual(source.read_text(encoding="utf-8"), "original")
            self.assertEqual(checks, 1)

    def test_restores_backup_when_post_write_check_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.yaml"
            backup = Path(directory) / "config.yaml.backup"
            source.write_text("original", encoding="utf-8")
            backup.write_text("original", encoding="utf-8")
            checks = 0

            def save_config(config):
                source.write_text("broken", encoding="utf-8")

            def check_config():
                nonlocal checks
                checks += 1
                if checks == 1:
                    raise RuntimeError("invalid generated config")

            with self.assertRaisesRegex(RuntimeError, "restored from backup"):
                _save_config_with_rollback(
                    {"moa": {}},
                    save_config=save_config,
                    source=source,
                    backup=backup,
                    check_config=check_config,
                )

            self.assertEqual(source.read_text(encoding="utf-8"), "original")
            self.assertEqual(checks, 2)


if __name__ == "__main__":
    unittest.main()
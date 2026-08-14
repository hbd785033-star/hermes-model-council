import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from model_council.recommender import Plan


class HermesConfigTests(unittest.TestCase):
    def test_backup_hermes_config_copies_source_with_timestamped_name(self):
        from model_council.hermes_config import backup_hermes_config

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.yaml"
            source.write_text("moa: original", encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout=f"ignored\n{source}\n")
            with patch("model_council.hermes_config.subprocess.run", return_value=completed) as run:
                returned_source, backup = backup_hermes_config()

            run.assert_called_once_with(
                ["hermes", "config", "path"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
                check=False,
            )
            self.assertEqual(returned_source, source)
            self.assertTrue(backup.name.startswith("config.yaml.model-council-backup-"))
            self.assertEqual(backup.parent, source.parent)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_text(encoding="utf-8"), "moa: original")

    def test_backup_hermes_config_fails_closed_when_path_command_fails(self):
        from model_council.hermes_config import backup_hermes_config

        completed = SimpleNamespace(returncode=1, stdout="")
        with patch("model_council.hermes_config.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "Could not resolve Hermes config path"):
                backup_hermes_config()

    def test_backup_fails_closed_when_resolved_config_path_is_not_a_file(self):
        from model_council.hermes_config import backup_hermes_config

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.yaml"
            completed = SimpleNamespace(returncode=0, stdout=f"{missing}\n")
            with patch(
                "model_council.hermes_config.subprocess.run",
                return_value=completed,
            ), patch("model_council.hermes_config.shutil.copy2") as copy:
                with self.assertRaises(RuntimeError) as raised:
                    backup_hermes_config()

            self.assertEqual(
                str(raised.exception),
                f"Hermes config file does not exist: {missing}",
            )
            copy.assert_not_called()
            self.assertFalse(any(Path(directory).glob("*.model-council-backup-*")))

    def test_check_hermes_config_reports_stderr_on_failure(self):
        from model_council.hermes_config import check_hermes_config

        completed = SimpleNamespace(returncode=1, stderr="invalid config", stdout="ignored")
        with patch("model_council.hermes_config.subprocess.run", return_value=completed) as run:
            with self.assertRaisesRegex(RuntimeError, "Hermes config check failed: invalid config"):
                check_hermes_config()

        run.assert_called_once_with(
            ["hermes", "config", "check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            shell=False,
            check=False,
        )

    def test_install_native_presets_wires_validation_normalization_and_rollback(self):
        from model_council.hermes_config import install_native_presets

        plans = [Mock(spec=Plan)]
        existing = {"moa": {"existing": True}}
        generated = {"raw": True}
        normalized = {"normalized": True}
        source = Path("source.yaml")
        backup = Path("backup.yaml")
        config = {"moa": existing["moa"]}
        load_config = Mock(return_value=config)
        save_config = Mock()
        normalize = Mock(return_value=normalized)
        validate = Mock(return_value=[])
        build = Mock(return_value=generated)
        backup_helper = Mock(return_value=(source, backup))
        transaction = Mock()
        config_module = ModuleType("hermes_cli.config")
        config_module.load_config = load_config
        config_module.save_config = save_config
        moa_module = ModuleType("hermes_cli.moa_config")
        moa_module.normalize_moa_config = normalize
        moa_module.validate_moa_payload = validate
        hermes_package = ModuleType("hermes_cli")

        with patch.dict(
            sys.modules,
            {
                "hermes_cli": hermes_package,
                "hermes_cli.config": config_module,
                "hermes_cli.moa_config": moa_module,
            },
        ), patch("model_council.hermes_config.build_native_moa_config", build), patch(
            "model_council.hermes_config.backup_hermes_config", backup_helper
        ), patch("model_council.hermes_config.save_config_with_rollback", transaction):
            returned_backup, returned_normalized = install_native_presets(plans)

        load_config.assert_called_once_with()
        build.assert_called_once_with(plans, existing["moa"])
        validate.assert_called_once_with(generated)
        backup_helper.assert_called_once_with()
        normalize.assert_called_once_with(generated)
        self.assertIs(config["moa"], normalized)
        transaction.assert_called_once_with(
            config,
            save_config=save_config,
            source=source,
            backup=backup,
        )
        self.assertIs(returned_backup, backup)
        self.assertIs(returned_normalized, normalized)

    def test_install_native_presets_rejects_invalid_generated_config_before_mutation(self):
        from model_council.hermes_config import install_native_presets

        plans = [Mock(spec=Plan)]
        config_module = ModuleType("hermes_cli.config")
        config_module.load_config = Mock(return_value={"moa": {}})
        config_module.save_config = Mock()
        moa_module = ModuleType("hermes_cli.moa_config")
        moa_module.validate_moa_payload = Mock(return_value=["bad payload"])
        moa_module.normalize_moa_config = Mock()
        hermes_package = ModuleType("hermes_cli")

        with patch.dict(
            sys.modules,
            {
                "hermes_cli": hermes_package,
                "hermes_cli.config": config_module,
                "hermes_cli.moa_config": moa_module,
            },
        ), patch("model_council.hermes_config.build_native_moa_config", return_value={"raw": True}), patch(
            "model_council.hermes_config.backup_hermes_config"
        ) as backup_helper, patch(
            "model_council.hermes_config.save_config_with_rollback"
        ) as transaction:
            with self.assertRaisesRegex(RuntimeError, "Invalid generated MoA config"):
                install_native_presets(plans)

        backup_helper.assert_not_called()
        moa_module.normalize_moa_config.assert_not_called()
        transaction.assert_not_called()


if __name__ == "__main__":
    unittest.main()

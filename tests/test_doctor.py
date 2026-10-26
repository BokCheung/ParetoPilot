from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.config import load_config
from paretopilot.doctor import inspect_environment

ROOT = Path(__file__).resolve().parents[1]


class DoctorTests(unittest.TestCase):
    def inspect(self, import_torch):
        config = load_config(ROOT/"examples/ascend910b_matmul_shared_rhs.json")
        config["measurement"]["timeout_s"] = 123
        def command(argv, **kwargs):
            self.assertEqual(kwargs["timeout"], 123)
            if "import torch;" in argv[-1]:
                return import_torch(argv)
            return subprocess.CompletedProcess(argv, 0, "import OK", "")
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/"version.info").write_text("version=8.5.0\n")
            with patch("paretopilot.doctor.sys.platform", "linux"), \
                 patch("paretopilot.doctor.platform.platform", return_value="Linux-test"), \
                 patch.dict("os.environ", {"ASCEND_HOME_PATH": temp, "ASCEND_TOOLKIT_HOME": temp}), \
                 patch("paretopilot.doctor.shutil.which", side_effect=lambda name: None if name == "npu-smi" else "/mock/"+name), \
                 patch("paretopilot.doctor.subprocess.run", side_effect=command) as runner:
                result = inspect_environment(config)
        self.assertEqual(runner.call_count, 4)
        self.assertEqual(result["import_timeout_s"], 123)
        return result

    def test_import_uses_configured_timeout_instead_of_fixed_thirty_seconds(self):
        result = self.inspect(lambda argv: subprocess.CompletedProcess(argv, 0, "2.6.0", ""))
        self.assertTrue(result["ready"])

    def test_torch_timeout_remains_a_failure_with_actionable_diagnostic(self):
        def timeout(argv):
            raise subprocess.TimeoutExpired(argv, 123)
        result = self.inspect(timeout)
        self.assertFalse(result["ready"])
        failed = [c for c in result["checks"] if not c["passed"]]
        self.assertEqual([c["name"] for c in failed], ["torch"])
        self.assertIn("123 seconds", failed[0]["detail"])
        self.assertIn("--timeout", failed[0]["detail"])


if __name__ == "__main__":
    unittest.main()

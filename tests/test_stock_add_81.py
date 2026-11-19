import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.stock_add_81 import apply_tiling
from paretopilot.util import TuningError


class StockAddTilingTests(unittest.TestCase):
    def test_scoped_schedule_and_capacity_checks(self):
        shape = [1048593]
        inputs = [{"shape": shape, "dtype": "float16"}] * 2
        tiling = {"need_multi_core": 1, "block_axis": 0, "ub_axis": 0, "block_factor": 26240,
                  "ub_factor": 8832, "is_need_db": 1, "is_need_nb": 0}
        info = {"block_dim": 40}
        params = {"numel": shape[0], "block_elems": 65552, "ub_elems": 8192, "double_buffer": True}
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp)/"tiling.jsonl"
            with patch.dict(os.environ, {"PP_ADD_RECORD": str(record), "PP_ADD_TILING": json.dumps(params)}):
                apply_tiling(tiling, info, [128], inputs, 64, 24576, 12288, 40)
                self.assertFalse(record.exists())
                with self.assertRaises(TuningError):
                    apply_tiling(tiling, info, shape, inputs, 64, 24576, 4096, 40)
                self.assertEqual(info["block_dim"], 40)
                apply_tiling(tiling, info, shape, inputs, 64, 24576, 12288, 40)
            self.assertEqual(info["block_dim"], 16)
            self.assertEqual(tiling["block_factor"], 65552)
            self.assertEqual(tiling["ub_factor"], 8192)
            self.assertEqual(json.loads(record.read_text())["before"]["ub_factor"], 8832)

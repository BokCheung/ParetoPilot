from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from paretopilot.cli import main
from paretopilot.profiling import read_profile
from paretopilot.util import TuningError


FIXTURE = Path(__file__).parent/"fixtures/cann850_op_summary.csv"


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"op_summary.csv"
        with FIXTURE.open(newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return self.path

    def test_reported_cann850_csv_uses_task_duration(self):
        result = read_profile([FIXTURE], {"Op Type": "Add"}, 2)
        self.assertEqual(result["samples_us"], [8.5, 8.48])
        self.assertAlmostEqual(result["duration_us"], 8.49)
        self.assertIsNone(result["measured_bytes"])

    def test_old_and_new_headers_and_selectors_are_interoperable(self):
        old_names = {"OP Type": "Op Type", "Stream ID": "Stream Id", "Model ID": "Model Id", "Device_id": "Device Id"}
        variants = [self.rows, [{old_names.get(k, k): v for k, v in row.items()} for row in self.rows]]
        variants.append([{" "+k+" ": v for k, v in row.items()} for row in self.rows])
        for rows in variants:
            for selector in ({"Op Type": "Add", "Stream Id": "46", "Model Id": "1", "Device Id": "0"},
                             {"OP Type": "Add", "Stream ID": "46", "Model ID": "1", "Device_id": "0"}):
                with self.subTest(columns=list(rows[0]), selector=selector):
                    result = read_profile([self.write_rows(rows)], selector, 2)
                    self.assertAlmostEqual(result["duration_us"], 8.49)

    def test_mixed_task_identities_are_rejected_with_cann_headers(self):
        for column, changed in (("Stream ID", "47"), ("Model ID", "2"), ("Device_id", "1"), ("Op Name", "/OtherAdd")):
            with self.subTest(column=column):
                rows = deepcopy(self.rows)
                rows[1][column] = changed
                with self.assertRaisesRegex(TuningError, "Ambiguous profiling samples"):
                    read_profile([self.write_rows(rows)], {"Op Type": "Add"}, 2)

    def test_aliases_do_not_weaken_exact_operator_or_sample_matching(self):
        rows = deepcopy(self.rows)
        rows.append({**rows[0], "OP Type": "AddExtra"})
        path = self.write_rows(rows)
        self.assertEqual(read_profile([path], {"Op Type": "Add"}, 2)["samples_us"], [8.5, 8.48])
        for paths, selector in (([path], {"Op Type": "add"}), ([path, path], {"Op Type": "Add"})):
            with self.subTest(paths=paths, selector=selector), self.assertRaisesRegex(TuningError, "Expected exactly"):
                read_profile(paths, selector, 2)

    def test_missing_columns_report_path_and_available_headers(self):
        for missing in ("Task Duration(us)", "OP Type", "Total Bytes"):
            rows = [{k: v for k, v in row.items() if k != missing} for row in self.rows]
            path = self.write_rows(rows)
            kwargs = {"traffic_column": "Total Bytes"} if missing == "Total Bytes" else {}
            with self.subTest(missing=missing), self.assertRaises(TuningError) as caught:
                read_profile([path], {"Op Type": "Add"}, 2, **kwargs)
            message = str(caught.exception)
            self.assertIn(str(path), message)
            self.assertIn("missing=", message)
            self.assertIn("Op Type" if missing == "OP Type" else missing, message)
            self.assertIn("available=", message)
            self.assertIn("aiv_time(us)", message)

    def test_duplicate_headers_or_selector_aliases_are_rejected(self):
        rows = [{**row, "Op Type": "Other"} for row in self.rows]
        with self.assertRaisesRegex(TuningError, "duplicate or ambiguous column aliases"):
            read_profile([self.write_rows(rows)], {"Op Type": "Add"}, 2)
        self.path.write_text("Op Type,Op Type,Task Duration(us)\nOther,Add,8.5\n", encoding="utf-8")
        with self.assertRaisesRegex(TuningError, "duplicate or ambiguous column aliases"):
            read_profile([self.path], {"Op Type": "Add"}, 1)
        with self.assertRaisesRegex(TuningError, "selector has duplicate column aliases"):
            read_profile([FIXTURE], {"Op Type": "Add", "OP Type": "Other"}, 2)

    def test_inspect_cli_includes_original_headers_and_canonical_identifiers(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["inspect-profile", str(FIXTURE)])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())[0]
        self.assertIn("OP Type", result["columns"])
        self.assertEqual(result["tasks"], [{"Op Name": "/Add", "Op Type": "Add", "Device_id": "0",
                                            "Stream Id": "46", "Model Id": "1", "samples": 2}])


if __name__ == "__main__":
    unittest.main()

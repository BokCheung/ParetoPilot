from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import test_910b as script
from paretopilot.package import create_package
from paretopilot.util import TuningError, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = script.parser().parse_args(["--cann-path", str(self.root/"cann"), "--soc", "Ascend910B2", "--device", "3"])

    def make_cann(self, directory, version="8.5.0"):
        directory.mkdir(parents=True)
        (directory/"version.info").write_text("Version="+version+"\n")
        (directory/"set_env.sh").write_text("export EXAMPLE=1\n")
        return directory

    def artifacts(self, directory, smoke=True, failed_confirmation=False):
        c = read_json(ROOT/"examples/ascend910b_add.json")
        c["measurement"]["confirmation_rounds"] = 2 if smoke else 3
        write_json(directory/"manifest.json", {"config":c})
        def trial(role):
            return {"status":"ok", "mode":"ascend", "role":role, "correctness":{"passed":True}}
        confirmation = [trial(r) for i in range(c["measurement"]["confirmation_rounds"]) for r in (("baseline","candidate") if i%2==0 else ("candidate","baseline"))]
        if failed_confirmation:
            confirmation[-1]["status"] = "failed"
        result = {"mode":"ascend", "baseline":trial("baseline"), "history":[trial("candidate")],
                  "confirmation":{"trials":confirmation, "reason":"gain insufficient", "accepted":False},
                  "selected":"baseline", "selected_params":None}
        write_json(directory/"summary.json",result)
        return result

    def test_new_and_legacy_cann_layouts(self):
        modern = self.make_cann(self.root/"cann")
        self.assertEqual(script.resolve_cann(modern)[:2],(modern,modern/"set_env.sh"))
        parent = self.root/"ascend-toolkit"
        legacy = self.make_cann(parent/"latest")
        (legacy/"set_env.sh").unlink()
        (parent/"set_env.sh").write_text("export EXAMPLE=1")
        self.assertEqual(script.resolve_cann(parent)[:2],(legacy,parent/"set_env.sh"))
        self.assertEqual(script.resolve_cann(legacy)[:2],(legacy,parent/"set_env.sh"))

    def test_exact_version_and_explicit_env_required(self):
        cann = self.make_cann(self.root/"cann", "8.5.0.alpha001")
        with self.assertRaisesRegex(TuningError,"8.5.0 version"):
            script.resolve_cann(cann)
        (cann/"version.info").write_text("version=8.5.0\n")
        (cann/"set_env.sh").unlink()
        with self.assertRaisesRegex(TuningError,"cann-env"):
            script.resolve_cann(cann)
        env = self.root/"custom env.sh"
        env.write_text("export EXAMPLE=1")
        self.assertEqual(script.resolve_cann(cann,env)[1],env)

    def test_source_path_is_data_and_environment_is_not_logged(self):
        name = self.root/"env $(literal); test.sh"
        result = SimpleNamespace(stdout=b"PATH=/tools\0PYTHONPATH=/cann/python\0EXAMPLE_SECRET=not-logged\0")
        with patch.object(script.subprocess,"run",return_value=result) as run:
            env = script.source_environment(self.root,name,self.root/"source.log")
        argv = run.call_args.args[0]
        self.assertEqual(argv[-1],str(name))
        self.assertNotIn(str(name),argv[2])
        self.assertTrue(env["PYTHONPATH"].startswith(str(ROOT)))
        self.assertEqual(env["EXAMPLE_SECRET"],"not-logged")
        self.assertNotIn("not-logged",(self.root/"source.log").read_text())

    def test_config_overrides_and_originals_preserved(self):
        source = (ROOT/"examples/ascend910b_add.json").read_bytes()
        configs = script.prepare_configs(self.args,self.root)
        full,smoke = read_json(configs["add"]),read_json(configs["add-smoke"])
        self.assertEqual(full["ascend"]["device"],3)
        self.assertEqual(full["ascend"]["soc_version"],"Ascend910B2")
        self.assertEqual(full["agent"],{"kind":"dummy"})
        self.assertEqual(smoke["space"]["block_dim"],[1])
        self.assertTrue(all(len(v)==1 for v in smoke["space"].values()))
        self.assertGreater(len(full["space"]["block_dim"]),1)
        self.assertEqual((ROOT/"examples/ascend910b_add.json").read_bytes(),source)

    def test_confirmation_must_pass_even_when_cli_would_exit_zero(self):
        self.artifacts(self.root,failed_confirmation=True)
        with self.assertRaisesRegex(TuningError,"failed confirmation"):
            script.verify_result(self.root,True)

    def test_baseline_selection_is_a_valid_pass(self):
        self.artifacts(self.root)
        result = script.verify_result(self.root,True)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["selected"],"baseline")

    def test_synthetic_or_incomplete_results_never_pass(self):
        result = self.artifacts(self.root)
        variants = []
        a = deepcopy(result); a["history"][0]["correctness"]["simulated"] = True; variants.append(a)
        a = deepcopy(result); a["confirmation"]["trials"].pop(); variants.append(a)
        a = deepcopy(result); a["baseline"]["correctness"]["passed"] = False; variants.append(a)
        a = deepcopy(result); a["confirmation"]["trials"][0]["role"] = "candidate"; variants.append(a)
        for result in variants:
            write_json(self.root/"summary.json",result)
            with self.assertRaises(TuningError):
                script.verify_result(self.root,True)

    def test_full_search_allows_failed_trials_but_requires_valid_winner(self):
        result = self.artifacts(self.root,smoke=False)
        result["history"].append({"status":"failed"})
        write_json(self.root/"summary.json",result)
        self.assertEqual(script.verify_result(self.root,False)["failed_candidates"],1)

    def test_stage_failure_keeps_exit_status_and_log(self):
        session = script.Session(self.root,self.args)
        with redirect_stdout(io.StringIO()), self.assertRaises(TuningError):
            session.run("bad",[sys.executable,"-c","print('failure details');raise SystemExit(7)"])
        stage = read_json(self.root/"test-summary.json")["stages"][0]
        self.assertEqual(stage["exit_code"],7)
        self.assertEqual(stage["status"],"failed")
        self.assertIn("failure details",(self.root/"bad.log").read_text())

    def test_skipped_unit_tests_do_not_pass_acceptance(self):
        session = script.Session(self.root,self.args)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(TuningError,"skipped"):
            session.run("unit-tests",[sys.executable,"-c","print('OK (skipped=1)')"])

    def test_smoke_and_full_stage_order(self):
        for scope in ("smoke","full"):
            output = self.root/scope
            configs = script.prepare_configs(self.args,output)
            session = script.Session(output,self.args)
            stages=[]
            def run(name,command):
                stages.append(name)
                if name in ("add-smoke","matmul-smoke","add-full","matmul-full"):
                    self.artifacts(output/name,smoke=name.endswith("smoke"))
            session.run=run
            self.args.scope=scope
            script.execute_suite(session,self.args,sys.executable,configs)
            self.assertLess(stages.index("doctor"),stages.index("unit-tests"))
            self.assertLess(stages.index("matmul-baseline"),stages.index("add-smoke"))
            self.assertEqual(stages[-2:],["add-full","matmul-full"] if scope=="full" else ["add-smoke","matmul-smoke"])

    def test_failed_smoke_stops_before_full_search(self):
        configs=script.prepare_configs(self.args,self.root)
        session=script.Session(self.root,self.args)
        stages=[]
        def run(name,command):
            stages.append(name)
            if name=="add-smoke":
                self.artifacts(self.root/name,failed_confirmation=True)
        session.run=run
        with self.assertRaises(TuningError):
            script.execute_suite(session,self.args,sys.executable,configs)
        self.assertEqual(stages[-1],"add-smoke")
        self.assertEqual(session.data["current_check"],"verify-add-smoke")

    def test_setup_failure_produces_summary_and_diagnostics(self):
        output=self.root/"failed"
        with patch.object(script.sys,"platform","linux"),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            code=script.main(["--cann-path",str(self.root/"missing"),"--soc","Ascend910B1","--output",str(output)])
        self.assertEqual(code,2)
        self.assertEqual(read_json(output/"test-summary.json")["status"],"failed")
        with zipfile.ZipFile(output/"diagnostics.zip") as archive:
            self.assertIn("test-summary.json",archive.namelist())

    def test_existing_output_rejected(self):
        write_json(self.root/"keep.json",{"keep":True})
        with self.assertRaises(TuningError):
            script.create_output(self.root)
        self.assertTrue((self.root/"keep.json").is_file())

    def test_diagnostics_exclude_models_build_trees_and_custom_venv(self):
        for name in ("trial/build.log","trial/profiler/op_summary.csv","trial/project/build.log","trial/opp/metadata.json","env/private.txt","trial/model.om"):
            path=self.root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("data")
        with zipfile.ZipFile(script.diagnostics(self.root,excluded=(self.root/"env",))) as archive:
            self.assertEqual(set(archive.namelist()),{"trial/build.log","trial/profiler/op_summary.csv"})

    def test_offline_install_never_sets_remote_index(self):
        self.args.deps="offline"
        wheelhouse=self.root/"wheels with spaces"
        wheelhouse.mkdir()
        (self.root/"path alias").mkdir()
        self.args.venv=self.root/"environment"
        (self.args.venv/"bin").mkdir(parents=True)
        (self.args.venv/"bin/python").write_text("placeholder")
        for path in (wheelhouse, self.root/"path alias"/".."/wheelhouse.name):
            with self.subTest(wheelhouse=path):
                self.args.wheelhouse=path
                session=script.Session(self.root,self.args)
                commands=[]
                session.run=lambda name,argv:commands.append((name,argv))
                with patch.object(script.sys,"version_info",(3,12,0)):
                    script.prepare_python(session,self.args)
                # Select the installation stage, regardless of when version checks run.
                installs=[argv for name,argv in commands if name=="install-offline-deps"]
                self.assertEqual(len(installs),1,
                                 f"Expected one offline install; loaded {script.__file__}; recorded {commands!r}")
                self.assertFalse(any(name in {"install-deps","install-torch"} for name,_ in commands))
                install=installs[0]
                self.assertIn("--no-index",install)
                self.assertIn("--find-links",install)
                self.assertEqual(Path(install[install.index("--find-links")+1]).resolve(),wheelhouse.resolve())
                self.assertFalse(any(x in {"--index-url","--extra-index-url"} or x.startswith(("http://","https://")) for x in install))

    def test_source_archive_includes_script_and_requirements(self):
        target=self.root/"package.zip"
        create_package(target)
        with zipfile.ZipFile(target) as archive:
            self.assertIn("ParetoPilot/scripts/test_910b.py",archive.namelist())
            self.assertIn("ParetoPilot/requirements/910b.txt",archive.namelist())


if __name__ == "__main__":
    unittest.main()

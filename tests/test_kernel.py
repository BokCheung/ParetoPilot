import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from paretopilot.artifacts import ConfigError, fingerprint
from paretopilot.backend import Evaluator
from paretopilot.cli import main
from paretopilot.config import load_config
from paretopilot.profiling import read_profile
from paretopilot.roofline import analyze
from paretopilot.search import breed, confirm, tune_kernel
from paretopilot.space import advise, candidates, legal


ROOT = Path(__file__).resolve().parents[1]


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / 'examples/matmul.demo.json')

    def test_matmul_constraints(self):
        p = self.c['baseline_params']
        self.assertTrue(legal(p, self.c))
        for update in ({'tile_m': 17}, {'tile_m': 4096}, {'block_dim': 64}, {'buffers': 3}):
            self.assertFalse(legal({**p, **update}, self.c))
        self.assertTrue(all(legal(p, self.c) for p in candidates(self.c)))

    def test_elementwise_tail_and_roofline(self):
        c = load_config(ROOT / 'examples/elementwise.demo.json')
        self.assertTrue(legal(c['baseline_params'], c))
        r = analyze(c)
        self.assertEqual(r['flops'], 65537)
        self.assertEqual(r['minimum_bytes'], 65537*2*3)
        self.assertEqual(r['traffic_basis'], 'shape_lower_bound')
        self.assertEqual(analyze(c, 3, 123456)['traffic_basis'], 'profiling')

    def test_matmul_math(self):
        r = analyze(self.c)
        self.assertEqual(r['flops'], 2*256**3)
        self.assertEqual(r['minimum_bytes'], 3*256**2*2)

    def test_exact_profile_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'op_summary.csv'
            with p.open('w', newline='') as stream:
                w = csv.writer(stream)
                w.writerow(['Op Name', 'Task Duration(us)'])
                w.writerows([['MatMul', t] for t in (9, 10, 11, 12, 13)])
                w.writerow(['MatMulOther', 100000])
            self.assertEqual(read_profile(p, self.c)['duration_us'], 11)
            for sample in ('nan', 'inf', '-1', '0'):
                p.write_text('Op Name,Task Duration(us)\n'+f'MatMul,{sample}\n'*5)
                with self.assertRaises(ConfigError):
                    read_profile(p, self.c)
            p.write_text('Op Name,Task Duration(us)\nMatMul,1\n')
            with self.assertRaises(ConfigError):
                read_profile(p, self.c)

    def test_advisor_refines_and_expands(self):
        c = copy.deepcopy(self.c)
        c['space']['tile_m'] = [16, 32, 48, 64, 80]
        best = {'params': c['baseline_params'], 'duration_us': 9, 'roofline': {'parameter_priority': ['tile_m', 'tile_n']}}
        self.assertEqual(advise(c, best, 0)['ranges']['tile_m'], [16, 32])
        self.assertEqual(advise(c, best, 2)['ranges']['tile_m'], [16, 32, 48, 64, 80])

    def test_breed_preserves_elite_and_legality(self):
        pool = candidates(self.c)
        ranked = [{'params': pool[0], 'duration_us': 1, 'roofline': analyze(self.c)}]
        offspring = breed(self.c, ranked, pool, {fingerprint(pool[0])}, random.Random(7), advise(self.c, ranked[0], 0))
        self.assertIn(pool[0], offspring)
        self.assertTrue(all(legal(p, self.c) for p in offspring))
        self.assertEqual(len({fingerprint(p) for p in offspring}), len(offspring))

    def test_reject_false_correctness_and_missing_selection(self):
        class Broken(Evaluator):
            def call(inner, stage, directory, params, role):
                if stage == 'atc':
                    (directory / 'model.om').write_bytes(b'om')
                    return {'om_path': 'model.om'}
                if stage == 'verify':
                    return {'passed': False}
                return {}
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = Broken(self.c, tmp)
            b = evaluator.evaluate(self.c['baseline_params'], 'baseline', 'baseline')
            self.assertEqual(b['status'], 'failed')
            self.assertIn('Correctness', b['error'])
            t = evaluator.evaluate(self.c['baseline_params'], 'candidate')
            self.assertEqual(t['status'], 'failed')
            self.assertIn('ATC selection', t['error'])

    def test_confirmation_failure_and_noise_fallback(self):
        class Fake:
            def __init__(inner, failed=False):
                inner.failed, inner.calls = failed, []
            def evaluate(inner, params, label, role):
                inner.calls.append(role)
                # 1% apparent gain does not pass the configured 3% threshold.
                return {'status': 'failed' if inner.failed else 'ok', 'role': role, 'duration_us': 100 if role == 'baseline' else 99}
        fake = Fake()
        trials, accepted, _, _ = confirm(fake, self.c, {'params': self.c['baseline_params']})
        self.assertFalse(accepted)
        self.assertEqual(fake.calls[:4], ['baseline', 'candidate', 'candidate', 'baseline'])
        self.assertEqual(len(trials), 6)
        self.assertFalse(confirm(Fake(True), self.c, {'params': {}})[1])

    def test_invalid_config_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.c)
            c['search']['mutation_rate'] = 2
            path = Path(tmp) / 'bad.json'
            path.write_text(json.dumps(c))
            with self.assertRaises(ConfigError):
                load_config(path)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['validate', '--config', str(ROOT/'examples/matmul.demo.json')]), 0)
        self.assertEqual(json.loads(out.getvalue())['status'], 'valid')

    def test_demo_cannot_impersonate_real_measurements(self):
        c = copy.deepcopy(self.c)
        c['mode'] = 'real'
        with tempfile.TemporaryDirectory() as tmp:
            trial = Evaluator(c, tmp).evaluate(c['baseline_params'], 'baseline', 'baseline')
            self.assertEqual(trial['status'], 'failed')
            self.assertTrue((Path(tmp)/'baseline/export.log').is_file())

    def test_demo_both_operators_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('matmul', 'elementwise'):
                c = load_config(ROOT/f'examples/{kind}.demo.json')
                c['search'].update(population=4, generations=2, max_evaluations=6, confirm_rounds=2)
                cfg = Path(tmp) / f'{kind}.json'
                cfg.write_text(json.dumps(c))
                root = Path(tmp) / kind
                result = tune_kernel(cfg, root)
                self.assertEqual(result['mode'], 'simulation')
                self.assertLessEqual(result['evaluations'], 6)
                summary = json.loads((root/'summary.json').read_text())
                self.assertEqual(len(summary['confirmation']), 4)
                history = json.loads((root/'history.json').read_text())
                self.assertEqual(len({fingerprint(t['params']) for t in history}), len(history))
                self.assertTrue(all(t['status'] == 'ok' for t in history))
                self.assertIn('simulated', (root/'report.md').read_text())
                with self.assertRaises(ConfigError):
                    tune_kernel(cfg, root)


if __name__ == '__main__':
    unittest.main()

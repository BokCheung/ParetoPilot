import json
from pathlib import Path
import random
import tempfile
import unittest

from paretopilot import official
from paretopilot.config import load_config
from paretopilot.search import next_population
from paretopilot.space import enumerate_space
from paretopilot.util import TuningError, digest

ROOT = Path(__file__).resolve().parents[1]


class ExplicitTileTests(unittest.TestCase):
    def test_domains_cover_all_three_shapes_and_search_reaches_every_combination(self):
        for suffix, count in (("", 48), ("_shared_rhs", 39), ("_batched_rhs", 18)):
            c = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
            c["search"]["domain_mode"] = "manual"
            pool = enumerate_space(c)
            self.assertEqual(len(pool), count)
            self.assertEqual(official.default_params(c)["base_m"], 0)
            seen, ranked, rng = set(), [], random.Random(c["search"]["seed"])
            for _ in range(c["search"]["generations"]):
                population = next_population(c, ranked, pool, seen, {"space": c["space"]}, rng)
                for p in population:
                    if digest(p) not in seen and len(seen) < c["search"]["budget"]:
                        seen.add(digest(p))
                        ranked.append({"params": p, "duration_us": 10})
            self.assertEqual(len(seen), count)
            if suffix == "_batched_rhs":
                self.assertTrue(all(p["base_m"] == 16 for p in pool if p["base_m"]))

    def test_rejects_partial_misaligned_or_incompatible_tile_requests(self):
        p = {**official.BATCH_DEFAULT_PARAMS, "tiling_policy": 1, "batch_strategy": 1, "base_m": 32, "base_n": 64}
        official.validate_params(p, batch=True)
        for bad in ({"base_m": 0}, {"base_m": 17}, {"base_n": 512}, {"base_k": 64},
                    {"tiling_policy": 0}, {"batch_strategy": 0}, {"base_m": True}):
            with self.assertRaises(TuningError, msg=str(bad)):
                official.validate_params({**p, **bad}, batch=True)

    def test_materialized_host_injects_before_api_and_preserves_common_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"project"
            official.materialize(root, official.BATCH_DEFAULT_PARAMS.copy(), batch=True, runtime=True)
            host = (root/official.HOST).read_text(encoding="utf8")
            batch = (root/official.BATCH_HOST).read_text(encoding="utf8")
            init = host[host.index("ge::graphStatus MatmulV3BaseTiling::InitTilingData()"):
                        host.index("void MatmulV3BaseTiling::SetParamsV310()")]
            self.assertLess(init.index("SetFixSplit(PP_BASE_M, PP_BASE_N, -1)"), init.index("mm_.GetTiling"))
            op = host[host.index("ge::graphStatus MatmulV3BaseTiling::DoOpTiling()"):
                      host.index("bool IsPowerOfTwo") if "bool IsPowerOfTwo" in host else host.index("void MatmulV3BaseTiling::OptimizeLoadBalanceBasicKernel")]
            self.assertLess(op.index("if (PP_EXPLICIT_TILES)"), op.index("GetTilingFromRepo"))
            self.assertIn("runInfo_.baseK = ppApi.baseK", op)
            self.assertIn("runInfo_.needUpdate = false", op)
            self.assertIn("if (!PP_EXPLICIT_TILES) { TuneBaseMKN", batch)
            self.assertIn("ppL1 > compileInfo_.l1Size + 256UL", batch)
            self.assertIn("PP_EXPLICIT_TILES && PP_BATCH_STRATEGY != 1", batch)

    def test_effective_evidence_rejects_silently_overwritten_tiles(self):
        p = {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1, "base_m": 16, "base_n": 32}
        record = {**p, "used_core_num": 12, "base_k": 64, "tiling_key": 65536,
                  "block_dim": 12, "batch_count": 32, "multi_batch": 0, "load_mode": 0}
        request = {"base_m": 16, "base_n": 32, "base_k_auto": True}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"host.log"
            def write(r=record, req=request):
                path.write_text(f"PARETOPILOT_OFFICIAL={official.COMMIT}\n"+
                    "PARETOPILOT_TILE_REQUEST="+json.dumps(req)+"\n"+
                    "PARETOPILOT_BATCH_TILING="+json.dumps(r), encoding="utf8")
            write()
            self.assertEqual(official.read_effective_tiling(path, p, batch=True), [record])
            for r, req in (({**record, "base_m": 32}, request), (record, {**request, "base_k_auto": False}),
                           (record, {**request, "base_n": 64})):
                write(r, req)
                with self.assertRaises(TuningError):
                    official.read_effective_tiling(path, p, batch=True)
            auto = {**p, "base_m": 0, "base_n": 0}
            write(record, {"base_m": 0, "base_n": 0, "base_k_auto": True})
            self.assertEqual(official.read_effective_tiling(path, auto, batch=True), [record])


if __name__ == "__main__":
    unittest.main()

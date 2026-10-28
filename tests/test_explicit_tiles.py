import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from paretopilot import official
from paretopilot.config import load_config
from paretopilot.search import next_population
from paretopilot.space import enumerate_space
from paretopilot.util import TuningError, digest

ROOT = Path(__file__).resolve().parents[1]


class ExplicitTileTests(unittest.TestCase):
    def test_small_populations_measure_both_tiling_branches_for_all_shapes(self):
        for suffix in ("", "_shared_rhs", "_batched_rhs"):
            c = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
            c["search"]["population"] = 2
            pool = enumerate_space(c)
            for cached_explicit in (None, False, True):
                seen, ranked, rng = set(), [], random.Random(42)
                seed = next((p for p in pool if bool(p["base_m"]) == cached_explicit), None)
                for generation in range(3):
                    population = next_population(c, ranked, pool, seen, {"space": c["space"]}, rng,
                                                 seed=seed if generation == 0 else None)
                    self.assertEqual(len(population), 2)
                    self.assertEqual({bool(p["base_m"]) for p in population}, {False, True})
                    if generation == 0 and seed is not None:
                        self.assertEqual(population[0], seed)
                    for p in population:
                        self.assertNotIn(digest(p), seen)
                        seen.add(digest(p))
                        ranked.append({"params": p, "duration_us": 1 if p["base_m"] else 10})
                    ranked.sort(key=lambda t: t["duration_us"])

    def test_branch_coverage_respects_agent_pool_and_exhaustion(self):
        c = load_config(ROOT/"examples/ascend910b_batchmatmul.json")
        c["search"]["population"] = 2
        pool = enumerate_space(c)
        for explicit in (False, True):
            restricted = [p for p in pool if bool(p["base_m"]) == explicit]
            for available in (restricted, restricted[:1], []):
                population = next_population(c, [], available, set(), {"space": c["space"]}, random.Random(42))
                self.assertEqual(len(population), min(2, len(available)))
                self.assertTrue(all(p in available for p in population))
            seen = {digest(p) for p in restricted}
            population = next_population(c, [], pool, seen, {"space": c["space"]}, random.Random(42))
            self.assertTrue(all(bool(p["base_m"]) != explicit for p in population))

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
            lib = batch[batch.index("ge::graphStatus BatchMatmulV3BaseTiling::DoLibApiTiling()"):
                        batch.index("static void CalcBaseMN")]
            branch = lib[lib.index("    if (PP_EXPLICIT_TILES) {"):lib.index("    DoCommonTiling();")]
            self.assertIn("return ge::GRAPH_SUCCESS", branch)
            self.assertNotIn("DoCommonTiling();", branch)
            self.assertNotRegex(branch, r"pp\.stepN\s*=(?!=)")
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
                plan = {k: r[k] for k in ("base_m", "base_n", "base_k", "used_core_num")}
                plan.update(raw="01020304", single_core_m=16, single_core_n=256, step_n=16, depth_a1=1, depth_b1=16)
                path.write_text(f"PARETOPILOT_OFFICIAL={official.COMMIT}\n"+
                    "PARETOPILOT_API_CUBE_PLAN="+json.dumps(plan)+"\n"+
                    "PARETOPILOT_FINAL_CUBE_PLAN="+json.dumps(plan)+"\n"+
                    "PARETOPILOT_TILE_REQUEST="+json.dumps(req)+"\n"+
                    "PARETOPILOT_BATCH_TILING="+json.dumps(r), encoding="utf8")
            write()
            observed = official.read_effective_tiling(path, p, batch=True)[0]
            self.assertEqual(observed["cube_plan"]["step_n"], 16)
            self.assertEqual(observed["base_m"], record["base_m"])
            for r, req in (({**record, "base_m": 32}, request), (record, {**request, "base_k_auto": False}),
                           (record, {**request, "base_n": 64})):
                write(r, req)
                with self.assertRaises(TuningError):
                    official.read_effective_tiling(path, p, batch=True)
            auto = {**p, "base_m": 0, "base_n": 0}
            write(record, {"base_m": 0, "base_n": 0, "base_k_auto": True})
            self.assertEqual(official.read_effective_tiling(path, auto, batch=True)[0]["base_k"], 64)

    @unittest.skipUnless(shutil.which("g++"), "C++ compiler required for the real Host branch fixture")
    def test_compiled_host_branch_preserves_16x16_api_partition_and_buffer_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            project = directory/"project"
            official.materialize(project, official.BATCH_DEFAULT_PARAMS.copy(), batch=True, runtime=True)
            batch = (project/official.BATCH_HOST).read_text(encoding="utf8")
            lib = batch[batch.index("ge::graphStatus BatchMatmulV3BaseTiling::DoLibApiTiling()"):
                        batch.index("static void CalcBaseMN")]
            branch = lib[lib.index("    if (PP_EXPLICIT_TILES) {"):lib.index("    DoCommonTiling();")]
            cpp = r'''
#include <cstdint>
#include <cstring>
#include <cassert>
#include "cube_plan.h"
namespace ge { const int GRAPH_SUCCESS=0, GRAPH_FAILED=1; }
#define PP_EXPLICIT_TILES true
#define PP_BASE_M 16
#define PP_BASE_N 16
#define OP_LOGE(...) ((void)0)
struct Cube {
 uint32_t usedCoreNum=16, singleCoreM=16, singleCoreN=256, singleCoreK=256;
 uint32_t baseM=16, baseN=16, baseK=256, stepM=1, stepN=16, stepKa=1, stepKb=1;
 uint32_t depthA1=1, depthB1=16;
};
struct Fixture {
 struct { struct { Cube matmulTiling; struct { int mTileCntL2=0,nTileCntL2=0; } tileL2cacheTiling; } matmulTiling; } bmmTilingData_;
 struct { uint64_t aicNum=24,l1Size=524288; } compileInfo_;
 struct { uint64_t kValue=256; const char *opName="fixture"; } args_;
 uint64_t aDtypeSize_=2,bDtypeSize_=2;
 int commonCalls=0,keys=0;
 void DoTilingKeyCustom() { ++keys; }
 void DoCommonTiling() { ++commonCalls; bmmTilingData_.matmulTiling.matmulTiling.stepN=1; }
 int Execute() {
''' + branch + r'''
   DoCommonTiling(); return ge::GRAPH_SUCCESS;
 }
};
int main() {
 Fixture f; Cube before=f.bmmTilingData_.matmulTiling.matmulTiling;
 paretopilot::PrintCubePlan("API", before);
 assert(f.Execute()==ge::GRAPH_SUCCESS);
 const auto &after=f.bmmTilingData_.matmulTiling.matmulTiling;
 paretopilot::PrintCubePlan("FINAL", after);
 assert(std::memcmp(&before,&after,sizeof(Cube))==0);
 assert(after.singleCoreN==256 && after.stepN==16 && after.depthA1==1 && after.depthB1==16);
 assert(f.commonCalls==0 && f.keys==1);
 f.bmmTilingData_.matmulTiling.matmulTiling.singleCoreK=128;
 assert(f.Execute()==ge::GRAPH_FAILED);
}
'''
            (directory/"main.cpp").write_text(cpp, encoding="utf8")
            shutil.copyfile(ROOT/"paretopilot/official_adapter/cube_plan.h", directory/"cube_plan.h")
            binary = directory/"fixture"
            subprocess.run(["g++", "-std=c++11", str(directory/"main.cpp"), "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn('"single_core_n":256', result.stderr)


if __name__ == "__main__":
    unittest.main()

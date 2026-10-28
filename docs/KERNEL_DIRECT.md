# BatchMatMulV3 kernel direct execution (910B2 / CANN 8.5.0)

This is an opt-in hardware backend. The existing OM backend remains the default.
The implementation is CPU-tested with native ABI fixtures; it has not been compiled
or executed on a 910B2 in the development workspace. Hardware smoke is required.

## First hardware check

```bash
cd /home/ma-user/work/ParetoPilot
git pull --ff-only origin main
python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --soc Ascend910B2 --device 0 --deps check \
  --operator matmul --matmul-implementation batch-v3 \
  --matmul-execution kernel --matmul-case all --scope smoke --timeout 3600
```

This checks all three workloads without ATC/OM or unit/simulation tests in the
smoke stage. The source package, device binaries and native timing bridge must
be built on the first cache miss. That first build may take substantial time.
Later shapes, parameter values and runs reuse the verified build. With the new
tile domains, smoke checks both the automatic source baseline and one legal
explicit candidate from the generated domain for each workload. See its actual
parameters in `tile_smoke/trial.json`.
Use `--matmul-case 2d` to start with one shape.

## Diagnose OM versus direct timing

```bash
python3 scripts/test_910b.py --cann-path /usr/local/Ascend/cann-8.5.0 --soc Ascend910B2 --device 0 --deps check --operator matmul --matmul-execution kernel --matmul-case all --scope compare --timeout 3600
```

This diagnostic does not search or accept tuning. For each case, it exports the
inputs/reference once and uses the same default source parameters, device,
warmup and repeat count in three sequential stages:

1. Unprofiled direct launch: the existing Event interval P50.
2. Fresh ATC-built V3 OM: profiler target-task `Task Duration(us)` P50.
3. The same offline direct binary: profiler target-task `Task Duration(us)` P50,
   plus its Event interval while profiling (reported separately).

The source/compiled cache is reused. Fresh OM compilation still takes time.
Profiler is initialized before direct binary loading, starts after correctness
and warmup, and stops before final output validation. Output is checked during
each measured iteration as in normal direct search. Search uses a separate
per-trial capture and ranks profiler task time; this comparison additionally
keeps a deliberately unprofiled Event stage. Raw launches need not have V3 operator metadata; the
isolated process launches only one target kernel per measured iteration. The
parser requires exactly the configured count of core tasks, a single task
identity and matching Block Dim when exported. It rejects ambiguity or extra
core tasks instead of selecting by duration. Inspect raw CSV/inventory on failure.

Read `test-summary.json` for each case's three main metrics. Full evidence is in
`matmul-compare/comparison.json` and the equivalent 3D case directories; the
compact table is `comparison.md`. Metrics:

- `om_task_p50_us`: OM profiler task time, comparable to earlier single-digit us.
- `kernel_task_p50_us`: direct profiler task time, the matching measurement scope.
- `kernel_event_p50_us`: unprofiled direct Event interval; not pure kernel time.
- `profiled_kernel_event_p50_us`: Event interval under profiling; instrumentation
  may change this and it is not substituted for unprofiled timing.

If both task times are single-digit but Event time is ~250 us, the extra time
lies outside the profiled kernel task. If direct task time itself is ~250 us,
there is an execution-path regression to investigate. `diagnosis` reports both
ratios and compares semantic effective tiling (including cube partitions and
steps, excluding raw C++ padding). Factor-two findings are diagnostic heuristics,
not causal proof or statistical acceptance. Same source/params do not prove
identical binaries: OM uses ATC compilation, direct uses offline compiled kernels.

Preserved evidence includes OM `atc.log`, `selection.json`, profiler CSVs;
direct `host_tiling_request.json`, `host_tiling.json`, `host_tiling.log`,
`kernel_selection.json` (entry, binary/metadata hashes), `direct_result.json`,
profiler inventory, selected raw task rows, correctness and output arrays.
Failed stages leave partial `comparison.json` and the suite's diagnostics ZIP.

Some raw-kernel exports omit Block Dim or use `N/A`/empty values. These are
reported as unavailable, never filled from the expected Host value. Numeric
integer forms such as `24.0` are accepted; a numeric mismatch, fractional value
or malformed text is still rejected with its original value in the error.
`kernel_profile/profile_metrics.json` records `block_dim_check` (observed values,
matched/unavailable sample counts), and diagnosis marks incomplete verification.
The observed raw-launch export value `0` (including `0.0`) is also retained as
unverified metadata, with `zero_reported_samples` counted separately. This does
not establish why zero was exported, a zero-block launch, or a profiler match
to the positive Host dimension. Positive numeric mismatches still fail.

After updating the parser, recover the most recent existing capture without
ATC, kernel execution or re-export, using a command with no shell wildcards:

```bash
python3 -m paretopilot.compare_execution --reanalyze-latest runs
```

This selects the most recently modified comparison with a direct result and
exported CSVs, prints its exact path and metrics, and updates that case's
`comparison.json`/`comparison.md`. It does not change the original suite's
`test-summary.json` or rerun unstarted cases. For a specific case use
`--reanalyze runs/910b-YOUR-RUN/matmul-compare` instead.

## Candidate search without unit/simulation tests

```bash
cd /home/ma-user/work/ParetoPilot
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
python3 -m paretopilot run \
  --config examples/ascend910b_batchmatmul.json \
  --execution kernel --output "runs/kernel-tune-$(date +%Y%m%d-%H%M%S)"
```

For 3D shared weights use `examples/ascend910b_batchmatmul_shared_rhs.json`;
for 3D independent batch weights use `examples/ascend910b_batchmatmul_batched_rhs.json`.
To perform the source/default and explicit-tile checks, add `--smoke` and use a new output
directory. `--deps check` belongs to the acceptance script, not the module CLI.

`implementation.execution="kernel"` is the equivalent JSON setting.
It requires `cann_batch_matmul_v3`, FP16/ND and `build_cache=true`.
To return to the original execution path, use `--execution om` or omit the setting.
The acceptance script also supports `--matmul-execution kernel --scope full`,
which includes the full preflight unit/simulation suite before hardware search.

## Execution and compilation boundaries

### Explicit baseM/baseN search

The three example configurations enable `search.domain_mode="shape"`. Their
`space` is a preview/manual fallback; it is regenerated after the source baseline
provides verified platform capabilities. A pair of zeros keeps automatic upstream tiling. Both
dimensions must be nonzero for an explicit request, which requires
`tiling_policy=1` and `batch_strategy=1`. Partial automatic/explicit pairs and
dimensions exceeding the conservatively aligned workload shape are filtered.
Budget=64, population=8 and generations=12 limit the search. Generated domains
may exceed the budget; the report does not claim exhaustive or globally optimal
coverage. Failed candidates count toward the budget and remain visible.
Set `search.domain_mode="manual"` to retain the configured static `space`.

The Host calls `MultiCoreMatmulTiling::SetFixSplit(base_m, base_n, -1)` before
`GetTiling`; baseK is always computed by the official API and is not searchable.
Explicit requests preserve the complete API TCubeTiling, including single-core
partitions, L1 depths and steps. They use the common Device scheduler, while
bypassing the Host `DoCommonTiling` rewrite as well as KB/basic/split-K,
automatic base-M/N retuning and L1-full-load/multibatch selection.
Default `(0,0)` keeps the upstream automatic path.
The adapter currently accepts FP16 ND, no transpose/bias, on-the-fly K/N <=65535.
The actual CANN API validates the platform buffers. Additional L0C/L1 capacity
checks and exact final baseM/baseN matching reject unsupported or overwritten
requests rather than silently timing an automatic substitute.

`host_tiling.log` contains `PARETOPILOT_TILE_REQUEST` with requested dimensions
and `base_k_auto=true`, followed by the actual `PARETOPILOT_BATCH_TILING` record.
`PARETOPILOT_API_CUBE_PLAN` and `PARETOPILOT_FINAL_CUBE_PLAN` record named fields
and the entire TCubeTiling object as hex. For an explicit candidate they must
match exactly before launch. The API may choose singleCoreN=256 and stepN=16
for baseM=baseN=16; these dependent fields must not be replaced with base-sized
single-core partitions or unrelated L1 depths. API split-K plans are rejected by
the current common Device contract. Outer L2 partitioning is disabled for the
explicit path; the API plan is retained. Default automatic Host behavior remains.
`trial.json` records requested parameters and effective tiling, including actual
baseK and the final cube plan. Direct launches still require an exact compiled
tiling-key match. A correctness failure retains `kernel_selection.json`,
`correctness.json` and `output.npy` for diagnosis.

Installing the explicit-tile Host adapter for the first time invalidates older
non-tile build fingerprints. Once it is built, later shapes and parameter
combinations reuse it. Updates limited to the shape-domain/controller logic do
not require kernel recompilation. No manual cache deletion is needed.
CPU tests validate contracts and mock ABI execution; the new explicit path must
still pass the above smoke command on the 910B2 before performance conclusions.

Official API: [CANN 8.5 SetFixSplit](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0681.html).

### Shape domains and historical parameter cache

The flow is source baseline -> verified CANN platform -> shape-generated domains
-> Roofline and Agent proposal -> official Host validation -> measured search ->
fresh OM confirmation -> historical parameter record. Kernel mode obtains the
platform from its persistent worker. OM mode runs a small isolated platform probe.
Both query the actual SoC and TBE `CORE_NUM`, `L0A_SIZE`, `L0B_SIZE`, `L0C_SIZE`
and `L1_SIZE`. Missing/mismatched resources stop automatic domain generation;
Roofline's illustrative peak/bandwidth/buffer settings are not physical capacity
evidence. Reference profiling discovers the stock MatMul type for each workload
and rejects changes in that identity during confirmation.

Base tile candidates include aligned powers of two through 256 and additional
small/tail choices for dimensions <=128. M=1 produces `base_m=[0,16]`; N=100
allows choices through 112. Core budgets include lower counts and the actual
device budget, constrained by a conservative task bound. Necessary FP16 L0/L1
bounds filter explicit pairs; CANN remains the final dependency validator.
K/N beyond the current fixed-tile branch's 65535 limit retain automatic choices.
L1-full-load and multibatch remain available to automatic candidates, with their
existing fixed-tile restrictions unchanged. Dummy Agent retains the generated
domains and can later be replaced by an advisor proposing subsets of them.

Automatic tiling (`base_m=base_n=0`) and explicit baseM/baseN are searched in
the same run. Each population reserves a fresh candidate from each branch when
both remain in the Agent's pool; the remaining slots use genetic proposals and
random exploration. A historical seed remains first, while the other branch
still gets a reserved slot. Evaluated elites remain available for parent/winner
selection without consuming these slots again. This guarantees branch coverage
with population >=2 and at least two evaluations of remaining budget, not equal
budget allocation or exhaustive coverage. A successful exact historical retest
can still skip search as described below; use `--fresh-search` to explore anew.

Full runs use `.cache/tuning-results` separately from compiled source/kernel
cache. `search.result_cache_dir` overrides its location. Keys include workload
shapes and broadcasting, dtype/layout/transposition attributes, device/SoC/CANN,
driver version evidence, actual platform limits, verified installed source/build
identity, controller code, input seed, Agent and measurement settings. Without
verified build or driver evidence, historical reuse is unavailable.

- Exact hit: remeasure the historical parameters first, including correctness.
  Only a previously OM-confirmed configuration with gain exceeding the configured
  threshold and fresh sample range, and speedup >=90% of its historical speedup,
  can skip remaining candidate search. OM confirmation still runs afresh.
- Nearby shape: same environment, operator family and broadcasting, with each
  M/N/K/batch dimension within a factor of two. A still-legal cached configuration
  becomes an initial candidate only; all normal search and confirmation follow.
- Invalid, incompatible, corrupted or regressed records trigger normal search.
  The Agent can exclude any cached seed from its proposed domain.

No historical timing samples substitute for a fresh measurement. Cache writes
use unique temporary files and atomic replacement. To bypass historical hints
while retaining compiled binaries, add `--fresh-search` to either run command
or the acceptance script. `search.result_cache=false` disables both read/write;
`search.reuse_results=false` disables reading while allowing a new result save.

Artifacts: `domain.json`, `effective_config.json` and updated `manifest.json`
record the generated domains and real capacities; `agent/*` records what the
Agent received; `result_cache.json` records the hit, retest and reuse decision.
This Python-side update preserves existing valid compiled source/kernel caches.
Runtime domain generation and cache reuse still require hardware acceptance.

Platform resource API: [CANN 8.5 get_soc_spec](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/aicpuopapi/atlasplatform_07_0005.html).

1. The pinned official Host implementation and the original Device sources are
   built offline. Upstream binary generation is restricted to the existing
   FP16/ND BatchMatMulV3 recipe; its supported template branches are compiled
   together. We do not compile all operators or dtype/layout combinations.
2. A small C++ bridge is compiled against the installed CANN headers and
   `libascendcl.so`. It submits start-event, kernel and end-event calls natively,
   avoiding Python gaps between these calls.
3. A persistent Python worker initializes one device/stream and uploads fixed
   inputs once. Each search trial initializes its own profiler output, reloads
   the selected binary/function after profiler init for metadata registration,
   and unloads it before finalizing that capture. ACL/TBE/device/input state
   remains alive across candidates; source/device compilation is cached.
4. Each request invokes `OpTilingForCompile` from `libregister.so` after loading
   the private tiling library through `TbeLoadSoAndSaveToRegistry`. Compile-info
   is taken from the generated kernel metadata, with actual device and current
   parameter inputs. The original registered tiling/parse functions produce
   complete tiling bytes, key, blockDim and workspace size.
5. Parameters are read on every Host invocation, including consecutive requests
   in the same process. Matching source/parameter log signatures and actual
   tiling evidence are mandatory. No stale process-static parameter snapshot is
   retained.
6. The key must match exactly one `kernelList` entry in verified compiler output.
   `aclrtBinaryLoadFromFile`, `aclrtBinaryGetFunction` and
   `aclrtLaunchKernelWithConfig` load and launch that entry. The ABI is the
   upstream seven pointers: A, B, null bias, null offset, C, workspace, tiling.
   Missing/ambiguous keys, changed binaries, invalid atomic flags or unsupported
   workspace contracts fail the candidate; no different kernel is substituted.

Kernel catalogs and the timing bridge are checked by hash. Device code is
unchanged. Workspace and tiling buffers are allocated from actual Host outputs,
and optional inputs retain their ABI positions. Each candidate checks output
before timing and after each measured launch. When Host requests atomic cleanup,
output and workspace are zeroed before every launch, including warmup. Otherwise
output is poisoned to detect missing writes. Initialization, warmup and comparisons
are outside timing. Event intervals exclude the separate cleanup operations.

## Timing and final acceptance

Search `duration_us` and `samples_us` now come from profiler target-task
`Task Duration(us)`, labelled `timing_method=kernel_task_duration_us`. Both the
default V3 source baseline and every candidate use this same P50 metric.
Selection, genetic ranking, Roofline and cache speedups never use Event time.
Every trial profiles exactly the configured repeat count after correctness and
warmup, finalizes capture, exports its CSV and validates a unique target identity.
Missing, ambiguous or failed captures fail the candidate; there is no Event fallback.

The raw worker `direct_result.json` retains Event measurements for diagnosis.
Canonical search evidence is `trial.json` and `profile_result.json`; its
`event_diagnostics` retains Event P50/samples with `under_profiler=true` and is
not the unprofiled Event measurement from `--scope compare`. `profile_metrics.json`,
`profile_inventory.json`, the raw CSVs and export command/log are retained per
baseline/candidate. Block Dim zero remains explicitly unverified, never a match.
The search metric is also explicit in reports and `trials.csv`.

Historical result-cache identity includes the metric and a new format version,
so old Event-based results cannot seed or short-circuit profiler-ranked search.
Compiled OPP/device binaries are unchanged and remain reusable. Each trial now
pays profiler initialization/export overhead, but no ATC conversion is performed
for search candidates. Repeated init/finalize to select separate output paths
follows the [CANN profiler lifecycle](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/appdevg/acldevg/aclpythondevg_0077.html).

Roofline and Dummy Agent remain in the search loop. Candidate ranking uses only
direct V3 profiler task measurements against the default direct V3 source baseline. The stock
V2 reference still uses OM. The winning parameters are independently checked in
the existing ATC/OM path: V2 reference, default V3 and candidate V3 are measured
in interleaved confirmation rounds with fresh ATC conversions, correctness and
profiling samples. Gain acceptance uses those OM measurements only. Therefore
the final confirmation still takes time; kernel search does not remove it.

## Diagnostics

If an older build fails with `$'\r': command not found` or `source` cannot find a
script whose name has an invisible carriage return, update through GitHub and
start a new smoke run. Extracted `.sh` files are now normalized to LF before
configuration, with before/after hashes recorded in source provenance. The SoC
map rewrite also explicitly writes LF on Windows. Original source archives and
Device calculation sources retain their checksum verification. The changed
adapter fingerprint automatically selects a new cache generation; no manual
cache deletion or CANN installation edits are needed.

The direct CMake target also explicitly receives `ASCENDC_PER_COMPILE_JOB_THREAD`
from `implementation.build_jobs` (default 4). Upstream normally exports this
variable in `build.sh`; invoking a subtarget without it drops the thread argument
and produces `build_binary_single_op_exe_task.sh: [ERROR]input error`. The selected
value is recorded in `kernel-build-environment.json` and build provenance.

- `direct_prepare/cache-build-logs/kernel-build.log`: initial device compilation.
- `direct_prepare/cache-build-logs/launcher-build.log`: native timing bridge build.
- `direct-worker.log` and `direct_prepare/ready.json`: capabilities, initialization
  and worker failures. Missing CANN symbols are explicit errors.
- `direct_prepare/platform_init.json` and `platform_init.log`: actual runtime SoC,
  TBE compilation SoC, core count and initialization failures. Before Host tiling,
  the worker reads `aclrtGetSocName` and calls
  `tbe.common.platform.set_current_compile_soc_info(soc, core_type="AiCore")`
  once per worker. ACL device selection alone does not configure TBE's compile
  target. A mismatched device/configuration, failed initializer or invalid TBE
  platform stops execution. This addresses the missing initialization exposed
  by `GetPlatformInstanceByDevice failed`; confirmation on 910B2 is still required.
  No hardware resource constants are substituted. This worker-only change does
  not invalidate the existing verified kernel package cache.
- `source_baseline/` and `trials/*/host_tiling_request.json`, `host_tiling.log`,
  `host_tiling.json`: actual Host invocation, signatures and complete raw result.
- `host_tiling_diagnostics.json`: native API return code, recent ACL error,
  returned byte count/preview and registered library paths, including failed
  calls. The worker enables CANN console diagnostics before loading libraries;
  both native stdout and stderr are captured and worker exceptions print a
  traceback. A silent native failure still produces a nonempty call log and
  structured diagnostic record. This worker-only diagnostic update reuses
  existing verified kernel builds.
- `kernel_selection.json`, `selection.json`, `direct_result.json`, `trial.json`:
  key/entry/binary hashes, samples, timing method and selection evidence.
- `correctness.json` and `output.npy`: numerical checks against the fixed reference.
- `confirmation/*/atc.log`, profiling CSVs and `trial.json`: independent OM evidence.

No test-machine network service or Agent API is required. Transfer changes via
GitHub and run as the normal environment user, without sudo. A failing direct
candidate is never replaced silently with an OM or simulated measurement.

API references: [CANN 8.5 binary kernel execution](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00043.html),
[Kernel Launch](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1793.html),
[Host tiling JSON bridge source](https://github.com/Ascend/metadef/blob/master/register/op_tiling/op_tiling_py.cc).

Compilation platform API:
[set_current_compile_soc_info](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/aicpuopapi/atlasplatform_07_0003.html).

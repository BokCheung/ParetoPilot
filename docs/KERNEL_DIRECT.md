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
tile domains, smoke checks both the automatic source baseline and one explicit
baseM=16/baseN=32 candidate for each workload. See `tile_smoke/trial.json`.
Use `--matmul-case 2d` to start with one shape.

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

The three example configurations add `base_m=[0,16,32,64,128]` and
`base_n=[0,32,64,128]`. A pair of zeros keeps automatic upstream tiling. Both
dimensions must be nonzero for an explicit request, which requires
`tiling_policy=1` and `batch_strategy=1`. Partial automatic/explicit pairs and
dimensions exceeding the conservatively aligned workload shape are filtered.
There are 48 legal combinations for 2D, 39 for shared RHS and 18 for batched RHS,
including the existing automatic policy/core combinations. Budget=64,
population=8 and generations=12 allow the seeded search to visit every legal
combination with Dummy Agent retaining the domains. Failed candidates count
toward the budget and remain visible in the report.

The Host calls `MultiCoreMatmulTiling::SetFixSplit(base_m, base_n, -1)` before
`GetTiling`; baseK is always computed by the official API and is not searchable.
Explicit requests bypass the KB/basic/split-K algorithms, then use the existing
BatchMatMul common tiling to derive dependent L1 depths, steps and single-core
dimensions. Automatic base-M/N retuning and L1-full-load/multibatch branches are
excluded for these requests. Default `(0,0)` keeps the upstream automatic path.
The adapter currently accepts FP16 ND, no transpose/bias, on-the-fly K/N <=65535.
The actual CANN API validates the platform buffers. Additional L0C/L1 capacity
checks and exact final baseM/baseN matching reject unsupported or overwritten
requests rather than silently timing an automatic substitute.

`host_tiling.log` contains `PARETOPILOT_TILE_REQUEST` with requested dimensions
and `base_k_auto=true`, followed by the actual `PARETOPILOT_BATCH_TILING` record.
`trial.json` records requested parameters and effective tiling, including actual
baseK. Direct launches still require an exact compiled tiling-key match.

This Host adapter change invalidates the previous source build fingerprint.
The first updated smoke builds one new package/binary cache; subsequent shapes
and parameter combinations reuse it. No manual cache deletion is needed.
CPU tests validate contracts and mock ABI execution; the new explicit path must
still pass the above smoke command on the 910B2 before performance conclusions.

Official API: [CANN 8.5 SetFixSplit](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0681.html).

1. The pinned official Host implementation and the original Device sources are
   built offline. Upstream binary generation is restricted to the existing
   FP16/ND BatchMatMulV3 recipe; its supported template branches are compiled
   together. We do not compile all operators or dtype/layout combinations.
2. A small C++ bridge is compiled against the installed CANN headers and
   `libascendcl.so`. It submits start-event, kernel and end-event calls natively,
   avoiding Python gaps between these calls.
3. A persistent Python worker initializes one device/stream, uploads the fixed
   inputs once and retains loaded binaries/function handles during search.
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

Direct `duration_us` is P50 of device-event intervals, explicitly labelled
`timing_method=device_event_us`. It includes runtime submission gaps within the
event interval. It is **not** profiling `Task Duration(us)` P50 and must not be
compared directly with the stock V2 OM timing. The raw event samples, complete
Host tiling data and kernel metadata selection are retained.

Roofline and Dummy Agent remain in the search loop. Candidate ranking uses only
direct V3 measurements against the default direct V3 source baseline. The stock
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

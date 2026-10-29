# Add on Ascend910B2 / CANN 8.5.0

There are two separate routes. The source-tuning route uses the official
ops-math repository's **experimental, community-contributed AddV2**, pinned to
`v8.5.0`, commit `f1a42c5381bbd33981877ad87751d242663c2ce6`.
It is not the installed stock Add implementation. The stock route measures and
inspects that installed implementation first; it does not claim an unverified
same-source tiling adapter or direct kernel ABI.

## Source AddV2 smoke

```bash
cd /home/ma-user/work/ParetoPilot
git pull --ff-only origin main
python3 scripts/test_910b.py --cann-path /usr/local/Ascend/cann-8.5.0 --soc Ascend910B2 --device 0 --deps check --operator add --add-implementation add-v2 --add-execution kernel --scope smoke --timeout 3600
```

After smoke passes, replace `--scope smoke` with `--scope full` for search and
fresh alternating OM confirmation. `--add-execution om` uses OM throughout.
The first run builds a private OPP, FP16 device binary and timing bridge.
Subsequent parameters reuse `.cache/official-add-builds`; the source cache also
supports `--build-cache`. This adapter always needs its verified build cache.
Existing MatMul behavior is retained, but shared build-adapter code changes can
select a new MatMul cache generation on the next run. Do not delete caches.

References and acceptance:

| Role | Implementation | Purpose |
|---|---|---|
| stock reference (`baseline`) | Installed CANN Add | Performance reference only |
| `source_baseline` | Pinned AddV2 with `core_limit=0,tile_elems=0` | Primary optimization baseline |
| `candidate` | Same AddV2 source with Host controls | Tiling optimization |

Acceptance requires changed effective tiling and a stable gain over the same
source default in OM confirmation. It does not require beating stock Add.
Reports explicitly identify the experimental/community source and commit.
Roofline and Dummy Agent remain enabled; no Agent service is needed.

## Supported tensor and tiling contract

The first example is 1D, equal-shape FP16 inputs, `numel=1048576` (2^20).
The upstream kernel transfers whole 32-byte blocks. To keep Device source
unchanged, this adapter requires numel divisible by 16, positive and <=int32.
Unaligned sizes (including the old 1048593 case), broadcasting and other dtypes
are not enabled for this source adapter. The old bundled PpAdd is still available
explicitly with `--add-implementation bundled` and uses its previous OM path.

- `core_limit=0`: official platform-derived vector-core budget; positive values
  constrain it before all partition calculations. Values above real hardware
  capacity are rejected, not clamped silently.
- `tile_elems=0`: official UB-derived tile; explicit values are aligned to 16 and
  bounded by the official UB calculation before loop/tail derivation. Two buffers
  remain upstream's fixed Device setting and are not a search parameter.
- Default domain: core limits `0,1,4,8,16,24,32,40,48`; tile elements
  `0,256,1024,4096,8192,16384`. Budget 32, population 8, generations 8.

The original Device files are byte-for-byte checked against the pinned archive.
Host default controls do not rewrite its algorithm; positive controls alter inputs
before downstream dependencies. Raw tiling is exactly eight int64 fields. The
direct launcher checks partition coverage, loop/tail counts, UB use, Host evidence
and exact compiled tiling key 3 before launching the five-pointer ABI
`x1,x2,y,workspace,tiling`. 48 vector blocks are supported independently of the
MatMul launch limit of 24. No optional MatMul pointers or attributes are inserted.

Source baseline and candidates rank profiler target-task `Task Duration(us)` P50
(`timing_method=kernel_task_duration_us`). Event intervals are diagnostic only.
Each candidate gets a separate capture after warmup; missing/ambiguous samples
fail the candidate without an Event fallback. Final confirmation remains OM.

The offline source/dependency archives, checksums, provenance, patch list, actual
Host controls, compiler-generated FP16 binary recipe, kernel catalog and raw
profiling evidence are retained. Compilation uses the exact requested B2 SKU;
no CANN installation files are edited. Build logs are copied into the run's
`direct_prepare/cache-build-logs/` for diagnostics.

## Stock Add source inspection

```bash
python3 scripts/test_910b.py --cann-path /usr/local/Ascend/cann-8.5.0 --soc Ascend910B2 --device 0 --deps check --operator add --add-implementation stock --scope inspect --timeout 3600
```

This runs stock Add for the same 2^20 workload, captures profiler baseline timing,
then replays ATC with diagnostic logging and a private empty compiler cache.
It reads 910B registrations, Python entry/dependency information and target-node
compile/reuse evidence. It uses no sudo and does not require strace.

The run's `stock-add-inspect/brief.json` is the compact exchange artifact;
`report.json`, `registration-and-sources.json` and full logs retain details.
`tuning_supported=false` is deliberate: static matches/imports do not prove which
implementation executed, nor that its complete kernel/tiling source is present.
Installed Add may use TBE rather than Ascend C, so the AddV2 pointer ABI must not
be assumed for it. Actual installed branch evidence is required to complete that
source-tuning route; this command is not described as a stock tuning run.

Upstream sources:
[8.5.0 AddV2](https://github.com/hicann/ops-math/tree/v8.5.0/experimental/math/add_v2),
[main Add's platform registration](https://github.com/hicann/ops-math/blob/v8.5.0/math/add/op_host/add_def.cpp).
The main Add at that tag is registered for ascend910_95; it is not substituted
for a 910B implementation.

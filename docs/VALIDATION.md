# 本地验证记录

日期：2026-10-22。环境：Windows、Python 3.12.14。

`python -m unittest discover -s tests -v`：48 项全部通过，无跳过。

2026-10-23 增量验证：MatMul 实机 CSV 显示 `/MatMul` 的类型为 `BatchMatMulV2`，已修正真实基线示例筛选条件。新增回归测试后，`python -m unittest discover -s tests -p test_profiling.py -v` 的 8 项全部通过；Ascend MatMul 配置 validate 通过。回放所提供的两条记录为 7.520/7.620 μs，同时确认旧类型得到 0 条、两条记录不能满足正式测试的 20 条要求。本次未在本机重跑完整硬件流程。

同日 ONNX parser 修复：测试机目录确认生成文件为 framework/onnx_plugin/pp_add_plugin.cc。`python -m unittest discover -s tests -p test_native_contract.py -v` 的 8 项全部通过，覆盖 .cc/.cpp/.cxx 定位、其他框架隔离、缺失/多文件拒绝、保留 .cc 路径和原 CMake、source_manifest 记录以及 910B2 参数传递。构建命令使用 mock，这些结果不代表已完成真实 CANN 编译。

同日 ATC tiling key 修复：测试机日志在打印 Host 参数签名后报 `get_v220_kernel_type_mix_flag` / `KeyError: '1'`。Add、MatMul 的 Host 均选择 key 1，旧 Device 模板均缺少对应分支。新增 Host/Device 源码合同测试在修复前检出两个遗漏；补齐 `TILING_KEY_IS(1)` 后，`python -m unittest discover -s tests -p test_native_contract.py -v` 的 9 项全部通过。该检查只验证模板声明对应关系，构建依旧使用 mock；本机没有 CANN/NPU，尚未验证修复后的真实 ATC 转换和设备执行。

同日 MatMul Host 编译修复：测试机 `opbuild` 在 `SetAType/SetBType/SetCType` 三处报告 `AscendC::TPosition has not been declared`。三处改用 Host 的 `matmul_tiling::TPosition`，同时显式限定 CubeFormat、DataType 的命名空间，和[官方 Host tiling 示例](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0683.html)保持一致。重新运行上述原生链路合同测试，9 项通过。本地 mock 不验证 CANN 头文件或 C++ 编译，修复后的真实 MatMul 构建仍需测试机确认。

- 核心：配置约束、Roofline 公式和流量口径、Dummy 不调整参数域、插件域验证。
- 搜索：固定 seed 可复现、参数去重和预算、失败候选、噪声/小收益回退、已有输出保护。
- 采样：精确筛选、中位数、重复导出、混合节点、NaN 拒绝。
- CANN CSV 兼容：新增 7 项测试，使用测试机提供的 Add 表头和两条记录验证 `OP Type`、`Stream ID`、`Model ID` 等别名；Task Duration 中位数为 8.49 μs。同时验证新旧配置互通、混合标识拒绝、重复/缺失列诊断和 inspect-profile 输出。该 CSV 回放不代表完整上板流程已通过。
- 原生链路合同：构建参数注入、独立 OPP、参数签名缺失、OM 被替换时拦截。
- PyACL mock：预热排除、ND padding、错误输出停止、设备调用失败清理。
- 实际 CPU 导出：内置 MatMul、Add 和外部 Python 基线的 PyTorch→ONNX 导出及 ONNX schema 检查。
- 子进程：超时终止及命令日志；修复了 Windows venv 启动器的子进程残留问题。
- 一键测试脚本：新增 17 项测试，覆盖 CANN 新旧目录与精确版本检查、环境脚本路径传参、配置复制、在线/离线命令生成、阶段顺序、真实结果验收、失败即停、退出码、已有输出保护及诊断包筛选。CANN 环境加载和设备阶段使用 mock 验证控制逻辑；不是 Linux/CANN 实测。

本地导出测试使用 torch 2.14.1、onnx 1.23.2、numpy 2.5.4；这些是本机验证版本，不是目标 CANN 环境的强制依赖版本。目标机器优先使用兼容其 Python/PyACL/ATC 的组件组合。导出固定使用 ONNX opset 13 和 TorchScript 导出路径。

`scripts/test_910b.py --help` 和 Python 编译检查通过。源码包包含一键脚本及 requirements/910b.txt。脚本默认安装的 torch 2.7.1、onnx 1.16.1、numpy 1.26.4 组合尚未在本机安装验证；Linux 的 set_env.sh 加载、该依赖组合和硬件阶段仍须在目标测试机验收。

另外已跑通 MatMul、Add 两套模拟搜索，每套 12 个候选，生成完整报告。模拟数据与 PyACL mock 不代表 NPU 计算结果。

**本机尚未验证**：910B 上的 Ascend C 编译、OPP 注册、ATC 自定义节点转换、真实 PyACL 推理/精度、完整 msprof 采集导出链路，以及硬件性能收益。交付中含完成这些阶段所需的源代码和启动命令，按 DEPLOYMENT.md 先运行 smoke 验收。
# Explicit baseM/baseN search (2026-10-27)

Added optional paired `base_m/base_n` controls to the pinned BatchMatMulV3 Host
adapter. Explicit requests enter `SetFixSplit(M,N,-1)` before official API
dependency calculation, retain automatic baseK, bypass tile-rewriting strategies
and verify final dimensions. CPU checks cover all three shapes, domain filtering,
complete seeded search coverage, persistent native ABI fixtures, overwritten-tile
rejection, smoke evidence and unchanged Device source hashes. Full local
`python -m unittest discover -s tests -q`: 150 tests, 147 passed, 3 skipped.
These checks do not establish successful CANN compilation or NPU execution of
the new branch; the updated three-case hardware smoke remains required.

# Shape-generated domains and historical parameters (2026-10-27)

Automatic domains now use verified CANN SoC/core/L0/L1 evidence before Agent proposal. Tests cover changed shapes and actual capacity bounds, the OM probe, exact and nearby historical parameters, build/device/driver prerequisites, corrupted cache, fresh-search override, regression fallback, source/explicit smoke, stock operator discovery and fresh OM confirmation after an exact hit. The final full local unittest run completed 157 tests: 154 passed and 3 skipped. One earlier run encountered a Windows timeout-log cleanup sharing error; its isolated rerun and the subsequent full run passed. CANN/NPU execution of the new platform queries and adaptive flow remains to be validated on the 910B2.

# Preserve the complete explicit API plan (2026-10-27)

910B2 smoke exposed a correctness failure at baseM=baseN=16, automatic baseK=256. The API log showed singleCoreN=256, stepN=16, depthA1=1 and depthB1=16; the adapter subsequently ran DoCommonTiling and replaced dependent fields. The explicit branch now returns with the complete API TCubeTiling preserved, uses the existing common Device scheduler, and disables the outer Host L2 partition rewrite. API split-K plans are rejected. API/final named fields and raw object bytes must agree before launch, and failing numerical checks retain kernel selection and output artifacts. Regression checks cover all three ABI workloads and reject a changed stepN before launch. A compiler-dependent fixture executes the actual extracted Host branch on the supplied 16x16 plan and checks byte preservation. Full local unittest run: 158 tests, 154 passed, 4 skipped (including unavailable C++ compiler checks). The development host has no CANN/NPU, so successful 910B2 compilation and output correctness after this fix remain unverified until hardware smoke.

# Joint automatic and explicit search coverage (2026-10-28)

BatchMatMulV3 populations reserve unseen candidates from both automatic and explicit tiling branches when both remain in the Agent pool. Historical seeds retain first priority without displacing the other branch from a population of two. Tests cover all three workloads, repeated generations with ranked elites, automatic/explicit historical seeds, Agent-restricted pools and exhausted branches. The targeted explicit-tile/shape-domain suite completed 14 tests: 13 passed and one compiler-dependent fixture skipped. The full local suite completed 160 tests: 156 passed and 4 skipped. This controller-only change does not modify Host/Device source or invalidate compiled OPP binaries. Historical-result reuse may still short-circuit search after a valid exact retest; use --fresh-search to force exploration.

# OM/direct profiler comparison (2026-10-28)

The acceptance script supports --scope compare for all three V3 cases, without search or tuning acceptance. Each case shares exported inputs, default parameters and the source build across unprofiled direct Event timing, fresh OM task profiling, and an isolated profiled direct replay. Native ABI fixtures verify profiler initialization before binary load, exclusion of correctness/warmup launches, exact measured launch count and cleanup for all three workloads. CSV tests cover unknown raw-kernel names, aliased headers, extra/duplicate/missing tasks, conflicting identities and Block Dim mismatch. Controller tests cover shared inputs/params, partial failure evidence, separate Event/task metrics, semantic tiling differences and diagnostic heuristics. Targeted comparison/direct/script suite: 33 passed. Full local regression: 165 tests, 161 passed, 4 skipped. The small final error-reporting adjustment passed a further four-test comparison rerun. No CANN/NPU is available locally; the actual raw-launch profiler export and the cause of the reported ~250 us Event interval remain to be established on 910B2. Host/Device source and native timing bridge are unchanged, so existing compiled OPP caches remain reusable.

# Profiler Block Dim compatibility and offline recovery (2026-10-28)

The original parser raised Invalid profiler Block Dim for any non-int string without preserving its value. Parsing now accepts exact integral decimal/scientific forms, records empty/missing/N/A metadata as unavailable, and still rejects mismatches, fractional, nonfinite and malformed values. The original CSV values and checked/unavailable counts remain in the result; diagnostic output states when independent profiler verification is incomplete. Reanalysis can recover existing case reports or locate the latest exported comparison without invoking compilation, device execution or export. Tests cover integer/decimal forms, unavailable/partial metadata, invalid/mismatched values, selection of an eligible capture over an incomplete newer run, and recovery from the prior parsing error. The targeted comparison/direct/script suite completed 35 tests, all passed. The actual offending test-machine value has not yet been supplied; the new audit/error output makes it visible without another hardware run.

Follow-up evidence identified the exact raw value as '0'. Zero is now retained as unverified profiler metadata rather than a positive dimension match, counted separately and accompanied by the incomplete-verification diagnostic. Positive mismatches still fail. Six comparison tests passed, including zero/decimal-zero handling and recovery of an existing zero-valued capture without compilation or device execution. This does not establish why the profiler exported zero or diagnose the original timing discrepancy.

# Rank kernel search by profiler task P50 (2026-10-28)

910B2 comparison evidence reported OM task 9.55 us, direct task 7.82 us and unprofiled Event interval 301.37 us. Kernel search now profiles the default source baseline and every candidate independently, exports each capture and ranks only target-task Task Duration(us) P50 (kernel_task_duration_us). Event samples remain diagnostic, explicitly marked as taken under profiling. Missing/ambiguous/failed captures never fall back to Event ranking. The persistent worker retains ACL/TBE/input state, reinitializes profiling output per trial, registers a temporary binary/function after profiler init and unloads before finalize. The shared strict parser retains Block Dim zero as unverified metadata. Compare mode explicitly requests unprofiled direct timing, and OM final confirmation is unchanged. Cache identity/retest/save rules separate Event and task metrics; compiled source/device cache inputs are unchanged.

Native fixtures cover repeated per-candidate profiling, warmup exclusion, registration/finalization ordering, failure cleanup and one ACL init across all three shapes. Backend tests use reversed Event/task ordering, validate task P50, and reject missing CSVs. Controller tests for all three shapes select the task-fast candidate despite reversed Event ordering and retain OM confirmation. Historical Event records cannot seed or short-circuit task search. Reports, smoke and full verification use the new metric. Full local regression completed 168 tests: 164 passed, 4 skipped. The extended three-case ranking fixture passed in a separate eight-test shape/cache rerun. No local CANN/NPU is available; the newly repeated per-trial lifecycle requires the test-machine run, although standalone direct profiler capture has already been demonstrated by the user's measurements.

# AddV2 source route and installed stock Add inspection (2026-10-29)

The source route pins ops-math v8.5.0 experimental/math/add_v2 at f1a42c5381bbd33981877ad87751d242663c2ce6. Upstream Device files remain byte-identical; the scoped FP16 aligned-size adapter injects core/tile constraints before official dependency derivation and uses the exact eight-int64/five-pointer ABI. Source/parameter markers, raw partition/loop/tail and UB validation precede direct launch. Tests cover the 48-vector-core path, two-input/no-attribute contract, independent CPU addition through the native ABI fixture, source default/candidate smoke, task-profiler search metadata, source-based gain policy, OM-only final confirmation, build-cache reuse/corruption rejection, and provenance/report distinction from stock Add. The actual upstream binary-recipe generator/merger is executed in a CPU test; the attribute-free FP16 case preserves its generated keys. Existing MatMul adapters remain covered by regression.

Stock inspection is a separate baseline+ATC-replay command with exact Add registration/log matching, candidate/dependency collection and a bounded brief.json; it explicitly reports tuning_supported=false until the installed compile branch and modifiable source/ABI are established. It never substitutes AddV2 for stock source tuning. No test-machine stock Add source evidence has yet been supplied. An early regression exposed a summarize-without-report.json compatibility issue, fixed without changing the original BatchMatMulV2 default. Final full local regression: 175 tests, 171 passed and 4 skipped. Local checks do not establish CANN compilation or NPU correctness/performance; AddV2 hardware smoke and stock-source evidence remain required.

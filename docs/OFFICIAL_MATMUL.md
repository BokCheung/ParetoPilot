# CANN 8.5.0 官方 MatMul 源码调优

默认真实 MatMul 配置已切换为 `implementation.kind=cann_matmul_v3`。使用 [CANN ops-nn 官方仓库](https://gitcode.com/cann/ops-nn) `8.5.0` 分支的固定提交 `9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef`，算子目录为 `matmul/mat_mul_v3`。原始源码、公共依赖、JSON 头文件、makeself 及许可证均随 GitHub 仓库分发，测试机不需要访问 GitCode，也不需要 Agent 服务。

这与旧的 `PpMatmul` 教学式高阶 API 模板不同：新流程直接编译官方 Host tiling 和完整 Device kernel；没有重新手写矩阵乘法计算实现。CANN 内置 ONNX MatMul 在已测机器上显示为 `BatchMatMulV2`，本源码导出为 `MatMulV3`，不能声称它就是安装包中那个算子的逐字源码或相同二进制。

## 调优入口

| 参数 | 默认值 | 当前域及含义 |
| --- | --- | --- |
| core_limit | 0 | 0/8/12/16/20/24；0 使用实际 AI Core 数，正数在官方 tiling 开始前限制核心预算，超过物理核数会失败 |
| tiling_policy | 0 | 0 保留官方知识库查询及启发式回退；1 跳过知识库，直接走官方启发式 tiling |

首先开放这两个有明确语义的入口。baseM/N/K、流水线、L1/L0 缓冲、分核方式、tiling key 与 workspace 仍由官方代码联合计算，不把旧模板的 tile_m/n/k 生硬写进官方 tiling 的最终输出。后续扩展细粒度参数时，需要同时处理官方算法中的资源约束和耦合字段。

支持固定二维 FP16、ND、M/N/K 为 16 的倍数、无转置/bias/batch；示例仍为 256×256×256。Roofline 保留原有流量口径和硬件峰值假设，优先级名称改为这两个实际参数。Dummy Agent 原样返回参数域，未来 Agent 接口保持 `propose(context)`；其中 `baseline` 是官方源码默认配置，新增 `stock_baseline` 是 CANN 内置对照。

## 在 910B2 测试机运行

在已有 Python 依赖的环境执行。所有代码更新都通过 GitHub。

```bash
cd /home/ma-user/work/ParetoPilot
git pull --ff-only origin main
python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --soc Ascend910B2 --device 0 \
  --deps check --operator matmul --scope full --timeout 3600
```

这条命令先做环境、单元测试和模拟检查，再运行内置基线、官方源码 smoke、完整 12 组参数搜索及交错复测。`--scope full` 已包含 smoke。仅首次验证编译与执行时改为 `--scope smoke`。省略 `--operator matmul` 则同时验收 Add。脚本自动创建新的输出目录，并加载指定 CANN 的环境。

需要完整 CANN 8.5.0 Toolkit、匹配的 910B ops 包、Ascend C 开发组件、ATC、msprof、PyACL、CMake、g++、make。官方 tiler 依赖安装包中的 `libophost_comm_legacy.so`，doctor 会提前检查。构建时只将该依赖复制到本次试验的隔离 OPP 相对目录，满足官方 `LegacyCommonMgr` 的路径规则；不修改系统 OPP。MatMul 流程不依赖 msopgen，Add 仍使用它。

`--deps check` 不安装 Python 包。官方源码构建的 JSON/makeself 依赖已内置；不下载 protobuf、Abseil 或 Eigen。使用基于 GE Operator 的简短 ONNX parser 连接现有 ATC 流程，因此不需要上游通用 ONNX parser 的 protobuf 工具链。

固定版本的 `MatMulV3` 只注册了形状推导。隔离自定义 OPP 的 ATC 路径还需要数据类型推导，因此适配层在同一条 `IMPL_OP_INFERSHAPE(MatMulV3)` 注册中补上 `InferDataType`：输出 `y` 的类型取输入 `x1` 的类型，与官方 OpDef 一致，当前用例均为 FP16。原有形状推导和 Device kernel 保持原样，默认源码与候选使用相同的类型推导适配。

若旧版本报 `Can not find Node target[MatMulV3] custom infer_datatype func`，更新 GitHub 代码后重新运行 smoke，脚本会在新目录重建 OPP。无需修改 CANN 安装目录或设置 `IGNORE_INFER_ERROR=1`；该适配也会记录在 `source_manifest.json` 的补丁哈希中。

上游打包规则在 CMake 配置阶段收集 `dynamic/*.py`，而这些 Python 编译入口在后续 `ascendc_impl_gen` 阶段才生成。首次构建会因此漏装 `paretopilot_nn_impl/dynamic`，导致 ATC 报 `No module named 'paretopilot_nn_impl.dynamic'`。适配层改为安装阶段收集 Python 文件；生成阶段必须产出 `mat_mul_v3.py`，安装后再逐个核对生成入口的路径和 SHA256，缺失或不一致会在 ATC 之前失败。更新 GitHub 后重新运行 smoke 即可重建。

## 三路测量与判定

1. `baseline/`：内置 ONNX MatMul → ATC，清除自定义 OPP 路径。
2. `source_baseline/`：官方源码，`core_limit=0, tiling_policy=0`，保留原始 tiling 决策。
3. `trials/`：同一份官方源码，只改变上述入口。

三路使用同一输入、CPU 参考输出、精度阈值、预热和采样设置。完整搜索之后，每轮按 `baseline → source_baseline → candidate` 和逆序交错重建、复测。每次仍必须恰好取得 20 个目标任务样本；无法匹配时保留错误，不放宽筛选混入其他任务。

| summary.json 字段 | 含义 |
| --- | --- |
| confirmation.baseline_us | 内置算子复测耗时中位数，微秒 |
| confirmation.source_baseline_us | 官方源码默认配置复测耗时中位数，微秒 |
| confirmation.candidate_us | 最佳候选复测耗时中位数，微秒 |
| confirmation.speedup | 内置耗时 / 候选耗时 |
| confirmation.source_speedup | 官方源码默认耗时 / 候选耗时 |
| confirmation.tuning_accepted | 相对官方源码默认配置的收益是否超过阈值和观测波动 |
| selected | baseline / source_baseline / candidate |

候选必须同时优于两个对照，才被选为 `candidate`。若默认官方源码优于内置、调参没有确立进一步收益，可选 `source_baseline`。否则保留 `baseline`。smoke 使用原始默认参数，不会把相同参数的重复构建误报为调优收益。流程通过只表示构建、精度、采样和复测成功，不保证提速。

`test-summary.json` 的 `stages[].seconds` 是包含编译、ATC 和执行的阶段墙钟时间；`results.*.task_metrics` 才是算子微秒指标。每个确认轮次重新编译，完整测试会明显长于单个 kernel 的运行时间。

## 溯源与日志

- `paretopilot/vendor/cann_8_5_0/manifest.json`：固定提交、源码逐文件 SHA256、归档 SHA256、依赖版本和许可证。
- 每个源码构建目录的 `source_manifest.json`：原始来源、适配修改前后哈希、参数哈希、生成源码哈希、未改 Device 源码哈希。
- `build_manifest.json` 的 `python_implementations`：已核验的 Python 编译入口相对 TBE 目录的路径及 SHA256。
- `configure.log`：官方 CMake 和开发依赖检查；`generate.log`：opbuild 及动态编译入口生成；`build.log`：Host 库、parser 与 OPP 打包；`install.log`：隔离安装。
- `atc.log`：必须同时出现参数签名、官方提交签名和 `PARETOPILOT_TILING`，记录实际核数、baseM/N/K、tiling key。缺签名不能进入性能验收。
- `selection.json` 与 `summary.json` 的 `selection`：保留上述实际 tiling、安装产物及 OM 哈希。
- `runtime.log`、`runtime.json`、`profiler_export.log`、CSV：执行、正确性和采样结果。

本地测试覆盖源码完整性、补丁范围、构建调用、测量流程与判定；本地没有 CANN/NPU，不能替代 910B2 真机编译和精度验收。

若要直接研究内置 `BatchMatMulV2` 的实现与可调入口，使用 [安装包实现定位脚本](INSPECT_BATCHMATMULV2.md)，先取得注册、实际编译入口和依赖证据。

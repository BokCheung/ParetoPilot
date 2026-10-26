# ParetoPilot

固定 shape/dtype 的昇腾算子参数搜索工具。目标测试环境为 **Linux、Ascend 910B、CANN 8.5.0**，无需 Agent 接口、API Key 或联网推理。

当前实现：PyTorch 参考输出与 ONNX 导出 → ATC 内置基线 → CANN 8.5.0 官方 MatMulV3 源码默认配置 → Roofline → Dummy Agent 参数域 → 遗传搜索 → 同源候选编译、ATC → PyACL 正确性与 profiling → 三路交错复测 → 报告。Add 保留原有内置 Ascend C 模板。

官方源码来源、参数语义和 910B2 命令见 [官方 MatMul 调优](docs/OFFICIAL_MATMUL.md)。

## 支持范围

| 项目 | 当前支持 |
| --- | --- |
| MatMul | 二维 FP16，M/N/K 为 16 的倍数，ND 输入输出，无转置、bias、batch |
| Elementwise | 两输入同 shape 的 FP16 Add，支持任意正整数 numel 与尾块 |
| 参数 | 官方 MatMul: core_limit、tiling_policy；分块和缓冲由官方 tiler 计算 |
| 参数 | Add: tile_elems、buffers=1/2、block_dim |
| Agent | 默认 DummyAgent，原样返回配置参数域；每代保存 Roofline/历史输入与建议输出 |
| 指标 | 各路 OM 目标任务 Task Duration(us) 中位数 |
| 报告 | HTML、Markdown、Roofline SVG、CSV、JSON、阶段日志、参数/源码/OM 哈希 |

官方 MatMul 源码及构建依赖已随仓库打包，测试机通过 GitHub 拉取即可；Add 使用 msopgen 构建框架。**本地没有 NPU/CANN，新官方源码路径仍需在 910B2 上完成编译、精度和性能验收。**

## 本机运行模拟闭环

Python 3.10+，控制器只依赖标准库。在项目根目录运行：

```bash
python -m paretopilot validate --config examples/simulation_matmul.json
python -m paretopilot run --config examples/simulation_matmul.json --output runs/sim-matmul
python -m paretopilot run --config examples/simulation_add.json --output runs/sim-add
```

打开输出目录的 `report.html`。模拟模式的耗时与正确性结果均为合成数据，不能用于声明硬件收益。输出目录必须为空；已有试验不会被覆盖。

## 搬到昇腾机器

在 Linux 测试机上，用能匹配 PyACL 的 Python 3.10–3.12 执行一键测试：

```bash
python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --soc Ascend910B2 \
  --device 0 --deps check --scope full --timeout 3600
```

按实际安装路径和芯片型号调整参数。上面的命令使用已装好的依赖；脚本自动加载 CANN 环境，然后执行环境检查、单元测试、两套模拟流程、真实基线、smoke 和完整搜索。只做首次试跑可添加 `--scope smoke`；复用已有 Python 依赖可添加 `--deps check`；离线安装使用 `--deps offline --wheelhouse /path/to/wheels`。

结果写入独立的 `runs/910b-*` 目录，包含 `test-summary.json`、阶段日志和 `diagnostics.zip`。任何阶段或最终复测失败都会停止后续阶段；收益不足而保留基线可正常通过。完整选项见 [部署步骤](docs/DEPLOYMENT.md)。

如需通过源码 ZIP 迁移：

```bash
python -m paretopilot package --output dist/ParetoPilot-ascend-910b-cann850.zip
```

ZIP 包含源码、算子模板、示例、一键脚本、依赖清单、文档和测试，不含本机虚拟环境、模拟结果、CANN 或 Python 第三方 wheel。拷贝解压后按 [部署步骤](docs/DEPLOYMENT.md) 执行。无需复制本机的 Windows 虚拟环境。

```bash
python -m paretopilot doctor --config examples/ascend910b_add.json --output environment.json
python -m paretopilot run --config examples/ascend910b_add.json --output runs/add-smoke --smoke
python -m paretopilot run --config examples/ascend910b_matmul.json --output runs/matmul-smoke --smoke
```

官方 MatMul 示例 SoC 为 `Ascend910B2`；一键脚本统一用 `--soc` 覆盖所选算子的配置。Roofline 的算力、带宽只是示例假设，需要替换成该芯片、dtype、稠密口径下的数值。配置通过不代表硬件参数已验证。

## 三维 MatMul 基线

现有二维官方源码调优用例保持不变，另提供两种 FP16 三维基线：

| 用例 | 输入 A | 输入 B | 输出 | 配置 |
| --- | --- | --- | --- | --- |
| 共享权重投影 | `[32,100,128]` | `[128,256]` | `[32,100,256]` | `examples/ascend910b_matmul_shared_rhs.json` |
| 每个请求独立的候选打分 | `[32,1,128]` | `[32,128,100]` | `[32,1,100]` | `examples/ascend910b_matmul_batched_rhs.json` |

在 910B2 上同时运行这两个基线：

```bash
python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann-8.5.0 \
  --soc Ascend910B2 --device 0 --deps check \
  --operator matmul --matmul-case 3d --scope smoke --timeout 3600
```

`--matmul-case` 默认 `2d`，可选 `shared_rhs`、`batched_rhs`、`3d`（两个三维用例）或 `all`（二维加两个三维用例）。三维阶段使用已安装 CANN 的内置实现，完成 ONNX 导出、ATC 编译、真实输出精度检查、预热后 20 次目标任务采样和 Roofline 分析。`test-summary.json` 中结果为 `matmul-shared-rhs-baseline`、`matmul-batched-rhs-baseline`，各阶段目录含 `baseline_summary.json`、`report.html` 和 `roofline.svg`。

这两个配置标为 `implementation.kind=baseline_only`，参数域为空。当前候选适配器只支持二维，因此三维阶段不会调用源码候选或 Agent，也不报告调优收益；`--scope full` 对它们仍只测基线。Roofline 和 Dummy Agent 接口继续保留，后续接入三维候选时再扩展参数域。Roofline 的共享权重流量只计一份，独立批次权重计 batch 份；这是形状下界，不代表实测 DRAM 流量。

配置中的 `operator.batch` 与 `operator.batch_mode` 必须同时设置，模式为 `shared_rhs` 或 `batched_rhs`。即使 batch 为 1，也保留三维输入。默认自动识别实际 MatMul profiling 算子，并严格要求恰好 20 个唯一目标任务样本；识别有歧义时先查看 `profile_inventory.json`，不能把其他任务混入耗时。

## 自定义基线与未来 Agent

- 基线：`examples/baseline_matmul.py` 提供 `build_case(operator, seed)`。输入/输出须保持上述固定语义，候选模板不会自动翻译任意 Python 算子。
- Agent：见 [扩展接口](docs/AGENT.md)。只需替换本地插件 `propose(context)`，保留现有构建和搜索流程。
- 构建与测量：见 [真实后端说明](docs/ADAPTER.md)。
- 搜索与判定：见 [设计](docs/DESIGN.md)。

## 验证

```bash
python -m unittest discover -s tests -v
```

标准库覆盖控制器和模拟闭环；安装 numpy 后运行 PyACL mock 与数值检查；安装 torch/onnx 后额外运行真实 PyTorch→ONNX 导出测试。缺依赖的相关测试会明确 skip。PyACL mock 只验证调用顺序和清理，不证明 NPU kernel 正确或可编译。


## 先识别实际 MatMul 算子，再匹配源码

Ascend MatMul 现在默认先运行内置 baseline，从导出的 profiling CSV 自动识别实际
`Op Name`、`Op Type` 和设备/流/模型身份，再匹配相同 GE 算子类型的源码适配器。
不再用配置中预设的 `BatchMatMulV2` 选择器猜测 baseline 类型。识别仍要求唯一目标
MatMul 任务、恰好 `repeats` 个样本；有多个目标或未知类型时保留 inventory 并报错。

每个 baseline 目录中的 `profile_inventory.json` 记录全部任务，运行根目录的
`routing.json` 记录实际算子、形状、匹配的源码适配器及是否支持调优。
`report.html` 也显示识别结果。Roofline 使用该 shape 的 FLOP/流量分析，只有匹配
源码适配器后才进入 Dummy Agent、参数域搜索及确认测量。

目前适配器支持二维 `MatMulV3` 的固定版本官方源码。只有 baseline 确实选择
`MatMulV3` 时才会构建该源码；候选和复测的 profiling 类型必须一致。
如果测得 `BatchMatMulV2`，则保留其 baseline 并记录 `tuning_supported=false`，
不会替换成 `MatMulV3`。三维源码候选仍未接入。一键脚本遇到未支持的映射后跳过
该用例的 smoke/full 候选阶段，结果标为 `tuning_unavailable`；这表示基线测试完成，
并不表示完成了调优。

算子类型相同不能证明内部 kernel、编译分支或源码版本完全相同。接入安装版
`BatchMatMulV2` 还需要之前的源码/编译入口诊断结果；`routing.json` 会给出
`inspection_argv`，指向刚测量的 baseline。诊断不修改系统 CANN 安装。
如确需复现旧版跨算子比较，可显式设置
`implementation.require_same_operator=false`；默认同算子策略保持开启。

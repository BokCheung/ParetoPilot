# ParetoPilot

固定 shape/dtype 的昇腾算子参数搜索工具。目标测试环境为 **Linux、Ascend 910B、CANN 8.5.0**，无需 Agent 接口、API Key 或联网推理。

当前实现：PyTorch 参考输出与 ONNX 导出 → ATC 基线 OM → Roofline → Dummy Agent 参数域 → 遗传搜索 → 内置 Ascend C 候选编译、独立注册、ATC → PyACL 正确性与 profiling → 交错复测 → 报告。

## 支持范围

| 项目 | 当前支持 |
| --- | --- |
| MatMul | 二维 FP16，M/N/K 为 16 的倍数，ND 输入输出，无转置、bias、batch |
| Elementwise | 两输入同 shape 的 FP16 Add，支持任意正整数 numel 与尾块 |
| 参数 | MatMul: tile_m/n/k、block_dim；buffers 固定 1，由 CANN 管理内部缓冲 |
| 参数 | Add: tile_elems、buffers=1/2、block_dim |
| Agent | 默认 DummyAgent，原样返回配置参数域；每代保存 Roofline/历史输入与建议输出 |
| 指标 | 两侧 OM 目标任务 Task Duration(us) 中位数 |
| 报告 | HTML、Markdown、Roofline SVG、CSV、JSON、阶段日志、参数/源码/OM 哈希 |

这里提供实际 Host/Device 源码和执行链路，不需要自行准备算子工程。msopgen 只生成与本机 CANN 配套的构建框架，ParetoPilot 注入内置实现。**本机没有 NPU/CANN，尚未完成真机编译与性能验收。**

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
  --cann-path /usr/local/Ascend/cann \
  --soc Ascend910B1 \
  --device 0
```

按实际安装路径和芯片型号调整参数。脚本自动加载 CANN 环境、在 `.venv-910b` 安装依赖，然后执行环境检查、单元测试、两套模拟流程、真实基线、smoke 和完整搜索。只做首次试跑可添加 `--scope smoke`；复用已有 Python 依赖可添加 `--deps check`；离线安装使用 `--deps offline --wheelhouse /path/to/wheels`。

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

示例 SoC 为 `Ascend910B1`；必须按测试机实际型号调整。Roofline 的算力、带宽只是示例假设，需要替换成该芯片、dtype、稠密口径下的数值。配置通过不代表硬件参数已验证。

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

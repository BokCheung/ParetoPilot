# 910B / CANN 8.5.0 测试机部署

MatMul 默认采用官方 MatMulV3 源码，910B2 已安装依赖的直接运行命令、双基线字段和日志说明见 [官方 MatMul 调优](OFFICIAL_MATMUL.md)。

## 一键测试（推荐）

前提：Linux、已安装且可用的 910B 驱动、CANN 8.5.0 Toolkit 和 910B ops 包，以及 bash、cmake、g++、make、npu-smi。脚本不安装驱动或 CANN。选择与本机 PyACL 匹配的 Python；自动安装依赖支持 Python 3.10–3.12。

首次拉取并运行：

```bash
git clone https://github.com/BokCheung/ParetoPilot.git
cd ParetoPilot
python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/cann \
  --soc Ascend910B2 \
  --device 0
```

已有仓库先执行 `git pull --ff-only`。`Ascend910B2` 必须替换为实际型号。无需先手动 source CANN：脚本读取指定安装树的版本记录，调用其 set_env.sh，把环境传给后续子进程。无需 Agent API 或 torch_npu。

默认流程：CANN 路径/版本与工具检查 → npu-smi → Python 虚拟环境和依赖 → doctor → 单元测试 → Add/MatMul 模拟闭环 → 两个真实基线 → Add 单核 smoke 与 MatMul 官方默认参数 smoke → 两个完整搜索 → 结果验收与诊断包。首次运行可能需要下载依赖，编译耗时取决于测试机。

```bash
# 只做首次上板试跑，包含前置测试、两侧基线和两个 smoke
python3 scripts/test_910b.py --cann-path /usr/local/Ascend/cann \
  --soc Ascend910B2 --device 0 --scope smoke

# 复用当前 Python 环境：不创建虚拟环境，不安装依赖
/path/to/python3.10 scripts/test_910b.py --cann-path /usr/local/Ascend/cann \
  --soc Ascend910B2 --device 0 --deps check

# 离线安装：wheelhouse 须包含 requirements/910b.txt 和 torch==2.7.1 的全部依赖
python3 scripts/test_910b.py --cann-path /usr/local/Ascend/cann \
  --soc Ascend910B2 --device 0 --deps offline --wheelhouse /data/wheels

# 旧目录布局或独立环境脚本
python3 scripts/test_910b.py \
  --cann-path /usr/local/Ascend/ascend-toolkit/latest \
  --cann-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --soc Ascend910B2 --device 0 --scope smoke

python3 scripts/test_910b.py --help
```

| 参数 | 用途 |
| --- | --- |
| --cann-path | 必填，安装根目录或旧布局的 latest；也接受含 latest 的 ascend-toolkit 目录 |
| --cann-env | 可选，set_env.sh 的显式路径 |
| --soc / --device | 必填具体 SoC；设备 ID 默认 0 |
| --operator all / matmul / add | 默认 all；仅执行所选算子的硬件阶段 |
| --scope full / smoke | 默认 full；smoke 完成全部前置验证后结束，不做完整搜索 |
| --deps install / offline / check | 默认 install；install/offline 使用独立虚拟环境，check 使用启动脚本的解释器 |
| --wheelhouse | offline 必填；不访问包索引，wheels 必须匹配 Linux 架构和 Python ABI |
| --venv | 安装模式的虚拟环境目录，默认项目下 .venv-910b；已有环境须匹配当前 Python 小版本 |
| --output | 新建或空输出目录；省略时自动创建 runs/910b-随机标识 |
| --config-dir | 配置来源目录，默认 examples；需包含 ascend910b_add.json 和 ascend910b_matmul.json |
| --timeout | 每个编译/ATC/运行子进程超时秒数，默认 1800，不是整个测试套件的时限 |

脚本把配置复制到本次输出的 config/，覆盖 SoC、设备、Dummy Agent 和阶段超时；不会修改原始 examples。Add smoke 取各域第一个值并固定 block_dim=1；官方 MatMul smoke 使用 core_limit=0、tiling_policy=0，full 使用原域。硬件峰值和域上限仍须按真实机器填写，示例硬件值不变成实测规格。

`test-summary.json` 的 status=passed 且进程退出码 0 才表示本次范围内的验收通过。脚本拒绝模拟冒充真机、正确性失败、复测数量/顺序不符以及单元测试跳过；即使底层 run 返回 0，也会额外检查全部交错复测。候选没有足够收益而选择 baseline 不影响流程通过。完整搜索允许个别非法/失败候选，但至少一个候选和优胜者的全部复测必须成功。

失败退出码为 2；中断为 130。每阶段都有独立 .log，状态文件记录命令及退出码。成功或失败后自动生成 diagnostics.zip，包含配置、阶段日志、JSON/CSV 和报告；不包含虚拟环境、生成工程、OPP 二进制、OM 和输入张量。完整产物仍在原目录。失败后可按日志修正配置，换新输出目录重跑，不覆盖已有试验，不自动续跑。脚本不保存整个进程环境，避免把环境中的凭据写入诊断包。

自动安装采用 `requirements/910b.txt`，从 CPU 索引安装 torch==2.7.1。依赖约束参照 [CANN 8.5 运行依赖](https://www.hiascend.com/doc_center/source/en/canncommercial/850/softwareinst/instg/instg_0045.html)：Python 3.13 以下使用 NumPy 1.x、Protobuf 3.20.x。[CANN 8.5 路径变化说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/releasenote/releasenote_0000.html)介绍新路径及旧路径兼容性。以下为需要逐步排查时的手动命令。

## 1. 准备环境

使用 Linux 测试机。需要匹配设备驱动的 CANN 8.5.0 Toolkit、910B 算子包、Ascend C 编译器、msopgen、msprof、ATC、PyACL，以及 bash、g++、CMake。只装运行库不足以编译候选。

解压源代码包，在其 ParetoPilot 根目录操作。按实际安装路径加载环境，例如：

```bash
source /usr/local/Ascend/cann/set_env.sh
npu-smi info
python --version
which atc msopgen msprof cmake g++
```

Python 要求 3.10+，且须与已安装的 PyACL ABI 兼容。保留 CANN 设置的 PYTHONPATH/LD_LIBRARY_PATH。不要从 PyPI 安装同名 acl 包来替代 CANN 的 PyACL。

如 msopgen 未在 PATH 中，根据安装位置添加其目录，通常在 Toolkit 的 python/site-packages/bin，或在配置 ascend.msopgen 中填写绝对可执行文件路径。atc/msprof 也支持同名配置覆盖。

有网络时，在测试机对应架构的 Python 环境安装导出依赖：

```bash
python -m pip install -r requirements/910b.txt
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
```

PyTorch 只在 CPU 上生成参考输出和 ONNX，不要求 torch_npu。离线机器需提前准备与 Linux 架构、Python 版本匹配的 wheels，再执行：

```bash
python -m pip install --no-index --find-links /path/to/wheels \
  -r requirements/910b.txt 'torch==2.7.1'
```

源码目录可直接使用 python -m paretopilot，不必安装本项目。若需要命令 paretopilot，可用 pip install --no-build-isolation --no-deps .（环境须已有 setuptools>=68）。

## 2. 修改配置

先用 examples/ascend910b_add.json，再用 examples/ascend910b_matmul.json。

- ascend.soc_version：按 npu-smi 的实际芯片型号设置，示例 Ascend910B2 不代表所有 910B。
- ascend.device：可用设备 ID。
- ascend.cann_version：8.5.0；source_tag 记录模板版本。
- hardware.max_blocks：Add 对应 Vector 核数，MatMul 对应 Cube 核数。Host 端还会查询实际资源进行拒绝校验。
- hardware.peak_flops / bandwidth_bytes_s：必须与对应算子执行单元、FP16、稠密/稀疏口径匹配。示例值仅用于演示 Roofline。
- hardware.local_memory_bytes：控制器粗筛预算，不等同于 MatMul 各级 L0/L1/UB 总容量。最终由 CANN Tiling API 检验。
- measurement.selectors：profiling 的精确任务筛选；基线示例使用 Add / BatchMatMulV2，Add 候选使用 PpAdd，官方 MatMul 默认源码与候选均使用 MatMulV3。MatMul 基线类型依据 910B / CANN 8.5.0 测试机的实际 CSV；其他环境导出名称不同应先查 CSV 再调整。
- measurement.timeout_s：每个编译、ATC 或运行进程的超时，默认 900 秒。
- search.budget：仅搜索候选数量，不含基线和最终复测。失败候选也计数。

## 3. 环境检查与试跑

```bash
python -m paretopilot doctor --config examples/ascend910b_add.json --output environment.json
python -m paretopilot validate --config examples/ascend910b_add.json
python -m paretopilot baseline --config examples/ascend910b_add.json --output runs/add-baseline
python -m paretopilot run --config examples/ascend910b_add.json --output runs/add-smoke --smoke
python -m paretopilot run --config examples/ascend910b_matmul.json --output runs/matmul-smoke --smoke
```

doctor 只检查软件依赖、Toolkit 版本文件并展示 npu-smi；仍须核对实际 SoC。版本无法读取时，设置 ASCEND_HOME_PATH 指向已加载的 8.5.0 Toolkit 根目录，其中应存在 version.info 或架构目录下的 ascend_toolkit_install.info。真实运行也会执行并保存同一环境检查。

smoke 只搜索一个候选、做两轮交错复测，成功后再扩预算。每次使用新输出目录。该阶段最坏需要 6 次构建/转换/运行（基线、搜索候选、两组复测），首次 ATC 编译可能较慢。

所有候选失败时退出码为 2，详细错误在 trials/*/trial.json。没有足够收益而保留基线属于正常成功，退出码为 0。

## 4. profiling 目标名称不匹配

`op_summary` 的列名兼容 `OP Type` / `Op Type`、`Stream ID` / `Stream Id`、`Model ID` / `Model Id` 和设备 ID 的已知写法，原配置可继续使用。算子名称和值仍精确匹配，`Task Duration(us)` 仍为微秒耗时；不会用 aicore_time/aiv_time 等不同指标代替。跨 stream/model/device 的混合采样仍会拒绝。

若收到 `Profiling CSV missing required columns`，错误会同时列出文件路径、缺失字段和实际表头。先用下面的 inspect-profile 查看 `columns` 原始表头及 `tasks` 标准化后的标识。不要修改原 CSV 来绕过校验；重复或冲突的列名也会拒绝。

若收到 Expected exactly ... target task samples，先找到日志中该试验的 profiler/ 下 op_summary*.csv。可执行：

```bash
python -m paretopilot inspect-profile /absolute/path/op_summary.csv
python -m paretopilot import-profile --config examples/ascend910b_add.json --role baseline /absolute/path/op_summary.csv
```

inspect-profile 列出实际 Op Name、Op Type、设备、stream 等及样本数。将准确字段写入 measurement.selectors.baseline/candidate；例如同时设置 Op Type 与 Op Name 来消除歧义。不要通过减少样本或选择另一算子来掩盖拆分/融合问题。必须恰好为 measurement.repeats 条目标任务记录，预热由程序在 profiler.start 之前执行。

已观察到测试机的 256×256 FP16 MatMul 基线导出为 `Op Name=/MatMul`、`OP Type=BatchMatMulV2`。旧配置中的 `MatMulV2` 会导致 `got 0`，因此真实 MatMul 示例已改为 `BatchMatMulV2`，仍要求 20 条记录。更新后可用 import-profile 和新配置检查保留的完整 CSV，无需重新编译或执行 NPU；一键脚本下一次运行会从 examples 复制新配置，已有试验目录中的 manifest/config 保留原值。

可选 measurement.traffic_column 必须为同一目标任务每次调用的**总字节数**，不能直接使用 GB/s 带宽列；未配置时 Roofline 用 shape 最低流量并标明理论口径。

## 5. 完整搜索与交付结果

```bash
python -m paretopilot run --config examples/ascend910b_matmul.json --output runs/matmul-full
```

- report.html / report.md / roofline.svg：报告与图。
- summary.json / best_params.json：选择结果；回退基线时 params=null。
- baseline/ 与 trials/：OM、构建源码、OPP、原始 CSV、正确性误差、命令与日志。
- confirmation/：优胜候选和基线重新构建后交错测量。
- agent/：每代 Roofline、历史数据与 Dummy 的原域输出。
- case/：固定输入 .npy、PyTorch 参考结果、原始 ONNX。
- environment.json / manifest.json：环境及配置、源码标识。

候选 OPP 只装入试验自己的 opp/vendors/，不安装到系统默认算子目录。基线编译和运行会清除继承的 ASCEND_CUSTOM_OPP_PATH。报告中的收益是“自定义实现加参数”相对 stock OM 的收益，不能全部解释为相同实现的调参收益。

当前没有自动续跑；中断或失败后保留已完成试验，请换新目录运行。重跑配置中的相同 seed 会生成相同输入与搜索随机序列，但硬件耗时仍有波动。

## 常见故障定位

若 MatMul 的 `build.log` 在 `opbuild` 阶段报 `AscendC::TPosition has not been declared`，原因是旧 Host 模板混用了 Device 的枚举命名空间。Host 的 `SetAType/SetBType/SetCType` 已改为 `matmul_tiling::TPosition::GM`，格式和数据类型也显式使用 `matmul_tiling`。随后出现的 `Makefile: No such file or directory` 是前面配置失败的连带结果。通过 GitHub 更新并创建新试验重新构建即可，不需要安装另一个编译器或修改生成目录。

若 ATC 日志已打印 `PARETOPILOT_PARAMS=...`，随后在 `get_v220_kernel_type_mix_flag` 中报 `tiling_key_kernel_type[tiling_key]` / `KeyError: '1'`，对应本项目旧模板的 Host/Device key 不一致：Host 选择了 key 1，Device 未声明对应分支。旧 bundled Add、MatMul 模板均已补齐 `TILING_KEY_IS(1)`。通过 GitHub 更新后新建试验，重新生成、编译和安装候选；旧 runs 中的工程和 OPP 不会自动更新。测试机可执行：

```bash
git pull --ff-only
python3 scripts/test_910b.py --cann-path /usr/local/Ascend/cann-8.5.0 --soc Ascend910B2 --device 0 --deps check --scope smoke
```

日志中的 Toolkit owner 不一致警告与上述 key 查表异常不同；该错误应先修正 kernel 分支，不需要因此修改 Toolkit 文件所有者。

旧版本的 `Expected one generated ONNX parser; found []` 可能是只查找 .cpp，漏掉了 msopgen 生成的 framework/onnx_plugin/pp_add_plugin.cc。当前版本已支持 .cc/.cpp/.cxx，并保留原生成文件和 CMake 引用。通过 GitHub 更新项目后，使用 `--soc Ascend910B2 --deps check --scope smoke` 创建新试验验证；无需手工重命名生成文件。实际 Ascend C 编译仍须在测试机确认。

| 阶段 | 文件 | 检查方向 |
| --- | --- | --- |
| 导出 | case/export.log | Python/PyTorch/ONNX 兼容性，自定义 build_case |
| 工程生成 | trials/*/generate.log | msopgen、SoC、生成工程布局 |
| 编译/安装 | build.log / install.log | CANN 头文件/编译器、CMake、生成的源码 |
| ATC | atc.log / selection.json | ONNX parser、OPP、参数签名、Tiling 资源约束 |
| 正确性/执行 | runtime.log / correctness.json | PyACL ABI、设备 ID、ND 布局、容差、kernel 输出 |
| 导出采样 | profiler_export.log | msprof、权限、生成的 CSV |
| 汇总 | trial.json | 名称歧义、数量不符、非有限耗时 |

本地验证不包含昇腾编译、实际 OPP 装载、设备正确性和加速收益。保留首次真机 smoke 的完整目录，可直接定位失败阶段，不需要 Agent 服务。

# 真实昇腾工程适配器协议 v1

配置 `adapter` 为可信程序 argv 数组。控制器通过 subprocess、shell=False 调用，在末尾附加绝对路径 `<stage.request.json> <stage.response.json>`。当前目录是本次试验目录。配置相对路径相对于配置文件，`{python}` 使用当前解释器。

请求字段：protocol_version=1、stage、role（baseline/candidate）、params、params_hash、完整 config、workdir。阶段间使用 workdir 共享产物。路径响应可为绝对路径或相对于 workdir。

成功响应共同字段：status="ok"、mode 与配置一致；失败返回非零退出码或失败 status。stdout/stderr 进入阶段日志。timeout_s 限制单阶段；适配器负责管理和清理编译器/采集器子进程。

| stage | 执行内容 | 响应字段 |
| --- | --- | --- |
| build（仅 candidate） | 注入参数，编译匹配 CANN 的 Host/Device 源码，独立 OPP 安装 | 共同字段，保留构建产物与日志 |
| export | baseline 导出原始 PyTorch ONNX；candidate 导出自定义算子图。校验描述与实际 shape/dtype/布局一致 | 共同字段，保留 ONNX 与输入记录 |
| atc | baseline 使用原始环境；candidate 设置独立 ASCEND_CUSTOM_OPP_PATH，完成注册和图适配 | om_path；candidate 另需 selected_params_hash、selected_source_sha256、implementation_id、selection_evidence 文件路径 |
| verify | 执行该 OM，按 atol/rtol 与 PyTorch 参照比较 | passed=true、reference="pytorch"、om_sha256；归档误差/种子/输入摘要 |
| profile | 执行该 OM，预热后多次采集目标任务 | csv_path、om_sha256、warmup；可选 measured_bytes（与时间样本统计口径一致的目标任务实测流量） |

`params_hash` 使用 `paretopilot.artifacts.fingerprint`；源码和 OM 使用 SHA256。selection_evidence 应是实际构建/ATC 选中记录，不能只生成一句“已选中”。控制器验证证据存在并记录哈希；适配器负责解释证据、确认候选生效。

profiling.target_name 精确匹配 name_column，duration_column 单位必须为微秒。CSV 至少包含 repeats 条目标任务样本；必须由适配器排除预热，拒绝歧义融合节点、意外拆分、跨设备/跨 stream 混样。baseline/candidate 输入数据、时钟/电源策略、运行和采集条件一致。

先用 `scripts/demo_adapter.py` 理解请求和响应，再实现 `adapters/ascend_driver.py`。真实环境需提供匹配版本的 PyTorch/ONNX/ATC、算子源码、注册与自定义图适配、OM runner 和 profiling 工具。示例驱动没有真实实现，未配置时明确返回失败，避免误把模拟当作真机。

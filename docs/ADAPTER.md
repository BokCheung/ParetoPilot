# CANN 8.5.0 真实后端

旧版外部 stage 请求/响应适配器占位协议已由内置后端替换。用户不需要实现 adapter 命令。入口为 backend.AscendBackend，重依赖和设备状态隔离在 worker 子进程。

## 模块与阶段

| 阶段 | 实现 | 产物 |
| --- | --- | --- |
| 导出参考 | worker.export_case | 输入 .npy、reference.npy、baseline.onnx、case.json |
| 工程/编译 | build.build_candidate + kernels/ | msopgen 工程、参数头、ONNX parser、.run、独立 OPP |
| ATC | backend.AscendBackend.evaluate | model.om、atc.log、selection.json |
| 校验/采样 | worker.execute_om | correctness.json、output.npy、runtime.json、profiling 原始数据 |
| CSV 导出 | msprof --export=on | op_summary*.csv |
| 统计 | profiling.read_profile | 严格筛选的样本与中位数 |
| 复测/报告 | search + report | confirmation、summary、best_params、HTML/SVG/CSV |

基线和候选使用相同固定输入与 PyTorch 参考输出。基线去掉 ASCEND_CUSTOM_OPP_PATH，候选只指向当前试验 opp/vendors/ 下的独立安装路径，不改系统默认 OPP。

## 模板

Add：UB 队列、DataCopyPad 处理非对齐尾块，Vector Add；tile_elems/buffers/block_dim 编译期注入。Host 查询 UB 与 AIV 核数。

MatMul：CANN Matmul 高阶 API，Host 设置原始矩阵步长、单块 shape、base tile；Device 按二维网格分配 tile，处理边缘 tile 并写回原始输出布局。Host 查询 AIC 核数，CANN API 校验 tiling 存储约束。buffers 固定 1，因为该版本不暴露受支持的手动 SetDoubleBuffer 开关。

msopgen 生成与安装版本匹配的框架。要求 op_host、op_kernel、framework、CMakePresets.json 和 build.sh 布局；不兼容的生成器布局会明确失败，保留工程用于检查。

ONNX parser 在 framework/onnx_plugin 或兼容目录 framework/onnx 中定位，接受 .cc、.cpp、.cxx；保留生成文件的路径、扩展名及原 CMakeLists.txt，在原文件中替换注册实现。910B2 / CANN 8.5.0 测试机实际生成 framework/onnx_plugin/pp_add_plugin.cc。source_manifest.json 记录选中的 onnx_parser 相对路径；零个或多个候选仍报错，并列出生成的 framework 文件，避免误改其他框架的插件。

## 原生运行要求

PyACL 使用 ACL_HOST 模式，两个输入、一个输出，外部格式为 ND。支持 ND 缓冲区末尾 padding，不支持 blocked 外部布局。每次复制/执行检查返回码，正确性在 profiler.start 之前和 stop 之后检查。

正确性比较要求 shape、dtype 匹配，输出有限，满足 numpy.allclose 的 rtol/atol。不能通过调整 CSV 或耗时报告让错误输出参与排名。

Profiler 在模型加载之前初始化，在预热之后启动。每个候选使用独立进程和目录，Linux 超时会终止该阶段进程组。失败日志、命令和已完成结果保留。

## 扩展

增加算子需要同时更新固定语义描述、PyTorch 构造、候选 ONNX、Host/Device 模板、参数域和合法性检查。不能仅把 operator.kind 改成任意 Elementwise 名称。

增加 Agent 只需本地 Python 插件，不需要改构建与运行部分。详见 AGENT.md。

## 官方接口参考

- [CANN 8.5 MatMul 集成](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0038.html)
- [CANN 8.5 Matmul Tiling API](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0671.html)
- [msopgen 工程生成与打包](https://www.hiascend.com/document/detail/en/mindstudio/2610/optools/Operatordevelopmenttools/docs/en/user_guide/msopgen_user_guide.md)
- [PyACL profiling 流程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0049.html)

后两项为工具/API 用法参考，不代表已在该文档版本或本项目目标硬件上验收。

# Roofline → Agent 参数域接口

默认 agent.kind="dummy"。每代搜索先构造 context，包含 schema_version、generation、operator、hardware、original_space、baseline、best、roofline 和完整 history。

官方 MatMul 模式下，context.baseline 指向 source_baseline（官方源码默认参数），context.stock_baseline 保留 CANN 内置对照；Roofline 来自同源基线或最佳候选。参数名为 core_limit、tiling_policy，默认值 0 是有效参数。

DummyAgent 原样复制 original_space，返回空 priority；不解释 Roofline，不调用网络，也不会收缩或扩展参数域。Roofline 的瓶颈假设和 suggested_priority 仍保存并传入，供未来 Agent 使用。

```json
{
  "agent": {
    "kind": "python",
    "path": "local_agent.py",
    "options": {}
  }
}
```

path 相对于配置文件。模块提供 create_agent(options)，返回实现 propose(context) 的对象：

```python
from copy import deepcopy

class Advisor:
    def propose(self, context):
        return {
            "space": deepcopy(context["original_space"]),
            "priority": [],
            "reason": "Retain configured domains"
        }

def create_agent(options):
    return Advisor()
```

返回参数名须与原域完全相同，值必须是原域内非空、无重复的整数列表；priority 只能使用已有参数名。插件收到 context 的深复制，不能通过修改输入改变控制器状态。非法提议会终止试验并报错，不静默吞掉。

Agent 可以在原域内收缩或重新放宽；若要新增原域外参数值，先更新配置和模板能力。合法性过滤仍由控制器与 Host tiling 执行。遗传搜索每代从本次提议域注入随机未测配置；Dummy 情况下即原始完整域。

每代 input/output 落盘于 agent/。未来可将真实 Agent 服务封装到这个本地插件，替换决策部分；当前部署完全不需要该服务。插件是可信 Python 代码，可访问所在机器，不是隔离脚本。

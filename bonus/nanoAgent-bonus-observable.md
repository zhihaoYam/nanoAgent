# 从零开始理解 Agent（番外篇）：Agent 出了问题怎么排查？

> **「从零开始理解 Agent」系列番外** —— 前面的番外讲了怎么看 Token 花在哪（Token 篇）、怎么判断任务做没做完（Eval 篇）、怎么让用户不干等（流式篇）。但有一个问题一直没解决：Agent 跑完了，结果不对，你怎么知道它在第几轮走偏的？

-----

## 一、Agent 的调试困境

传统程序出 bug，看堆栈、打断点、单步执行，问题通常能定位到某一行代码。

Agent 不一样。Agent 的"bug"往往不在代码里——代码逻辑没问题，工具执行也没报错，但 LLM 在第 3 轮做了一个错误的推理判断，后续所有步骤都基于这个错误继续执行，最终给出一个看似合理但实际错误的结果。

举个例子：你让 Agent "找到项目中所有未使用的依赖并删除"。Agent 执行了 5 轮，最后说"已删除 3 个未使用的依赖"。但你发现它删错了一个——那个依赖在测试文件中被间接引用了。

问题出在哪一轮？是 `grep` 的搜索范围漏了测试目录？还是 LLM 看到了 grep 结果但推理错了？你不知道，因为 Agent 只给了你最终结果，中间过程是黑盒。

这就是 Agent 的调试困境：**问题不在代码里，在对话流里。**

-----

## 二、最小可用的 Log：给每轮循环打日志

第一步是把 Agent 的每轮循环记录下来。回顾第一篇的核心循环——我们只需要在循环中把每轮的关键信息写入日志文件：

```python
import json
import time

class AgentLogger:
    """记录 Agent 每轮循环的关键信息"""

    def __init__(self, log_file="agent.log.jsonl"):
        self.log_file = log_file
        self.task_id = str(int(time.time()))

    def log_round(self, round_num, messages_snapshot, llm_response,
                  tool_name=None, tool_args=None, tool_result=None,
                  usage=None, duration_ms=None):
        """记录一轮循环"""
        entry = {
            "task_id": self.task_id,
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message_count": len(messages_snapshot),
            "llm_content": llm_response.get("content", ""),
            "has_tool_calls": tool_name is not None,
        }

        if tool_name:
            entry["tool"] = {
                "name": tool_name,
                "args": tool_args,
                "result_length": len(tool_result) if tool_result else 0,
                "result_preview": tool_result[:200] if tool_result else ""
            }

        if usage:
            entry["tokens"] = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens
            }

        if duration_ms:
            entry["duration_ms"] = duration_ms

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

嵌入 Agent 循环：

```python
logger = AgentLogger()

for i in range(max_iterations):
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )
    message = response.choices[0].message
    duration_ms = int((time.time() - start) * 1000)

    if not message.tool_calls:
        logger.log_round(i + 1, messages, {"content": message.content},
                         usage=response.usage, duration_ms=duration_ms)
        return message.content

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        function_response = available_functions[function_name](**function_args)

        logger.log_round(
            i + 1, messages,
            {"content": message.content or ""},
            tool_name=function_name,
            tool_args=function_args,
            tool_result=function_response,
            usage=response.usage,
            duration_ms=duration_ms
        )
        # ... 正常放回 messages ...
```

日志用 JSON Lines 格式（每行一个 JSON 对象），方便用 `jq` 或 Python 解析。一次 Agent 任务的日志长这样：

```jsonl
{"task_id":"1712345678","round":1,"timestamp":"2026-04-06T10:30:01","message_count":2,"llm_content":"","has_tool_calls":true,"tool":{"name":"execute_bash","args":{"command":"find . -name '*.py'"},"result_length":342,"result_preview":"./src/main.py\n./src/utils.py\n..."},"tokens":{"input":523,"output":87},"duration_ms":2100}
{"task_id":"1712345678","round":2,"timestamp":"2026-04-06T10:30:04","message_count":5,"llm_content":"","has_tool_calls":true,"tool":{"name":"execute_bash","args":{"command":"grep -r 'import requests' ."},"result_length":0,"result_preview":""},"tokens":{"input":1204,"output":103},"duration_ms":1800}
{"task_id":"1712345678","round":3,"timestamp":"2026-04-06T10:30:06","message_count":8,"llm_content":"项目中没有使用 requests 库...","has_tool_calls":false,"tokens":{"input":1850,"output":156},"duration_ms":3200}
```

出问题时，打开日志文件，逐行看：第几轮、调了什么工具、传了什么参数、返回了什么结果、LLM 基于结果说了什么。通常几分钟就能定位到"走偏"的那一轮。

-----

## 三、Trace：把对话流串成一条链

Log 是散的——每行是一个独立的事件。当 Agent 任务变复杂（十几轮循环、多次工具调用），逐行看日志会很吃力。

更好的方式是用 Trace 的视角来组织：一次 Agent 任务 = 一条 Trace，每轮循环 = 一个 Span。

如果你有运维背景，这个概念很熟悉——微服务调用链里，一次用户请求是一条 Trace，经过的每个服务是一个 Span。Agent 的 Trace 和它结构上完全一样：

```
Trace: "找到未使用的依赖并删除"
│
├── Span 1: LLM 推理 → 调用 execute_bash("find . -name '*.py'")
│   ├── duration: 2100ms
│   ├── tokens: {input: 523, output: 87}
│   └── tool_result: "./src/main.py\n./src/utils.py\n..."
│
├── Span 2: LLM 推理 → 调用 execute_bash("grep -r 'import requests' .")
│   ├── duration: 1800ms
│   ├── tokens: {input: 1204, output: 103}
│   └── tool_result: ""  ← grep 没找到，返回空
│
├── Span 3: LLM 推理 → "没有使用 requests"  ← 问题在这里！
│   ├── duration: 3200ms                        grep 只搜了 import，
│   └── tokens: {input: 1850, output: 156}      没搜 from requests import
│
└── ...
```

用代码表达，不需要引入外部框架：

```python
class Trace:
    """一次 Agent 任务的完整追踪"""

    def __init__(self, task):
        self.trace_id = str(int(time.time()))
        self.task = task
        self.spans = []

    def add_span(self, round_num, llm_content, tool_name=None,
                 tool_args=None, tool_result=None,
                 duration_ms=None, tokens=None):
        span = {
            "span_id": f"{self.trace_id}-{round_num}",
            "round": round_num,
            "llm_content": llm_content,
            "tool": None,
            "duration_ms": duration_ms,
            "tokens": tokens
        }
        if tool_name:
            span["tool"] = {
                "name": tool_name,
                "args": tool_args,
                "result_preview": tool_result[:200] if tool_result else ""
            }
        self.spans.append(span)

    def dump(self):
        """输出完整 Trace"""
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "total_rounds": len(self.spans),
            "total_duration_ms": sum(s["duration_ms"] or 0 for s in self.spans),
            "total_tokens": sum(
                (s["tokens"]["input"] + s["tokens"]["output"])
                for s in self.spans if s["tokens"]
            ),
            "spans": self.spans
        }
```

Trace 和 Log 记录的信息是一样的，区别在于组织方式：Log 是按时间排列的事件流，Trace 是按因果关系组织的调用链。排查问题时，Trace 让你一眼看到整条执行路径，而不是在散乱的日志行里拼凑上下文。

-----

## 四、SubAgent 的 Trace：从链到树

如果只是单 Agent 循环，Trace 是一条链（Span 1 → Span 2 → Span 3）。但回忆第四篇——SubAgent 来了之后，Trace 变成了树。

主 Agent 在第 3 轮决定调用 SubAgent，SubAgent 内部又跑了自己的循环（3 轮），然后把结果返回给主 Agent。Trace 结构变成：

```
Trace: "重构项目并更新文档"
│
├── Span 1: 主 Agent → execute_bash("ls src/")
├── Span 2: 主 Agent → execute_bash("cat src/main.py")
├── Span 3: 主 Agent → 调用 SubAgent("重构 main.py")
│   │
│   ├── Span 3.1: SubAgent → read_file("src/main.py")
│   ├── Span 3.2: SubAgent → write_file("src/main.py", ...)
│   └── Span 3.3: SubAgent → execute_bash("python -m pytest")
│
├── Span 4: 主 Agent → 调用 SubAgent("更新 README")
│   │
│   ├── Span 4.1: SubAgent → read_file("README.md")
│   └── Span 4.2: SubAgent → write_file("README.md", ...)
│
└── Span 5: 主 Agent → "重构完成，文档已更新"
```

和微服务的 parent span / child span 完全对应。实现上只需要给 SubAgent 的 Trace 传入 parent span id：

```python
def run_sub_agent(task, parent_span_id=None):
    sub_trace = Trace(task)
    sub_trace.parent_span_id = parent_span_id
    # ... SubAgent 的循环，每轮往 sub_trace 加 span ...
    return sub_trace
```

排查 SubAgent 的问题时，你能看到：主 Agent 为什么决定调用 SubAgent（Span 3 的 llm_content）、SubAgent 内部做了什么（Span 3.1-3.3）、返回结果是什么。整条因果链清清楚楚。

-----

## 五、Replay：用日志复现问题

Log 和 Trace 告诉你"发生了什么"。但有时候你想验证一个假设："如果第 2 轮的 grep 命令换一种写法，LLM 后面还会推理错吗？"

这就需要 Replay——拿着某一轮的 messages 快照重新跑一次。

前提是在日志中保存每轮发送给 LLM 的完整 messages：

```python
def log_round_with_messages(self, round_num, messages, ...):
    entry = {
        ...
        "messages_snapshot": messages.copy()  # 保存完整 messages
    }
```

有了 messages 快照，Replay 就是把它喂回 LLM：

```python
def replay_round(log_entry, temperature=0):
    """用保存的 messages 重新跑一轮，看 LLM 是否给出相同的结果"""
    messages = log_entry["messages_snapshot"]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=temperature  # 设为 0 提高可复现性
    )

    return response.choices[0].message
```

Replay 的几个用途：

**复现问题。** 把出问题那一轮的 messages 喂回去，看 LLM 是不是还会做同样的错误判断。如果 `temperature=0` 下结果一致，说明问题是确定性的（prompt 或上下文导致的），不是随机波动。

**测试修复。** 修改了 system prompt 或工具描述后，把同一份 messages 喂回去，看 LLM 是否改正了判断。不需要重新跑整个任务，只重放出问题的那一轮。

**和 Eval 篇的关系。** Eval 篇讲的是"批量验证 Agent 能力"，Replay 是"针对单次问题的定点调试"。Eval 是体检，Replay 是做 CT 查某个具体部位。

需要注意的是，messages 快照可能很大（包含所有历史对话和工具返回结果），不适合每轮都保存到日志文件。实际做法是：正常运行时只记录轻量日志（第二节的 AgentLogger），出问题后开启"详细模式"重跑一次，这次保存完整 messages。

-----

## 六、生产级 Agent 的可观测性

nanoAgent 的 AgentLogger 和 Trace 是最简实现。生产环境中，Agent 的可观测性通常围绕三个维度展开——如果你有云原生背景，这就是 metrics / logs / traces 三支柱在 Agent 场景的对应：

**Metrics（指标）：** 不是看单次任务，而是看趋势。关注的典型指标包括：任务成功率（多少任务顺利完成 vs 中途失败）、平均轮次（任务越来越多轮可能意味着 prompt 退化）、Token 消耗趋势（呼应 Token 番外篇）、工具调用失败率（某个工具频繁报错说明工具本身有问题）、压缩触发频率（呼应第六篇，频繁压缩说明任务普遍偏长）。

**Logs（日志）：** 就是本篇第二节的 AgentLogger。生产环境中通常会接入集中式日志系统，方便按 task_id 筛选、按时间范围查询、按工具名聚合。

**Traces（链路）：** 就是本篇第三四节的 Trace。生产环境中通常会接入分布式追踪系统，可视化展示每条 Trace 的时间线、各 Span 的耗时占比、SubAgent 的调用深度。

三个维度各解决一类问题：Metrics 回答"系统整体健不健康"，Logs 回答"某次任务发生了什么"，Traces 回答"某次任务的每一步是怎么串起来的"。

-----

## 七、小结

回顾一下这篇番外加了什么：

|层次              |解决什么问题               |代码量               |
|----------------|---------------------|------------------|
|Log（AgentLogger）|出问题时能看到每轮做了什么        |约 30 行            |
|Trace           |把散乱的日志串成因果链          |约 30 行            |
|SubAgent Trace  |多 Agent 场景下的树状追踪     |加一个 parent_span_id|
|Replay          |用保存的 messages 复现和验证问题|约 10 行            |

回到 Harness 番外篇的视角：可观测性是 Harness 的"后视镜"。Token 追踪告诉你花了多少，Eval 告诉你做得对不对，流式输出让用户看到当前在干什么，而可观测性让你看到过去发生了什么。四者合在一起，Agent 的运行才是透明的。

-----

*「从零开始理解 Agent」系列番外。前面的番外让 Agent 跑得更好，这篇让 Agent 出了问题时你能查得清楚。*

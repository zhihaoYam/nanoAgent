# 从零开始理解大模型（十）：从大模型到 Agent——下一个词预测如何长出手脚

> **「从零开始理解大模型」系列** —— 十篇文章，从”下一个词预测”到完整的大模型心智模型。每篇配可运行代码。
>
> - [第一篇：一切从”猜下一个词”开始](./llm-01-next-token.md)
> - [第二篇：Token——大模型眼中的”字”长什么样](./llm-02-token.md)
> - [第三篇：向量与 Embedding——把文字变成数学](./llm-03-embedding.md)
> - [第四篇：Attention——大模型的”阅读理解”机制](./llm-04-attention.md)
> - [第五篇：Transformer 全景——积木怎么搭成大厦](./llm-05-transformer.md)
> - [第六篇：训练——70 亿个参数是怎么”学”出来的](./llm-06-training.md)
> - [第七篇：推理——你按下回车后的这一秒发生了什么](./llm-07-inference.md)
> - [第八篇：上下文窗口——大模型的”工作记忆”有多大](./llm-08-context-window.md)
> - [第九篇：Scaling Law——为什么”大力出奇迹”有效](./llm-09-scaling-law.md)
> - **第十篇：从大模型到 Agent——下一个词预测如何长出手脚**（本文）

前九篇我们把大模型从里到外拆了个遍：Token、Embedding、Attention、FFN、训练、推理、上下文窗口、Scaling Law。

但大模型有一个根本的限制：**它只能生成文本。**

你问它”今天天气怎么样”，它没法真的去查天气。你让它”帮我创建一个文件”，它没法真的操作文件系统。它只能输出一串 token——至于这串 token 能不能变成真实的行动，那不是它能管的事。

那问题来了：ChatGPT 明明能搜索网页、Claude 能写文件、各种 AI Agent 能自主完成复杂任务——它们是怎么做到的？

答案是：**大模型本身没变，变的是它外面包的那一层。**

-----

## 一、先说结论

|你以为的|实际的|
|---|---|
|Agent 是一种新型 AI|Agent = 普通大模型 + 工具 + 循环|
|大模型能”执行”操作|大模型只输出文本，外部代码负责执行|
|Function Calling 是特殊能力|本质就是让模型输出一段特定格式的 JSON|
|Agent 有自主意识|Agent 的每一步”决策”都是一次”预测下一个词”|

一句话版本：**大模型负责”想”（输出文本），外部代码负责”做”（执行操作）。Agent 就是用一个循环把这两者串起来。**

-----

## 二、从”生成文本”到”调用工具”

大模型只会一件事：给定前文，预测下一个 token（第一篇）。它怎么”调用工具”的？

### 2.1 Function Calling 的本质

当你对 ChatGPT 说”今天北京天气怎么样”，模型内部做的事情是这样的：

```json
{
  "function": "search_weather",
  "arguments": {"city": "北京"}
}
```

模型没有”调用”任何函数。它只是生成了一段 JSON 格式的文本——**“我觉得应该调用 search_weather，参数是北京”**。

真正执行这个调用的是模型外面的代码：

```python
# 模型输出了一段 JSON
tool_call = {"function": "search_weather", "arguments": {"city": "北京"}}

# 外部代码真正执行
result = search_weather(city="北京")   # → "晴，26°C"

# 把结果塞回给模型，让它继续生成
messages.append({"role": "tool", "content": "晴，26°C"})
```

**模型做的是”决策”——决定调哪个工具、传什么参数。代码做的是”执行”——真正去调 API、读文件、跑命令。**

这就是第一篇到第五篇讲的所有东西在实际中的应用：模型把 “今天北京天气怎么样” 分词成 token（第二篇），查 Embedding 表变成向量（第三篇），经过 Attention 理解上下文（第四篇），经过多层 Transformer 处理（第五篇），最终输出的 token 恰好拼成了一段 JSON。

**Function Calling 不是什么新能力，它就是”预测下一个词”——只不过预测出来的词恰好构成了一个函数调用的 JSON。**

### 2.2 模型怎么知道要输出 JSON

靠训练（第六篇）。SFT 阶段会用大量这样的训练数据：

```text
用户: 帮我查一下北京的天气
助手: {"function": "search_weather", "arguments": {"city": "北京"}}
工具返回: 晴，26°C
助手: 北京今天天气晴，气温 26°C。
```

模型在训练中看过无数这样的例子，所以它学会了：看到某些类型的请求时，应该输出 JSON 格式的工具调用；看到工具返回结果时，应该把结果整合成自然语言回答。

**这和 “Thank you very → much” 是完全一样的机制——统计规律。** 模型见过足够多的例子，学会了”这种上下文后面应该接工具调用”。

-----

## 三、Agent = LLM + 工具 + 循环

理解了 Function Calling，Agent 的架构就很清楚了。

### 3.1 最小的 Agent

```python
def run_agent(user_message, max_steps=10):
    messages = [
        {"role": "system", "content": "你是一个能使用工具的助手。"},
        {"role": "user", "content": user_message}
    ]

    for step in range(max_steps):
        # 1. 让大模型“想”：预测下一个 token（可能是文本，也可能是工具调用）
        response = llm.predict(messages, tools=available_tools)

        # 2. 如果模型输出了普通文本 → 任务结束，返回给用户
        if response.type == "text":
            return response.text

        # 3. 如果模型输出了工具调用 → 让代码去“做”
        if response.type == "tool_call":
            result = execute_tool(response.function, response.arguments)
            messages.append({"role": "tool", "content": result})
            # 回到第 1 步，让模型看到工具结果后继续“想”

    return "达到最大步数"
```

就这么多。整个 Agent 的核心是一个 **while 循环**：

```text
用户输入 → LLM 预测 → 输出文本？结束
                     → 输出工具调用？执行 → 把结果喂回 LLM → 继续预测 → ...
```

每一次”LLM 预测”都是第七篇讲的完整推理过程：Prefill → Decode → 一个 token 一个 token 蹦出来。每一次循环，上下文窗口（第八篇）里的内容都在增长——加入了新的工具调用和返回结果。

### 3.2 一个具体的执行过程

用户说：“帮我看看 /tmp 目录下有什么文件，然后创建一个 hello.txt”

```text
Step 1: LLM 预测
  输入: 用户消息
  输出: {"function": "execute_bash", "arguments": {"command": "ls /tmp"}}
  → 模型决定先看看目录里有什么

Step 2: 代码执行
  execute_bash("ls /tmp") → "file1.txt  file2.log  data/"
  → 结果追加到 messages

Step 3: LLM 预测
  输入: 用户消息 + 上一步的工具调用和结果
  输出: {"function": "write_file", "arguments": {"path": "/tmp/hello.txt", "content": "Hello!"}}
  → 模型看到目录内容后，决定创建文件

Step 4: 代码执行
  write_file("/tmp/hello.txt", "Hello!") → "成功"
  → 结果追加到 messages

Step 5: LLM 预测
  输入: 全部历史
  输出: "已完成！/tmp 目录下原有 file1.txt、file2.log 和 data/ 目录，
        我已经创建了 hello.txt 文件。"
  → 这次模型预测出来的 token 不再拼成 JSON，而是拼成了自然语言 → 循环结束
```

**模型在每一步做的事情完全一样：预测下一个 token。** 只不过有时候预测出来的 token 拼成了工具调用的 JSON，有时候拼成了给用户的自然语言回答。“决定用什么工具”和”写一段回答”在模型看来没有区别——都是预测下一个词。

-----

## 四、用代码串起来：一个 50 行的 Agent

附件 [tiny_agent.py](./tiny_agent.py) 实现了一个能实际运行的最小 Agent，核心逻辑：

```python
# 工具定义：告诉模型“你有哪些工具可以用”
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "执行一条 bash 命令",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"]
            }
        }
    }
]

# 工具的实际实现
def execute_bash(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

# Agent 核心循环
while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 可替换为任意支持 tools 的模型
        messages=messages,
        tools=tools
    )

    # 没有工具调用 → 任务结束
    if not response.choices[0].message.tool_calls:
        print(response.choices[0].message.content)
        break

    # 有工具调用 → 执行，把结果加入 messages，继续循环
    for tool_call in response.choices[0].message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = execute_bash(args["command"])
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
```

这段代码和 Agent 系列第一篇里的 `agent.py` 本质上是同一个东西——只是现在你完全理解了 `client.chat.completions.create()` 背后发生的所有事情：分词 → Embedding → Attention → FFN × N 层 → Softmax → 采样 → 一个 token 一个 token 输出。

> 完整可运行代码见附件 [tiny_agent.py](./tiny_agent.py)。支持任意兼容 OpenAI 格式的 API，可搭配 ChatGPT、Qwen、DeepSeek 等支持 tools 的模型使用。

-----

## 五、大模型系列的知识如何映射到 Agent

现在我们可以把十篇文章和 Agent 的每个环节对应起来：

|Agent 中的环节|对应大模型系列的哪一篇|
|---|---|
|用户输入 → 分词|第二篇：Token|
|Token → 向量|第三篇：Embedding|
|理解上下文|第四篇：Attention|
|逐层处理|第五篇：Transformer 全景|
|模型为什么”会”做这些|第六篇：训练（SFT 阶段学会了工具调用格式）|
|输出 token（文本或 JSON）|第七篇：推理（Prefill + Decode）|
|上下文越来越长|第八篇：上下文窗口（Agent 多步执行吃窗口特别快）|
|大模型 vs 小模型怎么选|第九篇：Scaling Law（简单决策用小模型，复杂推理用大模型）|

**Agent 不是什么新东西。它就是大模型 + 一个 while 循环 + 几个工具函数。** 大模型系列讲的是”大脑怎么工作”，Agent 系列讲的是”大脑怎么指挥四肢”。两者合在一起，就是完整的 AI Agent 架构。

-----

## 六、Agent 面临的核心挑战——全都和大模型有关

理解了大模型的原理，你就能看透 Agent 的核心瓶颈：

### 6.1 上下文窗口是硬伤

Agent 每调用一次工具，上下文就会膨胀：用户消息 + 工具调用 JSON + 工具返回结果（可能是大段日志或代码）。几轮循环下来，窗口就满了。

第八篇讲过，窗口满了之后最早的消息会被截掉——Agent 就”忘了”之前做过什么。这是 Agent 系列第六篇讲上下文压缩的原因。

### 6.2 每一步都有推理延迟

Agent 每做一个决策就是一次完整的推理（第七篇）。Prefill 阶段要处理所有历史消息，上下文越来越长 → Prefill 越来越慢。一个涉及 10 次工具调用的任务，总延迟可能几十秒。

### 6.3 模型可能”走偏”

大模型的每一步输出都是概率采样（第七篇）。它可能调错工具、传错参数、甚至”幻觉”出一个不存在的工具。而且错误会累积——第 3 步调错了，第 4 步基于错误的结果继续走，越走越偏。

这就是为什么 Agent 需要安全机制（Agent 系列第七篇讲的 Hook 和权限控制）——不能让模型不受限制地执行任意操作。

### 6.4 成本和规模的矛盾

第九篇讲过，大模型越大越聪明。但 Agent 调用 LLM 的次数远多于普通对话——一个 Agent 任务可能顶 10 次普通对话的 token 消耗。用最大的模型做 Agent，成本会很高。

实际中常见的策略是**分级调用**：用小模型（7B）做简单决策（要不要调工具、调哪个），用大模型（70B+）做复杂推理（分析工具结果、生成最终回答）。

-----

## 七、回顾：十篇文章的完整地图

```text
输入: "帮我看看 /tmp 下有什么文件"
  │
  ▼ 分词 ──────────── 第二篇：Token
  │  "帮我看看..." → [token IDs]
  │
  ▼ Embedding ─────── 第三篇：向量
  │  [IDs] → [向量₁, 向量₂, ...]
  │
  ▼ Transformer ───── 第四篇 Attention + 第五篇 全景
  │  向量经过 N 层(Attention + FFN)变换
  │  训练（第六篇）让模型学会了工具调用格式
  │
  ▼ 推理输出 ──────── 第七篇：逐 token 生成
  │  模型输出: {"function": "execute_bash", "arguments": {"command": "ls /tmp"}}
  │
  ▼ 外部代码执行工具
  │  ls /tmp → "file1.txt  data/"
  │
  ▼ 结果塞回上下文 ── 第八篇：窗口在增长
  │
  ▼ 再次推理 ──────── 回到 Transformer
  │  模型输出: "目录下有 file1.txt 和 data/ 文件夹。"
  │
  ▼ 返回给用户
  │
  └── 整个过程中，模型越大越准（第九篇 Scaling Law）
```

**从头到尾，模型做的事情只有一件：预测下一个 token。** 分词、Embedding、Attention、FFN、训练、推理、上下文窗口、Scaling Law——所有这些，都是为了让这个”预测”做得更准、更快、更大规模。

而 Agent 做的事情也只有一件：**用一个循环，把大模型的”预测”转化为真实世界的”行动”。**

-----

## 八、结语：两个系列的交汇

这是「从零开始理解大模型」的最后一篇。

十篇文章，从 “Thank you very → much (99.2%)” 这个最简单的例子出发，我们一层一层拆开了大模型的全部核心机制：

|篇|你搞明白了什么|
|---|---|
|第一篇|大模型在干什么——预测下一个词，就这一件事|
|第二篇|模型看到的不是文字，是 token——BPE 怎么切、中英文效率差异|
|第三篇|Token 怎么变成数学——Embedding 查表、向量空间里的语义关系|
|第四篇|模型怎么理解上下文——Attention 的 Q·Kᵀ/√d → softmax → ·V|
|第五篇|完整的 Transformer——(Attention + FFN) × N 层 + 残差 + LayerNorm|
|第六篇|参数怎么学出来——Loss、梯度、反向传播、预训练→SFT→RLHF|
|第七篇|按下回车后发生了什么——Prefill + Decode、KV Cache|
|第八篇|模型的”记忆”有多大——上下文窗口、n² 瓶颈、RoPE|
|第九篇|为什么越大越强——Scaling Law、Chinchilla、涌现能力|
|第十篇|怎么从文本生成变成行动——Function Calling、Agent 循环|

如果你跟完了这十篇，你对大模型的理解已经超过了绝大多数使用者。下次有人问你”大模型是怎么工作的”，你可以从 token 讲到 Attention，从训练讲到推理，从 Scaling Law 讲到 Agent——而且每一步你都跑过代码、看过数据。

### 接下来可以读什么

如果你还没读过「从零开始理解 Agent」系列，现在是最好的时机。那个系列从 Agent 的角度出发，用 115 行到 265 行 Python 代码，逐层搭建出完整的 Agent 系统——工具调用、记忆、规划、SubAgent、Teams、上下文压缩、安全控制。

两个系列合在一起：

```text
「理解大模型」 = 大脑怎么工作
「理解 Agent」 = 大脑怎么指挥四肢

大脑 + 四肢 = 完整的 AI Agent
```

> *“Prediction is not just a task. It is the task.”*
>
> 预测下一个词不只是大模型的一个功能，而是它的全部。从理解语言到生成代码，从回答问题到调用工具——一切都是预测下一个 token。十篇文章，讲的就是这一件事为什么能走这么远。

感谢你读到这里。

-----

*本文配套代码：[tiny_agent.py](./tiny_agent.py)（50 行最小 Agent 实现）。需要 Python 3.8+、openai 库、任意支持 tools 的模型 API Key。*

*「从零开始理解大模型」是「从零开始理解 Agent」的姊妹系列。Agent 系列讲”四肢”，本系列讲”大脑”。建议对照阅读。*

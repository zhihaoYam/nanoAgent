# 从零开始理解大模型（二）：Token——大模型眼中的"字"长什么样

> **「从零开始理解大模型」系列** —— 十篇文章，从"下一个词预测"到完整的大模型心智模型。每篇配可运行代码。
> 
> - [第一篇：一切从"猜下一个词"开始](./llm-01-next-token.md)
> - **第二篇：Token——大模型眼中的"字"长什么样**（本文）
> - 第三篇：向量与 Embedding——把文字变成数学
> - 第四篇：Attention——大模型的"阅读理解"机制
> - 第五篇：Transformer 全景——积木怎么搭成大厦
> - 第六篇：训练——70 亿个参数是怎么"学"出来的
> - 第七篇：推理——你按下回车后的这一秒发生了什么
> - 第八篇：上下文窗口——大模型的"工作记忆"
> - 第九篇：Scaling Law——为什么"大力出奇迹"有效
> - 第十篇：从大模型到 Agent——下一个词预测如何长出手脚

上一篇我们看到，模型以 99.2% 的概率预测 "Thank you very" 后面接 "much"。代码里有一行被跳过了：

```python
input_ids = tokenizer.encode("Thank you very", return_tensors="pt")
# 输出: [10449, 345, 845]
```

三个英文单词变成了三个数字。模型看到的不是 "Thank you very"，而是 `[10449, 345, 845]`。

**这三个数字就是三个 token。** Token 是大模型的最小阅读单位——模型不认识字母，不认识汉字，它只认识 token。

本篇回答三个问题：token 到底是什么？怎么切出来的？为什么你需要关心它？

-----

## 一、先说结论

|你以为的         |实际的                                     |
|-------------|----------------------------------------|
|模型按"词"来读文本   |模型按 token 来读，一个 token 可能是一个词、半个词、一个字符   |
|中文和英文的处理方式相同 |一个英文常用词 ≈ 1 token，一个汉字 ≈ 1-2 个 token    |
|Token 数量 ≈ 字数|英文约 1 token ≈ 0.75 词；中文 1 汉字 ≈ 1-2 token|
|分词是个简单的事情    |分词方式直接影响成本、速度和模型能力                      |

一句话总结：**Token 是人类语言和模型之间的翻译层——人类写字，模型读 token。**

-----

## 二、Token 不是"词"

用第一篇的 `tokenizer` 来看几个例子：

```python
tokenizer.encode("Hello")        # → [15496]             1 个 token
tokenizer.encode("hello")        # → [31373]             1 个 token，但 ID 不同！
tokenizer.encode("Kubernetes")   # → [42, 18478, 3262, 274]  4 个 token，被切碎了
tokenizer.encode("strawberry")   # → [301, 1831, 8396]    3 个 token
tokenizer.encode("你好")         # → [19526, 254, 25001, 121]  4 个 token！
```

几个反直觉的事实：

**大小写不同 = 不同的 token。** "Hello"（15496）和 "hello"（31373）在模型眼里是两个完全不同的"字"。

**空格被编进 token 里。** `" capital"`（前面带空格）和 `"capital"` 是不同的 token。GPT 系列把空格粘在下一个词的前面。

**专业术语被切碎。** "Kubernetes" 不在词表里，被切成了四段碎片（K-uber-net-es）。模型需要从碎片中"拼"出含义。

**中文被切得更碎。** 两个汉字 "你好" 竟然变成了 4 个 token——因为 GPT-2 的词表主要基于英文构建，中文字符被拆成了字节级碎片。

> 完整的分词实验代码见附件 [tokenizer_demo.py](./tokenizer_demo.py)，支持命令行输入任意文本测试。

-----

## 三、BPE——Token 是怎么"切"出来的

为什么 "Hello" 是 1 个 token 而 "Kubernetes" 是 4 个？这由一个叫 **BPE（Byte Pair Encoding，字节对编码）** 的算法决定。

### 3.1 核心思路

你需要设计一个固定大小的"字典"（比如 50257 个 token），用来表示所有可能的文本。

极端方案一：收录所有词 → 词无穷多，不可行。
极端方案二：只收 256 个字节 → 任何文本都能表示，但效率极低。

BPE 的做法是：**从单个字节开始，反复合并最频繁的相邻对，直到词表达到目标大小。** 形式化表示：

```
初始词表 V₀ = {所有单字节}    // 256 个

重复以下步骤：
    (a, b) = argmax_{所有相邻对} count(a, b)    // 找出语料中出现次数最多的相邻 token 对
    合并 a + b → ab                              // 创建新 token
    V_{i+1} = V_i ∪ {ab}                        // 加入词表

直到 |V| = 目标大小（如 50257）
```

### 3.2 一个具体例子

训练语料：`low low low low low lower lower newest newest newest widest`

```
起点：每个字符是一个 token
  → {l, o, w, e, r, n, s, t, i, d}

轮次 1: (l, o) 出现 7 次 → 合并为 "lo"
轮次 2: (lo, w) 出现 7 次 → 合并为 "low"
轮次 3: (e, s) 出现 4 次 → 合并为 "es"
轮次 4: (es, t) 出现 4 次 → 合并为 "est"
轮次 5: (n, e) 出现 3 次 → 合并为 "ne"
轮次 6: (ne, w) 出现 3 次 → 合并为 "new"
轮次 7: (new, est) 出现 3 次 → 合并为 "newest"
...
```

训练完成后，用这些合并规则对新词分词：

```
"low"    → [low]          训练中高频，完整保留
"lowest" → [low, est]     两个已知子词拼接（"lowest" 训练时没见过！）
"newest" → [newest]       训练中高频，完整保留
"widest" → [w, i, d, est] 低频词被切碎，但 "est" 被识别出来
```

### 3.3 BPE 的三个关键性质

**纯统计过程。** BPE 不懂语言学。"est" 被合并不是因为算法知道它是后缀，而是因为它在语料中频繁出现。结果恰好和语言学吻合——这是统计的副产品。

**高频完整、低频切碎。** 训练数据中出现越多的片段越容易被合并成完整 token。这就是 "Hello" 是 1 个 token 而 "Kubernetes" 是 4 个的原因——前者在互联网文本中远比后者常见。

**能泛化到新词。** "lowest" 没出现在训练语料中，但 "low" 和 "est" 都是已学到的 token，新词自动被切成 `[low, est]`。BPE 天然处理了"未见过的词"。

> **关键洞察**：BPE 词表是在训练数据上学出来的。训练数据以英文为主 → 英文词表丰富、切得粗、效率高；中文作为"非主流语言" → 词表位置少、切得碎、效率低。这不是歧视，是统计分布的自然结果。

> 完整的 BPE 手写实现见附件 [bpe_demo.py](./bpe_demo.py)，纯 Python 无依赖，可以修改训练语料观察词表变化。

-----

## 四、Token 效率——为什么你需要关心

Token 数量直接决定三件事：**你花多少钱、等多久、能塞多少内容。**

### 4.1 中英文效率对比

同一句话，中文版的 token 数通常是英文版的 2-3 倍（GPT-2 分词器）：

|英文                                                          |EN tokens|中文                     |ZH tokens|倍率  |
|------------------------------------------------------------|---------|-----------------------|---------|----|
|Hello, how are you?                                         |6        |你好，你怎么样？               |18       |3.0x|
|Artificial intelligence is transforming the world.          |8        |人工智能正在改变世界。            |22       |2.8x|
|The Kubernetes cluster has been deployed across three zones.|13       |Kubernetes 集群已部署到三个可用区。|29       |2.2x|
|Please review the pull request before Friday.               |8        |请在周五之前审查这个 PR。         |21       |2.6x|

### 4.2 三个影响公式

```
API 费用 = token_count × price_per_token
生成延迟 ≈ output_tokens × time_per_token
可用上下文 ≈ context_window / avg_tokens_per_char
```

同样的内容用中文处理，费用翻倍、速度减半、能塞进上下文的内容更少。这就是为什么 DeepSeek、Qwen 等国产模型专门优化了中文词表：

|模型         |词表规模  |中文优化|中文效率               |
|-----------|------|----|-------------------|
|GPT-2      |约 5 万 |无   |差（1 汉字 ≈ 2-3 token）|
|LLaMA 2    |约 3 万 |无   |差                  |
|GPT-4o     |约 20 万|有   |好（1 汉字 ≈ 1 token）  |
|DeepSeek V3|约 10 万|有   |好                  |
|Qwen 2     |约 15 万|强   |很好                 |

趋势明确：**新一代模型都在扩大词表、增加多语言 token。** 词表大了，常见中文词可以作为完整 token 存在，不需要被切成字节碎片。

> 完整的中英文效率对比和费用估算代码见附件 [token_cost.py](./token_cost.py)。

-----

## 五、"strawberry 里有几个 r"——分词决定模型的视力

2024 年有一个著名的测试：问大模型 "How many r's in strawberry?"，很多模型答错。原因不在"智力"，而在"视力"：

```python
tokenizer.encode("strawberry")   # → ["st", "raw", "berry"]
```

模型看到的是三个 token，看不到单独的 "r"。要数 "r" 的数量，它需要在 token 内部做字符级推理——而这不是它被训练来做的事。

**Token 是模型能看到的最小单位。** Token 内部的字符结构对模型来说是模糊的。很多"能力缺陷"——数不对字母、处理不好罕见语言、无法做字符级操作——根源都在分词层。

> **模型的智力受限于它的视力。**

-----

## 六、特殊 Token：模型的控制信号

除了表示文本的普通 token，还有一类**特殊 token**——不对应自然语言，用来给模型发"指令"：

```
<|endoftext|>   → 文本结束
<|system|>      → 系统消息
<|user|>        → 用户消息
<|assistant|>   → "现在该我说了"
```

第一篇提到，当你问 "法国的首都是哪里？" 时，模型看到的输入其实是：

```
<|system|>你是一个有帮助的助手。<|end|>
<|user|>法国的首都是哪里？<|end|>
<|assistant|>
```

这些特殊 token 不来自 BPE，而是**手动添加**到词表中的。如果你读过 Agent 系列第一篇，Agent 的 `messages` 列表中的 `role` 字段，发送给模型前就会被转换成这些特殊 token。**JSON 是给人看的，特殊 token 才是给模型看的。**

模型生成 `<|endoftext|>` 时就意味着"我说完了"——这就是 Agent 循环中"任务结束"的底层信号。

-----

## 七、Token 在完整链路中的位置

回顾第一篇的代码，现在你对每一步的理解都深了一层：

```python
# 第一步：分词 —— 本篇的主题
input_ids = tokenizer.encode("Thank you very", return_tensors="pt")
# "Thank you very" → [10449, 345, 845]   ← 3 个 token ID

# 第二步：token ID → 向量（查 Embedding 表）→ 第三篇
# [10449, 345, 845] → 3 个 768 维的向量

# 第三步：向量经过 Attention 层层变换 → 第四、五篇
# 3 个向量 → 经过 12 层 Transformer → 3 个新向量

# 第四步：最后一个向量 → 50257 个概率 → 第一篇已讲过
# softmax(linear(最后一个向量)) → P("much") = 99.2%
```

Token 是这条链路的起点。它决定了模型看到什么、看不到什么、用户花多少钱、模型跑多快、上下文能装多少。

下一篇，我们打开第二个黑盒：token ID 是怎么变成向量的？为什么 "France" 和 "Paris" 编号差很远，模型却知道它们有关系？

-----

## 八、结语

Token 是大模型和人类语言之间的翻译层。

这个翻译层看起来不起眼，但它决定了模型的视力边界、使用成本和处理效率。"数不对字母、中文比英文贵、长文本容易截断"——追到底，都能在 token 这一层找到原因。

> *"Between the human world and the model world, there is a thin layer of translation. That layer is tokenization."*

理解了 token，你就理解了大模型的"入口"。

-----

*本文配套代码：[tokenizer_demo.py](./tokenizer_demo.py)（分词实验）、[bpe_demo.py](./bpe_demo.py)（手写 BPE）、[token_cost.py](./token_cost.py)（效率对比）。需要 Python 3.8+、transformers。*

*「从零开始理解大模型」是「从零开始理解 Agent」的姊妹系列。Agent 系列讲"四肢"，本系列讲"大脑"。建议对照阅读。*

"""
token_cost.py —— 中英文 token 效率与费用影响
从零开始理解大模型（二）配套代码

用法：
    python token_cost.py
    python token_cost.py "你想测试的任意文本"

需要：pip install transformers
"""

import sys
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def analyze_text(text, label=""):
    """分析一段文本的 token 统计信息"""
    tokens = tokenizer.encode(text)
    decoded = [tokenizer.decode([t]) for t in tokens]

    return {
        "label": label,
        "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
        "char_count": len(text),
        "token_count": len(tokens),
        "ratio": len(tokens) / len(text) if text else 0,
        "tokens_preview": decoded[:8],
    }


# ==================== 自定义输入模式 ====================

if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
    result = analyze_text(text)
    tokens = tokenizer.encode(text)
    decoded = [tokenizer.decode([t]) for t in tokens]

    print(f"输入: '{text}'")
    print(f"字符数: {result['char_count']}")
    print(f"Token 数: {result['token_count']}")
    print(f"Token/字符比: {result['ratio']:.2f}")
    print(f"Tokens: {decoded}")
    sys.exit(0)

# ==================== 完整对比模式 ====================

print("=" * 80)
print("GPT-2 分词器 · 中英文 Token 效率对比")
print("=" * 80)

pairs = [
    ("Hello, how are you?", "你好，你怎么样？"),
    ("Artificial intelligence is transforming the world.",
     "人工智能正在改变世界。"),
    ("The Kubernetes cluster has been deployed across three zones.",
     "Kubernetes 集群已部署到三个可用区。"),
    ("Please review the pull request before Friday.",
     "请在周五之前审查这个 PR。"),
    ("Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
     "机器学习是人工智能的一个子集，它使系统能够从数据中学习。"),
]

print(f"\n  {'语言':<4} {'文本':<50} {'字符':>4} {'Token':>5} {'比率':>6}")
print("  " + "-" * 75)

for en, zh in pairs:
    en_r = analyze_text(en)
    zh_r = analyze_text(zh)
    overhead = zh_r["token_count"] / en_r["token_count"]

    print(f"  EN   {en:<50} {en_r['char_count']:>4} {en_r['token_count']:>5} {en_r['ratio']:>6.2f}")
    print(f"  ZH   {zh:<50} {zh_r['char_count']:>4} {zh_r['token_count']:>5} {zh_r['ratio']:>6.2f}")
    print(f"       → 中文 token 数是英文的 {overhead:.1f} 倍")
    print()

# ==================== 费用估算 ====================

print("=" * 80)
print("费用影响估算")
print("=" * 80)

# 生成约 1000 词/字的文档
en_doc = "The quick brown fox jumps over the lazy dog and then runs across the field. " * 67
zh_doc = "敏捷的棕色狐狸跳过了懒狗然后跑过了田野。" * 53

en_tokens = len(tokenizer.encode(en_doc))
zh_tokens = len(tokenizer.encode(zh_doc))

print(f"\n  英文文档 (~1000 词, {len(en_doc)} 字符): {en_tokens} tokens")
print(f"  中文文档 (~1000 字, {len(zh_doc)} 字符): {zh_tokens} tokens")
print(f"  中文 token 开销倍率: {zh_tokens / en_tokens:.1f}x")

# 按几个常见的 API 价格档位估算
price_tiers = [
    ("经济型（如 GPT-4o-mini）", 0.00015),
    ("标准型（如 GPT-4o）", 0.005),
    ("旗舰型（如 Claude Opus）", 0.015),
]

print(f"\n  API 费用估算（处理上述文档，输入 token 价格）：")
print(f"  {'价格档位':<30} {'英文':>10} {'中文':>10} {'差额':>10}")
print("  " + "-" * 65)

for name, price_per_token in price_tiers:
    en_cost = en_tokens * price_per_token / 1000
    zh_cost = zh_tokens * price_per_token / 1000
    diff = zh_cost - en_cost
    print(f"  {name:<30} ${en_cost:>8.4f} ${zh_cost:>8.4f} +${diff:>8.4f}")

print(f"\n  注意：以上基于 GPT-2 分词器。新一代模型（GPT-4o、Qwen2、DeepSeek）")
print(f"  的中文分词效率已大幅改善，差距缩小到 1.0-1.5x。")

# ==================== 不同类型文本的 token 效率 ====================

print(f"\n\n{'=' * 80}")
print("不同类型文本的 token 效率")
print("=" * 80)

samples = [
    ("英文散文", "The beauty of nature lies in its impermanence and the way it teaches us to appreciate the present moment."),
    ("中文散文", "自然之美在于它的无常，以及它教会我们珍惜当下的方式。"),
    ("Python 代码", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
    ("JSON 数据", '{"name": "Alice", "age": 30, "city": "Beijing", "hobbies": ["reading", "coding"]}'),
    ("URL", "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/"),
    ("数学表达式", "f(x) = 3x^2 + 2x - 1, where x = 4, f(4) = 3(16) + 2(4) - 1 = 55"),
]

print(f"\n  {'类型':<12} {'字符数':>5} {'Token数':>6} {'Token/字符':>10}")
print("  " + "-" * 40)

for label, text in samples:
    r = analyze_text(text, label)
    print(f"  {label:<12} {r['char_count']:>5} {r['token_count']:>6} {r['ratio']:>10.2f}")
    print(f"    → {r['tokens_preview']}")

print(f"\n  观察：代码和 URL 的 token 效率通常低于自然语言散文。")
print(f"  这意味着在 Agent 场景中，工具调用结果（JSON、日志、URL）")
print(f"  比你想象的更快吃掉上下文窗口。")

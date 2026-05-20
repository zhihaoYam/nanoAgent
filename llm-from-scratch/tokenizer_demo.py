"""
tokenizer_demo.py —— 亲眼看到 token 是怎么切的
从零开始理解大模型（二）配套代码

用法：
    python tokenizer_demo.py
    python tokenizer_demo.py "Kubernetes is awesome"
    python tokenizer_demo.py "你好世界"

需要：pip install transformers
"""

import sys
from transformers import GPT2Tokenizer

# ==================== 1. 加载分词器 ====================

print("加载 GPT-2 分词器…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ==================== 2. 自定义输入或默认示例 ====================

if len(sys.argv) > 1:
    examples = [" ".join(sys.argv[1:])]
else:
    examples = [
        "Hello",
        "hello",
        "The capital of France is Paris",
        "unhappiness",
        "Kubernetes",
        "transformer",
        "I don't think so",
        "   spaces   ",
        "123456789",
        "Hello! 你好！こんにちは！",
        "strawberry",
        "def hello_world():",
        "https://github.com/user/repo",
    ]

# ==================== 3. 分词展示 ====================

print(f"\nGPT-2 词表大小: {tokenizer.vocab_size} 个 token")
print("=" * 70)

for text in examples:
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([id]) for id in token_ids]

    print(f"\n  输入: '{text}'")
    print(f"  Token 数量: {len(tokens)}")
    print(f"  Tokens: {tokens}")
    print(f"  Token IDs: {token_ids}")

# ==================== 4. 中英文效率对比 ====================

if len(sys.argv) <= 1:
    print("\n" + "=" * 70)
    print("中英文 token 数量对比")
    print("=" * 70)

    pairs = [
        ("The weather is nice today", "今天天气真好"),
        ("Artificial Intelligence", "人工智能"),
        ("I love programming", "我喜欢编程"),
        ("The president of the United States", "美国总统"),
        ("Machine learning is a subset of artificial intelligence",
         "机器学习是人工智能的一个子集"),
    ]

    print(f"\n  {'英文':<45} tok | {'中文':<20} tok | 倍率")
    print("  " + "-" * 90)

    for en, zh in pairs:
        en_tokens = len(tokenizer.encode(en))
        zh_tokens = len(tokenizer.encode(zh))
        ratio = zh_tokens / en_tokens if en_tokens > 0 else 0
        print(f"  {en:<45} {en_tokens:>3} | {zh:<20} {zh_tokens:>3} | {ratio:.1f}x")

# ==================== 5. 词表抽样 ====================

if len(sys.argv) <= 1:
    import random
    random.seed(42)

    print(f"\n\n词表随机抽样（30 个 token）：")
    print("-" * 40)

    sample_ids = random.sample(range(tokenizer.vocab_size), 30)
    sample_ids.sort()

    for token_id in sample_ids:
        token_text = tokenizer.decode([token_id])
        display = repr(token_text) if not token_text.strip() else f"'{token_text}'"
        print(f"  ID {token_id:>5}: {display}")

"""
embedding.py —— 亲眼看到 Embedding 表和向量相似度
从零开始理解大模型（三）配套代码

用法：
    python embedding.py
    python embedding.py "France" "Paris" "cat"

需要：pip install transformers torch
"""

import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ==================== 加载模型 ====================

print("加载模型…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ==================== 取出 Embedding 表 ====================

embedding_table = model.transformer.wte.weight.detach()
print(f"\nEmbedding 表的形状: {embedding_table.shape}")
print(f"  行数（词表大小）: {embedding_table.shape[0]}")
print(f"  列数（向量维度）: {embedding_table.shape[1]}")
print(f"  总参数量: {embedding_table.shape[0] * embedding_table.shape[1]:,}")


def get_vector(word):
    """获取一个词的 Embedding 向量，自动处理带/不带空格的变体"""
    for variant in [word, " " + word, word.lower(), " " + word.lower(),
                    word.capitalize(), " " + word.capitalize()]:
        ids = tokenizer.encode(variant)
        if len(ids) == 1:
            return embedding_table[ids[0]], ids[0], variant.strip()
    ids = tokenizer.encode(word)
    return embedding_table[ids[0]], ids[0], tokenizer.decode([ids[0]]).strip()


def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (v1.norm() * v2.norm())


# ==================== 获取词列表 ====================

if len(sys.argv) > 1:
    words = sys.argv[1:]
else:
    words = ["France", "Paris", "Germany", "Berlin", "cat", "dog", "the",
             "king", "queen", "man", "woman"]

# ==================== 展示向量 ====================

print(f"\n各 token 的向量（前 8 维）：")
print("-" * 70)

vectors = {}
for word in words:
    vec, tid, display = get_vector(word)
    vectors[word] = vec
    preview = [f"{v:.3f}" for v in vec[:8].tolist()]
    print(f"  '{display}' (ID {tid}): [{', '.join(preview)}, …]")

# ==================== 相似度矩阵 ====================

print(f"\n\n向量相似度（余弦相似度）：")
print("=" * 60)

computed_pairs = set()
results = []

for w1 in words:
    for w2 in words:
        if w1 >= w2:
            continue
        key = tuple(sorted([w1, w2]))
        if key in computed_pairs:
            continue
        computed_pairs.add(key)

        if w1 in vectors and w2 in vectors:
            sim = cosine_similarity(vectors[w1], vectors[w2]).item()
            results.append((w1, w2, sim))

# 按相似度降序排列
results.sort(key=lambda x: -x[2])

for w1, w2, sim in results:
    bar = "█" * int(max(0, sim) * 30)
    print(f"  '{w1}' vs '{w2}': {sim:>7.3f}  {bar}")

# ==================== 向量统计 ====================

print(f"\n\n向量统计信息：")
print("-" * 40)

for word in words[:5]:
    if word in vectors:
        vec = vectors[word]
        print(f"  '{word}': 范数={vec.norm().item():.3f}, "
              f"均值={vec.mean().item():.4f}, "
              f"标准差={vec.std().item():.4f}")

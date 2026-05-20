"""
visualize.py —— 把 768 维向量降到 2D，看看语义聚类
从零开始理解大模型（三）配套代码

用法：
    python visualize.py

需要：pip install transformers torch numpy
输出：终端文本散点图 + embedding_2d.json 数据文件
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import json
from itertools import combinations

print("加载模型…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

embedding_table = model.transformer.wte.weight.detach()

# ==================== 1. 选择要可视化的词 ====================

word_groups = {
    "国家": ["France", "Germany", "Japan", "China", "Italy", "Spain", "Russia", "Brazil", "India"],
    "城市": ["Paris", "Berlin", "Tokyo", "Beijing", "Rome", "Madrid", "Moscow", "London", "Delhi"],
    "动物": ["cat", "dog", "fish", "bird", "horse", "tiger", "lion", "bear", "wolf"],
    "颜色": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"],
    "数字": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "编程": ["Python", "Java", "code", "function", "variable", "class", "data", "array"],
    "食物": ["apple", "bread", "rice", "cheese", "pizza", "chicken", "chocolate", "coffee"],
}

all_words = []
all_vectors = []
all_groups = []

for group, words in word_groups.items():
    for word in words:
        for variant in [" " + word, word, word.lower(), " " + word.lower()]:
            ids = tokenizer.encode(variant)
            if len(ids) == 1:
                vec = embedding_table[ids[0]].numpy()
                all_words.append(word)
                all_vectors.append(vec)
                all_groups.append(group)
                break

print(f"收集了 {len(all_words)} 个词的向量（来自 {len(word_groups)} 个语义组）")

# ==================== 2. PCA 降维 ====================

vectors_matrix = np.array(all_vectors)
mean = vectors_matrix.mean(axis=0)
centered = vectors_matrix - mean

cov = centered.T @ centered / len(centered)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 取最大的两个特征向量
pc1 = eigenvectors[:, -1]
pc2 = eigenvectors[:, -2]

x = centered @ pc1
y = centered @ pc2

# 解释的方差比例
total_var = eigenvalues.sum()
explained_1 = eigenvalues[-1] / total_var * 100
explained_2 = eigenvalues[-2] / total_var * 100
print(f"PCA: PC1 解释 {explained_1:.1f}% 方差, PC2 解释 {explained_2:.1f}% 方差")

# ==================== 3. 按组输出坐标 ====================

group_symbols = {
    "国家": "●", "城市": "■", "动物": "▲", "颜色": "◆",
    "数字": "★", "编程": "◎", "食物": "♦",
}

print(f"\n{'=' * 60}")
print("PCA 降维后的 2D 坐标")
print("=" * 60)

for group in word_groups:
    print(f"\n  {group_symbols.get(group, '○')} {group}:")
    for i in range(len(all_words)):
        if all_groups[i] == group:
            print(f"    {all_words[i]:<12} ({x[i]:>7.2f}, {y[i]:>7.2f})")

# ==================== 4. 文本散点图 ====================

print(f"\n\n{'=' * 60}")
print("文本散点图（PC1 × PC2）")
print("=" * 60)

width = 70
height = 30

x_min, x_max = x.min() - 0.5, x.max() + 0.5
y_min, y_max = y.min() - 0.5, y.max() + 0.5

grid = [[' ' for _ in range(width)] for _ in range(height)]

# 先放标记
placed = {}
for i in range(len(all_words)):
    gx = int((x[i] - x_min) / (x_max - x_min) * (width - 1))
    gy = int((y[i] - y_min) / (y_max - y_min) * (height - 1))
    gy = height - 1 - gy

    gx = max(0, min(width - 1, gx))
    gy = max(0, min(height - 1, gy))

    symbol = group_symbols.get(all_groups[i], '?')
    key = (gy, gx)
    if key not in placed:
        grid[gy][gx] = symbol
        placed[key] = all_words[i]

print()
for row in grid:
    print("  │" + "".join(row) + "│")
print("  └" + "─" * width + "┘")

print(f"\n  图例: ", end="")
for group, symbol in group_symbols.items():
    print(f"{symbol}={group}  ", end="")
print()

# ==================== 5. 组间/组内距离统计 ====================

print(f"\n\n{'=' * 60}")
print("组内 vs 组间平均余弦相似度")
print("=" * 60)

group_indices = {}
for i, g in enumerate(all_groups):
    group_indices.setdefault(g, []).append(i)

vecs_t = torch.tensor(all_vectors)
norms = vecs_t.norm(dim=1, keepdim=True)
normalized = vecs_t / norms

print(f"\n  {'组':<10} {'组内相似度':>10} {'vs 其他组':>10} {'差值':>8}")
print("  " + "-" * 45)

for group, indices in group_indices.items():
    if len(indices) < 2:
        continue

    # 组内相似度
    intra_sims = []
    for i, j in combinations(indices, 2):
        sim = torch.dot(normalized[i], normalized[j]).item()
        intra_sims.append(sim)
    intra_avg = sum(intra_sims) / len(intra_sims)

    # 组间相似度
    other_indices = [i for i in range(len(all_words)) if i not in indices]
    inter_sims = []
    for i in indices:
        for j in other_indices[:30]:  # 采样避免太慢
            sim = torch.dot(normalized[i], normalized[j]).item()
            inter_sims.append(sim)
    inter_avg = sum(inter_sims) / len(inter_sims)

    diff = intra_avg - inter_avg
    print(f"  {group:<10} {intra_avg:>10.3f} {inter_avg:>10.3f} {diff:>+8.3f}")

print(f"\n  差值 > 0 说明组内词比组外词更相似 → 语义聚类有效")

# ==================== 6. 保存数据 ====================

output = []
for i in range(len(all_words)):
    output.append({
        "word": all_words[i],
        "group": all_groups[i],
        "x": float(x[i]),
        "y": float(y[i]),
    })

with open("embedding_2d.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n2D 坐标已保存到 embedding_2d.json")

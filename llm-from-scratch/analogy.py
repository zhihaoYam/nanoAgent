"""
analogy.py —— 向量算术：国王 - 男人 + 女人 ≈ ？
从零开始理解大模型（三）配套代码

用法：
    python analogy.py
    python analogy.py "man" "king" "woman"
    python analogy.py "France" "Paris" "Japan"

需要：pip install transformers torch
"""

import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("加载模型…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

embedding_table = model.transformer.wte.weight.detach()


def get_vector(word):
    """获取一个词的 Embedding 向量"""
    for variant in [word, " " + word, word.lower(), " " + word.lower(),
                    word.capitalize(), " " + word.capitalize()]:
        ids = tokenizer.encode(variant)
        if len(ids) == 1:
            return embedding_table[ids[0]], ids[0], variant.strip()
    ids = tokenizer.encode(word)
    return embedding_table[ids[0]], ids[0], tokenizer.decode([ids[0]]).strip()


def find_nearest(target_vector, exclude_ids=None, top_k=8):
    """在词表中找到最接近 target_vector 的 token"""
    if exclude_ids is None:
        exclude_ids = set()

    norms = embedding_table.norm(dim=1)
    target_norm = target_vector.norm()
    similarities = torch.mv(embedding_table, target_vector) / (norms * target_norm + 1e-8)

    for eid in exclude_ids:
        similarities[eid] = -float('inf')

    top_values, top_indices = torch.topk(similarities, top_k)

    results = []
    for i in range(top_k):
        token = tokenizer.decode([top_indices[i].item()])
        sim = top_values[i].item()
        results.append((token.strip(), sim))

    return results


def analogy(a, b, c, top_k=8):
    """
    解向量类比题：a 之于 b，相当于 c 之于 ？
    计算：向量(b) - 向量(a) + 向量(c)
    """
    vec_a, id_a, repr_a = get_vector(a)
    vec_b, id_b, repr_b = get_vector(b)
    vec_c, id_c, repr_c = get_vector(c)

    target = vec_b - vec_a + vec_c

    print(f"\n  '{a}' 之于 '{b}'，相当于 '{c}' 之于 ？")
    print(f"  公式：向量('{b}') - 向量('{a}') + 向量('{c}')")

    results = find_nearest(target, exclude_ids={id_a, id_b, id_c}, top_k=top_k)

    print(f"  排除输入词后，最接近的 {top_k} 个词：")
    for i, (token, sim) in enumerate(results):
        bar = "█" * int(max(0, sim) * 25)
        marker = " ◀ 期望答案？" if i == 0 else ""
        print(f"    {i+1}. '{token}' \t相似度: {sim:.3f}  {bar}{marker}")

    return results


# ==================== 运行 ====================

print("=" * 60)
print("向量算术实验（GPT-2 Embedding）")
print("=" * 60)

if len(sys.argv) == 4:
    # 命令行模式
    analogy(sys.argv[1], sys.argv[2], sys.argv[3])
else:
    # 默认实验
    experiments = [
        ("man", "king", "woman", "经典：性别类比"),
        ("France", "Paris", "Germany", "国家 → 首都"),
        ("France", "Paris", "Japan", "国家 → 首都（亚洲）"),
        ("France", "French", "Germany", "国家 → 语言/国民"),
        ("walk", "walked", "run", "动词时态"),
        ("cat", "cats", "dog", "单复数"),
        ("good", "better", "bad", "比较级"),
        ("small", "smaller", "big", "比较级"),
    ]

    for a, b, c, label in experiments:
        print(f"\n{'─' * 60}")
        print(f"  【{label}】")
        analogy(a, b, c, top_k=5)

    print(f"\n\n{'=' * 60}")
    print("说明：")
    print("  GPT-2 的 Embedding 层只是初始表示，类比结果不总是完美的。")
    print("  经过 Transformer 各层变换后，语义表示会更精确。")
    print("  试试自己的类比：python analogy.py 'word_a' 'word_b' 'word_c'")

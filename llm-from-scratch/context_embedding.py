"""
context_embedding.py —— 同一个词在不同上下文中的向量变化
从零开始理解大模型（三）配套代码

用法：
    python context_embedding.py

需要：pip install transformers torch
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("加载模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def get_all_layer_vectors(text, target_word):
    """获取目标词在每一层 Transformer 输出的向量"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]

    # 找到目标词的位置
    target_pos = None
    for i, t in enumerate(tokens):
        if target_word.lower() in t.strip().lower():
            target_pos = i
            break

    if target_pos is None:
        return None, tokens, None

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # hidden_states[0] = Embedding 层, [1]-[12] = Transformer 各层
    layer_vectors = [hs[0, target_pos, :].detach() for hs in outputs.hidden_states]

    return layer_vectors, tokens, target_pos


def cosine_sim(v1, v2):
    return torch.dot(v1, v2) / (v1.norm() * v2.norm())


# ==================== 实验 1: 一词多义 ====================
print("\n" + "=" * 65)
print("实验 1: 'bank' 在不同语境中的向量变化")
print("=" * 65)

sentences = [
    ("I deposited money at the bank today", "bank", "银行"),
    ("The river bank was covered with flowers", "bank", "河岸"),
    ("The bank approved my mortgage application", "bank", "银行"),
    ("We had a picnic on the bank of the lake", "bank", "河岸"),
]

results = []
for text, word, meaning in sentences:
    vectors, tokens, pos = get_all_layer_vectors(text, word)
    if vectors:
        results.append({"text": text, "meaning": meaning, "vectors": vectors})
        print(f"\n  [{meaning}] '{text}'")
        print(f"  tokens: {tokens}, 目标位置: {pos}")

if len(results) >= 4:
    print(f"\n\n相似度对比：Embedding 层 vs 最后一层")
    print("-" * 65)

    comparisons = [
        (0, 2, "银行 vs 银行"),
        (1, 3, "河岸 vs 河岸"),
        (0, 1, "银行 vs 河岸"),
        (0, 3, "银行 vs 河岸"),
    ]

    print(f"\n  {'对比':<25} {'Embedding':>10} {'最后一层':>10} {'变化':>10}")
    print("  " + "-" * 60)

    for i, j, label in comparisons:
        sim_e = cosine_sim(results[i]["vectors"][0], results[j]["vectors"][0]).item()
        sim_f = cosine_sim(results[i]["vectors"][-1], results[j]["vectors"][-1]).item()
        diff = sim_f - sim_e
        arrow = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "→")
        print(f"  {label:<25} {sim_e:>10.3f} {sim_f:>10.3f}   {arrow} {diff:+.3f}")

    # 逐层追踪
    print(f"\n\n逐层追踪：银行 vs 河岸 的 'bank' 相似度")
    print("=" * 50)

    print(f"\n  {'层':<14} {'相似度':>8}  可视化")
    print("  " + "-" * 45)

    for layer in range(len(results[0]["vectors"])):
        sim = cosine_sim(results[0]["vectors"][layer], results[1]["vectors"][layer]).item()
        bar = "█" * int(max(0, sim) * 30)
        name = "Embedding" if layer == 0 else f"Layer {layer:>2}"
        print(f"  {name:<14} {sim:>8.3f}  {bar}")

    print(f"\n  Embedding 层相似度 = 1.000（同一个 token，同一行向量）")
    print(f"  最后一层相似度 < 1.000（上下文不同，语义不同）")
    print("  → Transformer 逐层将'一词一义'的固定向量")
    print("    变成了'一词多义'的上下文感知向量")


# ==================== 实验 2: 另一组多义词 ====================
print(f"\n\n{'=' * 65}")
print("实验 2: 'apple' 在不同语境中")
print("=" * 65)

apple_sentences = [
    ("I ate a red apple for breakfast", "apple", "水果"),
    ("Apple released a new iPhone today", "Apple", "公司"),
    ("The apple pie was delicious and warm", "apple", "水果"),
    ("Apple stock price hit a new record", "Apple", "公司"),
]

apple_results = []
for text, word, meaning in apple_sentences:
    vectors, tokens, pos = get_all_layer_vectors(text, word)
    if vectors:
        apple_results.append({"text": text, "meaning": meaning, "vectors": vectors})
        print(f"  [{meaning}] '{text}'")

if len(apple_results) >= 4:
    print(f"\n  {'对比':<25} {'Embedding':>10} {'最后一层':>10}")
    print("  " + "-" * 50)

    for i, j, label in [(0, 2, "水果 vs 水果"), (1, 3, "公司 vs 公司"), (0, 1, "水果 vs 公司")]:
        sim_e = cosine_sim(apple_results[i]["vectors"][0], apple_results[j]["vectors"][0]).item()
        sim_f = cosine_sim(apple_results[i]["vectors"][-1], apple_results[j]["vectors"][-1]).item()
        print(f"  {label:<25} {sim_e:>10.3f} {sim_f:>10.3f}")


# ==================== 实验 3: 位置编码的影响 ====================
print(f"\n\n{'=' * 65}")
print("实验 3: 位置编码")
print("=" * 65)

pos_embed = model.transformer.wpe.weight.detach()
print(f"位置编码表: {pos_embed.shape}")
print(f"  最大位置: {pos_embed.shape[0]}（= GPT-2 的上下文窗口大小）")
print(f"  维度: {pos_embed.shape[1]}")

# 相邻位置 vs 远距离位置的相似度
print(f"\n相邻位置 vs 远距离位置的余弦相似度：")
print("-" * 45)

position_pairs = [
    (0, 1), (0, 2), (0, 5), (0, 10),
    (0, 50), (0, 100), (0, 500), (0, 1023),
]

for p1, p2 in position_pairs:
    sim = cosine_sim(pos_embed[p1], pos_embed[p2]).item()
    bar = "█" * int(max(0, sim) * 20)
    print(f"  位置 {p1} vs 位置 {p2:>4}: {sim:>7.3f}  {bar}")

print("\n  观察：相邻位置的向量更相似，远距离位置差异更大。")
print("  这让模型能区分'第 1 个词'和'第 100 个词'。")

"""
multi_head.py —— 多头注意力：不同的头关注不同的东西
从零开始理解大模型（四）配套代码

用法：
    python multi_head.py
    python multi_head.py "Thank you very much"
    python multi_head.py "The capital of France is" --layer 11

需要：pip install transformers torch
"""

import sys
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

print("加载模型…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()


def show_all_heads(text, layer=0, focus_position=-1):
    """展示指定层所有头对某个位置的注意力分布"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]

    if focus_position < 0:
        focus_position = len(tokens) + focus_position

    with torch.no_grad():
        outputs = model(input_ids)

    attention_layer = outputs.attentions[layer][0]  # [heads, seq_len, seq_len]

    focus_token = tokens[focus_position]
    print(f"\n输入: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer}，'{focus_token}'（位置 {focus_position}）的注意力分布：")
    print()

    col_w = max(max(len(t) for t in tokens[:focus_position + 1]) + 1, 8)

    # 表头
    print(f"  {'Head':<8}", end="")
    for t in tokens[:focus_position + 1]:
        print(f"{t:>{col_w}}", end="")
    print(f"  {'最关注':>10}")
    print("  " + "-" * (8 + (focus_position + 1) * col_w + 12))

    for head in range(attention_layer.shape[0]):
        att = attention_layer[head, focus_position, :focus_position + 1].numpy()
        max_idx = att.argmax()

        print(f"  Head {head:<3}", end="")
        for j, score in enumerate(att):
            if j == max_idx:
                print(f"  [{score:.2f}]", end="")
                padding = col_w - len(f"  [{score:.2f}]")
                print(" " * max(0, padding), end="")
            else:
                formatted = f"{score:.2f}"
                print(f"{formatted:>{col_w}}", end="")
        print(f"  → '{tokens[max_idx]}'")

    return attention_layer


def head_diversity_analysis(text, layer=0, focus_position=-1):
    """分析 12 个头的多样性——它们是否在看不同的东西"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]

    if focus_position < 0:
        focus_position = len(tokens) + focus_position

    with torch.no_grad():
        outputs = model(input_ids)

    attention_layer = outputs.attentions[layer][0]

    print(f"\n12 个头的关注焦点分析（Layer {layer}，位置 '{tokens[focus_position]}'）：")
    print("-" * 50)

    focus_counts = {}
    for head in range(12):
        att = attention_layer[head, focus_position, :focus_position + 1].numpy()
        max_idx = att.argmax()
        max_token = tokens[max_idx]
        focus_counts[max_token] = focus_counts.get(max_token, 0) + 1

    for token, count in sorted(focus_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 3)
        heads = [str(h) for h in range(12)
                 if tokens[attention_layer[h, focus_position, :focus_position + 1].numpy().argmax()]
                 == token]
        print(f"  '{token}': {count} 个头关注 {bar}  (Head {', '.join(heads)})")

    # 计算头之间的平均余弦相似度（越低越多样）
    att_vectors = []
    for head in range(12):
        att_vectors.append(attention_layer[head, focus_position, :focus_position + 1].numpy())

    similarities = []
    for i in range(12):
        for j in range(i + 1, 12):
            cos_sim = np.dot(att_vectors[i], att_vectors[j]) / (
                    np.linalg.norm(att_vectors[i]) * np.linalg.norm(att_vectors[j]) + 1e-8)
            similarities.append(cos_sim)

    avg_sim = np.mean(similarities)
    print(f"\n  头间平均注意力相似度: {avg_sim:.3f}")
    print(f"  （越低说明各头越多样化，关注不同的东西）")


def compare_layers(text, focus_position=-1, head=0):
    """对比不同层的注意力模式变化"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]

    if focus_position < 0:
        focus_position = len(tokens) + focus_position

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"\n'{tokens[focus_position]}' 在不同层的注意力焦点变化（Head {head}）：")
    print("-" * 55)

    for layer_idx in [0, 3, 6, 9, 11]:
        att = outputs.attentions[layer_idx][0, head, focus_position, :focus_position + 1].numpy()
        max_idx = att.argmax()

        print(f"  Layer {layer_idx:>2}: ", end="")
        for j in range(focus_position + 1):
            score = att[j]
            blocks = " ░▒▓█"
            level = min(int(score * 5), 4)
            print(f"{blocks[level]}", end="")
        print(f"  最关注 '{tokens[max_idx]}' ({att[max_idx]:.2f})")

    print(f"\n  图例: ' '≈0%  ░≈20%  ▒≈40%  ▓≈60%  █≈80%+")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="多头注意力分析")
    parser.add_argument("text", nargs="*",
                        default=["Thank", "you", "very"],
                        help="输入文本")
    parser.add_argument("--layer", "-l", type=int, default=0, help="层（0-11，默认 0）")
    parser.add_argument("--pos", "-p", type=int, default=-1,
                        help="关注哪个位置（-1=最后一个词）")
    args = parser.parse_args()

    text = " ".join(args.text)

    print("=" * 70)
    print(f"多头注意力分析")
    print("=" * 70)

    # 1. 所有头的注意力分布
    show_all_heads(text, layer=args.layer, focus_position=args.pos)

    # 2. 多样性分析
    print()
    head_diversity_analysis(text, layer=args.layer, focus_position=args.pos)

    # 3. 跨层对比
    print()
    compare_layers(text, focus_position=args.pos, head=0)

    print(f"\n\n提示：")
    print(f"  python multi_head.py 'She saw the cat on the mat' --layer 11")
    print(f"  python multi_head.py 'Barack Obama was the president' --layer 6 --pos 3")


if __name__ == "__main__":
    main()

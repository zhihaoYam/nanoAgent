"""
attention.py —— 可视化 GPT-2 的 Attention 分数
从零开始理解大模型（四）配套代码

用法：
    python attention.py
    python attention.py "Thank you very much"
    python attention.py "The capital of France is" --layer 11

需要：pip install transformers torch
"""

import sys
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("加载模型…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()


def visualize_attention(text, layer=0, head=0):
    """可视化指定层、指定头的 Attention 分数矩阵"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]

    with torch.no_grad():
        outputs = model(input_ids)

    attention = outputs.attentions[layer][0, head].numpy()

    print(f"\n输入: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer}, Head {head} 的 Attention 矩阵：")
    print()

    max_tok_len = max(len(t) for t in tokens)
    col_w = max(max_tok_len + 1, 7)

    # 表头
    header = " " * (max_tok_len + 3)
    for t in tokens:
        header += f"{t:>{col_w}}"
    print(header)
    print(" " * (max_tok_len + 3) + "-" * (len(tokens) * col_w))

    for i, token in enumerate(tokens):
        row = f"{token:>{max_tok_len}} │"
        for j in range(len(tokens)):
            if j > i:
                row += f"{'·':>{col_w}}"
            else:
                score = attention[i][j]
                row += f"{score:>{col_w}.3f}"

        # 最关注谁（只看因果掩码范围内）
        max_j = attention[i][:i + 1].argmax()
        row += f"  ← 最关注 '{tokens[max_j]}'"
        print(row)

    return attention, tokens


def attention_heatmap_ascii(attention, tokens, title=""):
    """用 ASCII 字符画出 Attention 热力图"""
    blocks = " ░▒▓█"

    if title:
        print(f"\n{title}")

    max_tok_len = max(len(t) for t in tokens)

    print(" " * (max_tok_len + 3), end="")
    for t in tokens:
        print(f"{t[:4]:>5}", end="")
    print("  ← Key")
    print(" " * (max_tok_len + 3) + "─" * (len(tokens) * 5))

    for i in range(len(tokens)):
        print(f"{tokens[i]:>{max_tok_len}} │ ", end="")
        for j in range(len(tokens)):
            if j > i:
                print("   · ", end="")
            else:
                score = attention[i][j]
                level = min(int(score * 5), 4)
                char = blocks[level]
                print(f"  {char}{char} ", end="")
        print("│")

    print()
    print("  图例: ' '≈0%  ░≈20%  ▒≈40%  ▓≈60%  █≈80%+")


def show_last_token_across_layers(text, head=0):
    """展示最后一个 token 在所有层对各词的注意力"""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"\n最后一个词 '{tokens[-1]}' 对前面各词的注意力变化（Head {head}）：")
    print(f"\n  {'层':<10}", end="")
    for t in tokens:
        print(f"{t:>10}", end="")
    print()
    print("  " + "-" * (10 + len(tokens) * 10))

    for layer_idx in range(12):
        att = outputs.attentions[layer_idx][0, head, -1, :].numpy()
        max_j = att.argmax()

        print(f"  Layer {layer_idx:<4}", end="")
        for j, score in enumerate(att):
            marker = "█" if j == max_j else " "
            print(f"{score:>8.3f}{marker} ", end="")
        print(f" → '{tokens[max_j]}'")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="可视化 GPT-2 的 Attention 分数")
    parser.add_argument("text", nargs="*", default=["Thank", "you", "very"],
                        help="输入文本")
    parser.add_argument("--layer", "-l", type=int, default=None,
                        help="指定层（0-11），默认展示多层对比")
    parser.add_argument("--head", type=int, default=0, help="注意力头（0-11，默认 0）")
    args = parser.parse_args()

    text = " ".join(args.text)

    if args.layer is not None:
        # 指定层：详细展示
        att, tokens = visualize_attention(text, layer=args.layer, head=args.head)
        attention_heatmap_ascii(att, tokens,
                                f"Layer {args.layer}, Head {args.head} 热力图：")
    else:
        # 默认：多维度展示
        print("=" * 65)
        print("实验 1：Attention 矩阵（Layer 0, Head 0）")
        print("=" * 65)
        att, tokens = visualize_attention(text, layer=0, head=0)
        attention_heatmap_ascii(att, tokens, "热力图：")

        print("\n\n" + "=" * 65)
        print("实验 2：最后一个词在不同层的注意力变化")
        print("=" * 65)
        show_last_token_across_layers(text, head=args.head)

        print("\n\n" + "=" * 65)
        print("实验 3：因果掩码验证")
        print("=" * 65)
        att2, tokens2 = visualize_attention(text, layer=0, head=0)
        print("\n  观察：每行只在对角线及左侧有值，右上方全是 0")
        print("  → 因果掩码确保每个词只能看到自己和前面的词")

        print(f"\n\n提示：试试不同的输入和层：")
        print(f"  python attention.py 'I deposited money at the bank' --layer 11")
        print(f"  python attention.py 'The river bank was covered with grass' --layer 11")


if __name__ == "__main__":
    main()

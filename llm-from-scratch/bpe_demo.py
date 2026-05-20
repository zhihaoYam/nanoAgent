"""
bpe_demo.py —— 从零手写 BPE，看看词表是怎么训练出来的
从零开始理解大模型（二）配套代码

用法：
    python bpe_demo.py
    python bpe_demo.py --merges 15
    python bpe_demo.py --text "apple apple apple banana banana cherry"

需要：无额外依赖（纯 Python）
"""

import argparse
from collections import Counter


def get_pairs(tokens_list):
    """统计所有相邻 token 对的出现次数"""
    pairs = Counter()
    for tokens in tokens_list:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs


def merge_pair(tokens_list, pair, new_token):
    """在所有序列中，把指定的 pair 合并成 new_token"""
    result = []
    for tokens in tokens_list:
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        result.append(merged)
    return result


def train_bpe(text, num_merges=10):
    """
    训练一个最简 BPE 分词器。

    参数：
        text: 训练文本
        num_merges: 合并几次（决定最终词表大小）

    返回：
        vocab: 最终词表
        merge_rules: 合并规则列表（按顺序）
    """
    # 把每个词拆成单个字符 + 词尾标记
    words = text.split()
    tokens_list = [list(word) + ["</w>"] for word in words]

    # 初始词表
    vocab = set()
    for tokens in tokens_list:
        vocab.update(tokens)

    print(f"训练文本: '{text}'")
    print(f"词的数量: {len(words)}")
    print(f"初始词表 ({len(vocab)} 个): {sorted(vocab)}")
    print(f"\n开始 BPE 合并（共 {num_merges} 轮）：")
    print("=" * 65)

    merge_rules = []

    for step in range(num_merges):
        pairs = get_pairs(tokens_list)
        if not pairs:
            print(f"\n  轮次 {step + 1}: 没有更多的 pair 可以合并了，提前停止")
            break

        best_pair = pairs.most_common(1)[0]
        pair, count = best_pair
        new_token = pair[0] + pair[1]

        print(f"\n  轮次 {step + 1}: 合并 ('{pair[0]}', '{pair[1]}') → '{new_token}'  "
              f"（出现 {count} 次）")

        tokens_list = merge_pair(tokens_list, pair, new_token)
        vocab.add(new_token)
        merge_rules.append((pair, new_token))

        # 展示前几个词的当前分词状态
        preview = ["|".join(t) for t in tokens_list[:8]]
        if len(tokens_list) > 8:
            preview.append("...")
        print(f"  当前: {preview}")

    print("\n" + "=" * 65)
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    print(f"最终词表 ({len(vocab)} 个):")
    for i, v in enumerate(sorted_vocab[:20]):
        print(f"  {v}")
    if len(sorted_vocab) > 20:
        print(f"  ... 以及 {len(sorted_vocab) - 20} 个更短的 token")

    return vocab, merge_rules


def tokenize_with_bpe(word, merge_rules):
    """用训练好的 BPE 规则对新词进行分词"""
    tokens = list(word) + ["</w>"]

    for pair, new_token in merge_rules:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                tokens = tokens[:i] + [new_token] + tokens[i + 2:]
            else:
                i += 1

    return tokens


def main():
    parser = argparse.ArgumentParser(description="从零手写 BPE 分词算法")
    parser.add_argument("--text", type=str, default=None, help="自定义训练文本")
    parser.add_argument("--merges", type=int, default=10, help="合并轮数（默认 10）")
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = ("low low low low low "
                "lower lower "
                "newest newest newest newest newest newest "
                "widest widest widest")

    vocab, rules = train_bpe(text, num_merges=args.merges)

    # ==================== 用训练好的分词器切新词 ====================
    print("\n\n对新词进行分词（包括训练时未见过的词）：")
    print("-" * 45)

    # 从训练文本中提取词 + 添加几个新词
    seen_words = list(set(text.split()))
    unseen_words = ["lowest", "newer", "news", "wide", "slower"]

    for word in seen_words:
        tokens = tokenize_with_bpe(word, rules)
        print(f"  '{word}'  → {tokens}  （训练时见过）")

    print()
    for word in unseen_words:
        tokens = tokenize_with_bpe(word, rules)
        print(f"  '{word}'  → {tokens}  （训练时没见过！）")

    print("\n观察要点：")
    print("  1. 高频词（如 'low'）变成了完整的 token")
    print("  2. 低频词被切成了多个子词碎片")
    print("  3. 没见过的词也能用已有的 token 组合来表示")
    print("  4. 合并规则的顺序很重要——先学到的子词会被优先使用")


if __name__ == "__main__":
    main()

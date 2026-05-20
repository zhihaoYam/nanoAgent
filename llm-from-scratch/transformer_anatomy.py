"""
transformer_anatomy.py —— 拆开 GPT-2，看 Transformer 的完整结构
从零开始理解大模型（五）配套代码

用法：
    python transformer_anatomy.py
    python transformer_anatomy.py "I love programming"

需要：pip install transformers torch
"""

import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("加载模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The capital of France is"


# ==================== 1. 模型总览 ====================
print("\n" + "=" * 70)
print("GPT-2 模型结构总览")
print("=" * 70)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,} ({total_params / 1e6:.1f}M)")

# 按模块统计
embedding_params = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
layer0 = model.transformer.h[0]
attn_params = sum(p.numel() for n, p in layer0.named_parameters() if 'attn' in n)
ffn_params = sum(p.numel() for n, p in layer0.named_parameters() if 'mlp' in n)
norm_params = sum(p.numel() for n, p in layer0.named_parameters() if 'ln' in n)
final_norm_params = sum(p.numel() for p in model.transformer.ln_f.parameters())

categories = [
    ("Token Embedding", model.transformer.wte.weight.numel()),
    ("Position Embedding", model.transformer.wpe.weight.numel()),
    ("Attention (每层)", attn_params),
    ("FFN (每层)", ffn_params),
    ("LayerNorm (每层)", norm_params),
    ("Attention (12层)", attn_params * 12),
    ("FFN (12层)", ffn_params * 12),
    ("最终 LayerNorm", final_norm_params),
]

print(f"\n  {'组件':<25} {'参数量':>12} {'占比':>8}")
print("  " + "-" * 50)

for name, count in categories:
    pct = count / total_params * 100
    bar = "█" * int(pct / 2)
    print(f"  {name:<25} {count:>12,} {pct:>7.1f}% {bar}")


# ==================== 2. 单层详细结构 ====================
print(f"\n\n{'=' * 70}")
print("一层 Transformer 的内部结构（Layer 0）")
print("=" * 70)

layer_total = sum(p.numel() for p in layer0.parameters())
print(f"\n  单层总参数: {layer_total:,}")
print()

for name, param in layer0.named_parameters():
    pct = param.numel() / layer_total * 100
    print(f"  {name:<40} {str(param.shape):<20} {param.numel():>10,} ({pct:.1f}%)")


# ==================== 3. 数据流追踪 ====================
print(f"\n\n{'=' * 70}")
print(f"数据流追踪: '{text}'")
print("=" * 70)

input_ids = tokenizer.encode(text, return_tensors="pt")
tokens = [tokenizer.decode([id]) for id in input_ids[0]]
print(f"\nTokens: {tokens} (共 {len(tokens)} 个)")

with torch.no_grad():
    # Step 1-3: Embedding
    token_embeds = model.transformer.wte(input_ids)
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    pos_embeds = model.transformer.wpe(positions)
    hidden = token_embeds + pos_embeds

    print(f"\n  ① Token Embedding:    {input_ids.shape} → {token_embeds.shape}")
    print(f"  ② Position Embedding: {positions.shape} → {pos_embeds.shape}")
    print(f"  ③ 相加:               {hidden.shape}")

    # 保存原始 Embedding 用于后续对比
    original = hidden.clone()

    # Step 4-15: 12 层 Transformer
    print(f"\n  逐层变换:")
    print(f"  {'层':<10} {'输入范数':>8} {'Attn后':>8} {'FFN后':>8} {'变化':>8} "
          f"{'vs原始':>8}")
    print("  " + "-" * 55)

    for layer_idx in range(12):
        layer = model.transformer.h[layer_idx]
        input_norm = hidden.norm(dim=-1).mean().item()

        # Attention block
        residual = hidden
        normed = layer.ln_1(hidden)
        attn_out = layer.attn(normed)[0]
        hidden = residual + attn_out
        after_attn = hidden.norm(dim=-1).mean().item()

        # FFN block
        residual = hidden
        normed = layer.ln_2(hidden)
        ffn_out = layer.mlp(normed)
        hidden = residual + ffn_out
        output_norm = hidden.norm(dim=-1).mean().item()

        # vs 原始 Embedding
        cos_sim = torch.nn.functional.cosine_similarity(
            original.flatten(), hidden.flatten(), dim=0
        ).item()

        delta = (output_norm - input_norm) / input_norm * 100
        print(f"  Layer {layer_idx:>2}  {input_norm:>8.1f} {after_attn:>8.1f} "
              f"{output_norm:>8.1f} {delta:>+7.1f}%  {cos_sim:>7.3f}")

    # 最终 LayerNorm + LM Head
    hidden = model.transformer.ln_f(hidden)
    logits = model.lm_head(hidden)

    print(f"\n  ⑯ 最终 LayerNorm → {hidden.shape}")
    print(f"  ⑰ LM Head       → {logits.shape} (768 → 50257)")

    # 预测
    probs = torch.softmax(logits[0, -1, :], dim=0)
    top5_probs, top5_ids = torch.topk(probs, 5)

    print(f"\n  预测 Top 5:")
    for i in range(5):
        tok = tokenizer.decode([top5_ids[i]])
        print(f"    {i + 1}. '{tok}' ({top5_probs[i].item() * 100:.1f}%)")


# ==================== 4. 权重共享验证 ====================
print(f"\n\n{'=' * 70}")
print("权重共享: Embedding 和 LM Head")
print("=" * 70)

shared = model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
print(f"\n  Token Embedding 形状: {model.transformer.wte.weight.shape}")
print(f"  LM Head 形状:         {model.lm_head.weight.shape}")
print(f"  是否共享同一块内存:    {shared}")

if shared:
    print(f"\n  → 输入端和输出端用的是同一个向量空间。")
    print(f"    模型把 'Paris' 映射到某个方向，输出时只要把向量")
    print(f"    推向那个方向，'Paris' 就能获得高概率。")
    print(f"    这也意味着 LM Head 不额外占用参数。")


# ==================== 5. 残差连接效果 ====================
print(f"\n\n{'=' * 70}")
print("残差连接效果: 原始信息在 12 层中的保持程度")
print("=" * 70)

with torch.no_grad():
    token_embeds = model.transformer.wte(input_ids)
    pos_embeds = model.transformer.wpe(positions)
    original = token_embeds + pos_embeds
    hidden = original.clone()

    # 分别看有残差和无残差（模拟）的情况
    print(f"\n  每层输出 vs 原始 Embedding 的余弦相似度:\n")

    for layer_idx in range(12):
        layer = model.transformer.h[layer_idx]

        residual = hidden
        normed = layer.ln_1(hidden)
        attn_out = layer.attn(normed)[0]
        hidden = residual + attn_out

        residual = hidden
        normed = layer.ln_2(hidden)
        ffn_out = layer.mlp(normed)
        hidden = residual + ffn_out

        cos_sim = torch.nn.functional.cosine_similarity(
            original.flatten(), hidden.flatten(), dim=0
        ).item()
        bar = "█" * int(cos_sim * 30)
        print(f"    Layer {layer_idx:>2}: {cos_sim:.3f} {bar}")

    print(f"\n  → 残差连接让原始信息一路保持到最后一层。")
    print(f"    没有它，12 层叠加后原始信号会完全消失。")

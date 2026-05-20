"""
context_window.py —— 上下文窗口与位置编码实验
从零开始理解大模型（八）配套代码

用法：
    python context_window.py

需要：pip install transformers torch numpy
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import math

print("加载模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (v1.norm() * v2.norm())


# ==================== 1. GPT-2 位置编码表 ====================
print("\n" + "=" * 60)
print("实验 1：GPT-2 的位置编码")
print("=" * 60)

pos_embed = model.transformer.wpe.weight.detach()
print(f"\n  位置编码表形状: {pos_embed.shape}")
print(f"  最大位置: {pos_embed.shape[0]}（= GPT-2 的上下文窗口大小）")
print(f"  每个位置的向量维度: {pos_embed.shape[1]}")

# 相邻 vs 远距离位置的相似度
print(f"\n  不同距离的位置编码相似度:")
print(f"  {'位置对':<25} {'余弦相似度':>10}")
print("  " + "-" * 40)

distances = [1, 2, 5, 10, 50, 100, 500, 1023]
for dist in distances:
    sim = cosine_similarity(pos_embed[0], pos_embed[dist]).item()
    bar = "█" * int(max(0, sim) * 20)
    print(f"  位置 0 vs 位置 {dist:>4}:    {sim:>8.3f}  {bar}")

print("\n  → 相邻位置相似度高，远距离位置相似度低")
print("  → 模型靠这种差异来区分'紧挨着'和'隔得远'")


# ==================== 2. 位置编码的可视化 ====================
print(f"\n\n{'=' * 60}")
print("实验 2：位置编码的模式")
print("=" * 60)

# 取前 8 个维度，看前 20 个位置的编码值
print(f"\n  前 20 个位置在前 8 个维度上的值:")
print(f"  {'位置':<6}", end="")
for d in range(8):
    print(f"  dim{d:>2}", end="")
print()
print("  " + "-" * 60)

for pos in range(0, 20):
    print(f"  {pos:<6}", end="")
    for d in range(8):
        val = pos_embed[pos, d].item()
        print(f"  {val:>5.2f}", end="")
    print()

print("\n  → 每个维度随位置变化的模式不同")
print("  → 组合起来，每个位置有唯一的'指纹'")


# ==================== 3. 超出上下文窗口会怎样 ====================
print(f"\n\n{'=' * 60}")
print("实验 3：超出上下文窗口")
print("=" * 60)

# 构造不同长度的输入
test_lengths = [10, 100, 500, 1000, 1024]

for length in test_lengths:
    try:
        input_ids = tokenizer.encode("hello " * length, return_tensors="pt")
        actual_len = input_ids.shape[1]
        if actual_len > 1024:
            # 手动截断来演示
            input_ids = input_ids[:, :1024]
            actual_len = 1024
            status = "截断到 1024"
        else:
            with torch.no_grad():
                outputs = model(input_ids)
            status = "正常"
        print(f"  {length:>5} 个 'hello' → {actual_len:>5} tokens → {status}")
    except Exception as e:
        print(f"  {length:>5} 个 'hello' → 错误: {type(e).__name__}")

# 尝试超过 1024
print(f"\n  尝试输入 1100 个 token:")
try:
    long_input = torch.zeros(1, 1100, dtype=torch.long)
    with torch.no_grad():
        model(long_input)
    print(f"  → 成功（不应该发生）")
except Exception as e:
    print(f"  → {type(e).__name__}: 超出位置编码表范围!")
    print(f"  → GPT-2 的窗口被位置编码硬性锁死在 1024")


# ==================== 4. RoPE 模拟 ====================
print(f"\n\n{'=' * 60}")
print("实验 4：RoPE（旋转位置编码）模拟")
print("=" * 60)

def apply_rope(x, position, theta_base=10000.0):
    """
    对向量 x 应用 RoPE 旋转。
    position: 位置索引
    theta_base: 基础频率
    """
    d = x.shape[-1]
    # 不同维度用不同的旋转频率
    freqs = 1.0 / (theta_base ** (torch.arange(0, d, 2).float() / d))
    angles = position * freqs
    
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    # 把 x 的相邻维度配对旋转
    x_pairs = x.reshape(-1, 2)
    x_rot = torch.stack([
        x_pairs[:, 0] * cos_a - x_pairs[:, 1] * sin_a,
        x_pairs[:, 0] * sin_a + x_pairs[:, 1] * cos_a,
    ], dim=-1).flatten()
    
    return x_rot


# 演示 RoPE 的关键性质
dim = 64
x = torch.randn(dim)

print(f"\n  向量维度: {dim}")
print(f"\n  RoPE 的关键性质：相对距离决定相似度")
print(f"  {'位置对':<30} {'内积':>10}  说明")
print("  " + "-" * 55)

# 相同相对距离，不同绝对位置
pairs = [
    (0, 1, "距离 1"),
    (100, 101, "距离 1（不同绝对位置）"),
    (10000, 10001, "距离 1（非常远的位置）"),
    (0, 10, "距离 10"),
    (500, 510, "距离 10（不同绝对位置）"),
    (0, 100, "距离 100"),
    (0, 1000, "距离 1000"),
]

for pos_a, pos_b, label in pairs:
    x_a = apply_rope(x, pos_a)
    x_b = apply_rope(x, pos_b)
    dot = torch.dot(x_a, x_b).item()
    print(f"  位置 {pos_a:>5} vs {pos_b:>5}:    {dot:>8.2f}  {label}")

print(f"""
  → 距离 1 的三组（0-1, 100-101, 10000-10001）内积几乎一样
  → 这就是"相对位置编码"的含义：只看距离，不看绝对位置
  → 所以 RoPE 天然支持任意长度——位置 100000 和位置 1 用同一个公式
""")


# ==================== 5. KV Cache 显存估算 ====================
print(f"{'=' * 60}")
print("实验 5：不同上下文长度的 KV Cache 显存需求")
print("=" * 60)

def kv_cache_size_mb(n_tokens, n_layers, n_heads, head_dim, dtype_bytes=2):
    """计算 KV Cache 的显存占用（MB）"""
    return n_tokens * n_layers * n_heads * head_dim * 2 * dtype_bytes / (1024 * 1024)

# 不同模型配置
configs = [
    ("GPT-2",       12, 12,  64, 2),
    ("LLaMA 7B",    32, 32, 128, 2),
    ("LLaMA 70B",   80, 64, 128, 2),
]

context_lengths = [1024, 4096, 32768, 131072, 1000000]

print(f"\n  KV Cache 显存（float16）:")
print(f"\n  {'模型':<14}", end="")
for ctx in context_lengths:
    if ctx >= 1000000:
        label = f"{ctx//1000000}M"
    elif ctx >= 1024:
        label = f"{ctx//1024}K"
    else:
        label = str(ctx)
    print(f"  {label:>8}", end="")
print()
print("  " + "-" * (14 + len(context_lengths) * 10))

for name, layers, heads, head_dim, dtype in configs:
    print(f"  {name:<14}", end="")
    for ctx in context_lengths:
        size_mb = kv_cache_size_mb(ctx, layers, heads, head_dim, dtype)
        if size_mb >= 1024:
            print(f"  {size_mb/1024:>6.1f}GB", end="")
        else:
            print(f"  {size_mb:>6.0f}MB", end="")
    print()

print(f"""
  → 上下文长度和 KV Cache 显存成正比
  → LLaMA 7B 在 128K 上下文时，KV Cache 就要约 64 GB
  → 这就是长上下文窗口的核心工程挑战
""")


# ==================== 6. 上下文长度对预测质量的影响 ====================
print(f"{'=' * 60}")
print("实验 6：上下文长度对预测的影响")
print("=" * 60)

# 同一个问题，给不同长度的前缀
target = "The capital of France is"
filler = "This is some random text to fill the context. " 

print(f"\n  目标句子: '{target}'")
print(f"  在不同长度的填充文本后面放这句话，看模型的预测变化:\n")

for filler_count in [0, 5, 20, 50]:
    prefix = filler * filler_count + target
    input_ids = tokenizer.encode(prefix, return_tensors="pt")
    
    # 确保不超过 GPT-2 的 1024 限制
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, -1024:]  # 取最后 1024 个 token
    
    total_tokens = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids)
        probs = torch.softmax(outputs.logits[0, -1, :], dim=0)
        top5_probs, top5_ids = torch.topk(probs, 5)
    
    top1_token = tokenizer.decode([top5_ids[0]])
    top1_prob = top5_probs[0].item() * 100
    
    print(f"  填充 {filler_count:>2} 段（共 {total_tokens:>4} tokens）: "
          f"Top1 = '{top1_token}' ({top1_prob:.1f}%)")

print(f"\n  → 填充文本越多，模型的注意力越分散")
print(f"  → 即使目标句子不变，长上下文也可能影响预测质量")

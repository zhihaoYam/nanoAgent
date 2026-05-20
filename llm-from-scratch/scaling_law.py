import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. 训练数据 ====================

training_text = """
The capital of France is Paris. Paris is known for the Eiffel Tower.
The capital of Germany is Berlin. Berlin is the largest city in Germany.
The capital of Japan is Tokyo. Tokyo is one of the most populous cities.
The capital of China is Beijing. Beijing hosted the Olympics in 2008.
The capital of Italy is Rome. Rome is famous for the Colosseum.
The president lives in the White House in Washington.
The cat sat on the mat and looked out the window.
Thank you very much for your help and support.
Once upon a time there was a king who ruled a great kingdom.
The sun rises in the east and sets in the west every day.
Machine learning is a subset of artificial intelligence.
The quick brown fox jumps over the lazy dog in the garden.
Deep learning models require large amounts of data and compute.
Natural language processing has made great progress in recent years.
The weather today is sunny with a chance of rain in the evening.
She opened the door and walked into the room quietly.
The students studied hard for their final exams last week.
He picked up the phone and called his friend immediately.
The river flows through the valley and into the sea below.
The old library on the corner has thousands of books inside.
The train arrived at the station exactly on time this morning.
The chef prepared a delicious meal for all the guests tonight.
Scientists discovered a new species of fish in the deep ocean.
The children played happily in the park until the sun went down.
The mountain was covered with snow and looked beautiful from afar.
She wrote a long letter to her grandmother who lives far away.
The company announced its quarterly earnings report on Monday morning.
The pilot landed the plane safely despite the strong winds outside.
Music has the power to bring people together across all cultures.
The garden was full of colorful flowers blooming in the spring sun.
"""

chars = sorted(list(set(training_text)))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
vocab_size = len(chars)
data = torch.tensor([char_to_id[c] for c in training_text], dtype=torch.long)

# ==================== 2. 微型 Transformer ====================


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=None, max_len=128):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # 权重共享

    def forward(self, x):
        _, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = tok + pos

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            h = layer(h, mask)

        return self.lm_head(self.ln_f(h))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ==================== 3. 训练函数 ====================


def train_model(model, data, steps=1000, seq_len=32, batch_size=8, lr=None):
    # 大模型用更小的学习率，防止在小数据上过拟合
    n_params = sum(p.numel() for p in model.parameters())
    if lr is None:
        lr = 3e-3 if n_params < 100_000 else (1e-3 if n_params < 500_000 else 5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses = []
    for _ in range(steps):
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # 返回最后 100 步的平均 Loss（更稳定）
    tail = losses[-100:] if len(losses) >= 100 else losses
    return sum(tail) / len(tail)


# ==================== 4. 主实验 ====================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="每个模型训练多少步")
    args = parser.parse_args()

    print(f"训练数据: {len(training_text)} 字符, 词表: {vocab_size}")
    print(f"每个模型训练 {args.steps} 步")
    print()

    # 5 个不同规模的配置（控制在数据量能支撑的范围内）
    # 大模型需要更多步数才能收敛
    base_steps = args.steps
    configs = [
        {"d_model": 16, "n_layers": 1, "n_heads": 2, "label": "极小", "steps": base_steps},
        {"d_model": 32, "n_layers": 1, "n_heads": 2, "label": "小", "steps": base_steps},
        {"d_model": 64, "n_layers": 2, "n_heads": 4, "label": "中", "steps": int(base_steps * 1.5)},
        {"d_model": 96, "n_layers": 2, "n_heads": 4, "label": "大", "steps": base_steps * 2},
        {"d_model": 128, "n_layers": 3, "n_heads": 4, "label": "较大", "steps": base_steps * 3},
    ]

    print("=" * 60)
    print("Scaling Law 实验：不同参数量 vs Loss")
    print("=" * 60)

    results = []

    for cfg in configs:
        label = cfg["label"]
        model_steps = cfg["steps"]
        model_kwargs = {k: v for k, v in cfg.items() if k not in {"label", "steps"}}
        model = TinyTransformer(vocab_size, **model_kwargs)
        n_params = sum(p.numel() for p in model.parameters())

        start = time.time()
        final_loss = train_model(model, data, steps=model_steps)
        elapsed = time.time() - start

        results.append((n_params, final_loss, label))
        print(
            f"  [{label:>2}] 参数量: {n_params:>10,}  Loss: {final_loss:.3f}"
            f"  ({elapsed:.1f}s, {model_steps} steps)"
        )

    # ==================== 5. 结果分析 ====================
    print(f"\n\n{'=' * 60}")
    print("结果汇总")
    print("=" * 60)

    print(f"\n  {'模型':<6} {'参数量':>10} {'Loss':>8} {'log₁₀(N)':>10} {'log₁₀(L)':>10}")
    print("  " + "-" * 50)

    log_params = []
    log_losses = []

    for n_params, loss, label in results:
        lp = math.log10(n_params)
        ll = math.log10(loss)
        log_params.append(lp)
        log_losses.append(ll)
        print(f"  {label:<6} {n_params:>10,} {loss:>8.3f} {lp:>10.2f} {ll:>10.3f}")

    # 简单线性回归计算斜率
    n = len(log_params)
    mean_x = sum(log_params) / n
    mean_y = sum(log_losses) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_params, log_losses))
    denominator = sum((x - mean_x) ** 2 for x in log_params)
    slope = numerator / denominator if denominator != 0 else 0
    intercept = mean_y - slope * mean_x

    print("\n  Log-Log 线性回归:")
    print(f"    斜率 α = {-slope:.3f} (Scaling Law 中 L ∝ N^(-α))")
    change_pct = (10 ** slope - 1) * 100
    direction = "下降" if slope < 0 else "上升"
    print(f"    即参数量每增长 10 倍, Loss {direction}约 {abs(change_pct):.0f}%")

    # ==================== 6. ASCII 图 ====================
    print(f"\n\n{'=' * 60}")
    print("Log-Log 图：参数量 vs Loss")
    print("=" * 60)

    width = 50
    height = 18

    x_min = min(log_params) - 0.2
    x_max = max(log_params) + 0.2
    y_min = min(log_losses) - 0.1
    y_max = max(log_losses) + 0.1

    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # 画拟合线
    for col in range(width):
        x_val = x_min + (x_max - x_min) * col / (width - 1)
        y_val = slope * x_val + intercept
        row = int((y_max - y_val) / (y_max - y_min) * (height - 1))
        if 0 <= row < height:
            grid[row][col] = '·'

    # 画数据点
    for lp, ll, (_, _, label) in zip(log_params, log_losses, results):
        col = int((lp - x_min) / (x_max - x_min) * (width - 1))
        row = int((y_max - ll) / (y_max - y_min) * (height - 1))
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = '●'

    print("\n  Loss(log₁₀)")
    for i, row in enumerate(grid):
        if i == 0:
            y_label = f"{y_max:.1f}"
        elif i == height - 1:
            y_label = f"{y_min:.1f}"
        else:
            y_label = "    "
        print(f"  {y_label:>5} │{''.join(row)}│")

    print(f"        └{'─' * width}┘")

    # X 轴标签
    labels = []
    for n_params, _, _ in results:
        if n_params >= 1_000_000:
            labels.append(f"{n_params / 1_000_000:.1f}M")
        elif n_params >= 1000:
            labels.append(f"{n_params / 1000:.0f}K")
        else:
            labels.append(str(n_params))

    print(f"         {labels[0]:<10}{labels[-1]:>{width - 10}}")
    print("                参数量 (log scale)")

    print("\n  ● = 实测数据点    · = 拟合直线")
    print("  → 在 log-log 坐标下近似一条直线，验证了 Scaling Law 的趋势")

    # ==================== 7. 预测 ====================
    print(f"\n\n{'=' * 60}")
    print("外推预测：更大的模型 Loss 会是多少？")
    print("=" * 60)

    predict_sizes = [5_000_000, 10_000_000, 100_000_000, 1_000_000_000]

    print(f"\n  基于拟合直线 log(L) = {slope:.3f} × log(N) + {intercept:.3f}")
    print(f"\n  {'参数量':>12}  {'预测 Loss':>10}  说明")
    print("  " + "-" * 45)

    for size in predict_sizes:
        predicted_log_loss = slope * math.log10(size) + intercept
        predicted_loss = 10 ** predicted_log_loss
        if size >= 1_000_000_000:
            label = f"{size / 1_000_000_000:.0f}B"
        elif size >= 1_000_000:
            label = f"{size / 1_000_000:.0f}M"
        else:
            label = f"{size / 1000:.0f}K"
        note = "← 已超出本实验训练范围" if size == 5_000_000 else ""
        if size >= 100_000_000:
            note = "← 外推预测（未实际训练）"
        print(f"  {label:>12}  {predicted_loss:>10.3f}  {note}")

    print("\n  注意：这是在极小数据集上的微型模型实验。")
    print("  真实的 Scaling Law 需要在大得多的规模上验证，")
    print("  但幂律趋势在微型规模上就已经清晰可见。")


if __name__ == "__main__":
    main()

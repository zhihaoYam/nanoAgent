"""
train_tiny.py —— 亲手训练一个微型语言模型
从零开始理解大模型（六）配套代码

从零开始：随机初始化 → 训练 → 看 Loss 下降 → 验证预测效果
不需要 GPU，CPU 上几分钟就能跑完。

需要：pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ==================== 1. 训练数据 ====================

# 用几句话作为训练数据（极小，只为演示训练过程）

training_text = """
The capital of France is Paris.
The capital of Germany is Berlin.
The capital of Japan is Tokyo.
The capital of China is Beijing.
The capital of Italy is Rome.
The president lives in the White House.
The cat sat on the mat.
Thank you very much for your help.
Once upon a time there was a king.
The sun rises in the east.
"""

# 极简分词：按字符级别

chars = sorted(list(set(training_text)))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
vocab_size = len(chars)

def encode(text):
    return [char_to_id[c] for c in text]

def decode(ids):
    return "".join(id_to_char[i] for i in ids)

data = torch.tensor(encode(training_text), dtype=torch.long)
print(f"训练数据: {len(training_text)} 个字符, 词表大小: {vocab_size}")
print(f"词表: {''.join(chars)}")

# ==================== 2. 微型 Transformer ====================

class TinyTransformer(nn.Module):
    """
    一个极简的 Transformer 语言模型。
    和 GPT-2 结构完全一样，只是小得多。
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=256, max_len=64):
        super().__init__()
        self.d_model = d_model

        # Embedding（第三篇）
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Transformer 层（第四、五篇）
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # 最终 LayerNorm + LM Head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（第五篇第九节）
        self.lm_head.weight = self.token_emb.weight
        
        # 参数量统计
        total = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {total:,} ({total/1000:.1f}K)")

    def forward(self, x):
        B, T = x.shape
        
        # Embedding + 位置编码
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = tok + pos
        
        # 因果掩码（第四篇第七节）
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # N 层 Transformer
        for layer in self.layers:
            h = layer(h, mask)
        
        # 输出
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

class TransformerLayer(nn.Module):
    """一层 Transformer = Attention + FFN + 残差 + LayerNorm"""
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
        # Attention + 残差（第五篇第四节）
        x = x + self.attn(self.ln1(x), mask)
        # FFN + 残差
        x = x + self.ffn(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力（第四篇）"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, T, C = x.shape
        
        # 生成 Q, K, V（第四篇第三节）
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 打分（第四篇第四节 4.1）
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # 因果掩码
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax 归一化（第四篇第四节 4.2）
        weights = F.softmax(scores, dim=-1)
        
        # 加权求和（第四篇第四节 4.3）
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# ==================== 3. 训练循环 ====================

model = TinyTransformer(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

seq_len = 32     # 每次看 32 个字符
batch_size = 8   # 每次 8 个样本
n_steps = 500    # 训练 500 步

print(f"\n训练配置: seq_len={seq_len}, batch_size={batch_size}, steps={n_steps}")
print(f"{'=' * 60}")

loss_history = []
start_time = time.time()

for step in range(n_steps):
    # 随机抽取训练片段
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])       # 输入
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])   # 正确答案（下一个字符）

    # 前向传播：预测
    logits = model(x)

    # 计算 Loss（本篇第三节）
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    # 反向传播：算梯度（本篇第四节）
    optimizer.zero_grad()
    loss.backward()

    # 更新参数：梯度下降（本篇第四节）
    optimizer.step()

    loss_value = loss.item()
    loss_history.append(loss_value)

    if step % 50 == 0 or step == n_steps - 1:
        elapsed = time.time() - start_time
        bar = "█" * int((1 - loss_value / 4) * 30) if loss_value < 4 else ""
        print(f"  Step {step:>4}: Loss = {loss_value:.3f}  "
              f"({elapsed:.1f}s)  {bar}")

elapsed = time.time() - start_time
print(f"\n训练完成! 耗时 {elapsed:.1f} 秒")
print(f"Loss: {loss_history[0]:.3f} → {loss_history[-1]:.3f}")

# ==================== 4. 验证：训练前 vs 训练后 ====================

print(f"\n\n{'=' * 60}")
print("训练效果验证")
print("=" * 60)

def generate(model, prompt, max_len=50, temperature=0.8):
    """用训练好的模型生成文本"""
    model.eval()
    ids = encode(prompt)

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ids[-32:]], dtype=torch.long)  # 最多看 32 个字符
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            
            if id_to_char.get(next_id) == '\n':
                break

    return decode(ids)

prompts = [
    "The capital of ",
    "Thank you very",
    "The cat sat",
    "Once upon a",
]

for prompt in prompts:
    if all(c in char_to_id for c in prompt):
        result = generate(model, prompt, max_len=40)
        print(f"\n  输入: '{prompt}'")
        print(f"  输出: '{result}'")

# ==================== 5. Loss 曲线 ====================

print(f"\n\n{'=' * 60}")
print("Loss 下降曲线")
print("=" * 60)

height = 15
width = 50
max_loss = max(loss_history[:20])  # 用前几步的最大值做上限
min_loss = min(loss_history)

print(f"\n  Loss")
print(f"  {max_loss:.1f} ┤")

for row in range(height):
    threshold = max_loss - (max_loss - min_loss) * (row + 1) / height
    line = "  "
    if row == height - 1:
        line += f"{min_loss:.1f} ┤"
    else:
        line += "     │"

    for col in range(width):
        idx = int(col * len(loss_history) / width)
        if idx < len(loss_history):
            if loss_history[idx] >= threshold:
                line += "█"
            else:
                line += " "

    print(line)

print("     └" + "─" * width)
print(f"      Step 0{' ' * (width - 12)}Step {n_steps}")
print(f"\n  Loss 从 {loss_history[0]:.2f} 下降到 {loss_history[-1]:.2f}")
print(f'  模型从"完全瞎猜"变成了"能续写训练数据中的句子"')

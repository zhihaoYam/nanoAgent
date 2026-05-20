"""
predict.py —— 亲眼看到"下一个词预测"
从零开始理解大模型（一）配套代码

用法：
    python predict.py
    python predict.py "The president of the United States is"
    python predict.py "Once upon a time"

需要：pip install transformers torch
"""

import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ==================== 1. 加载模型和分词器 ====================

print("正在加载模型（首次运行会下载约 500MB）…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ==================== 2. 输入一句话 ====================

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Thank you very"
print(f"\n输入: '{prompt}'")

# 把文字变成模型能懂的数字（token ID）
input_ids = tokenizer.encode(prompt, return_tensors="pt")
tokens = [tokenizer.decode(id) for id in input_ids[0]]
print(f"Token IDs: {input_ids.tolist()[0]}")
print(f"对应的 tokens: {tokens}")
print(f"Token 数量: {len(tokens)}")

# ==================== 3. 模型预测下一个词 ====================

with torch.no_grad():
    outputs = model(input_ids)
    # outputs.logits 的形状: [1, token数量, 词表大小(50257)]
    # 我们只关心最后一个位置的预测（即"下一个词"）
    next_token_logits = outputs.logits[0, -1, :]

print(f"\n词表大小: {next_token_logits.shape[0]} 个 token")

# ==================== 4. 看看模型觉得哪些词最可能 ====================

probabilities = torch.softmax(next_token_logits, dim=0)

top_k = 10
top_probs, top_indices = torch.topk(probabilities, top_k)

print(f"\n模型预测 '{prompt}' 后面最可能的 {top_k} 个词：")
print("-" * 50)
for i in range(top_k):
    token = tokenizer.decode(top_indices[i])
    prob = top_probs[i].item() * 100
    bar = "█" * int(prob / 2)
    print(f"  {i+1:2d}. '{token}' \t {prob:5.1f}%  {bar}")

# ==================== 5. 选择概率最高的词 ====================

best_token = tokenizer.decode(top_indices[0])
print(f"\n选择概率最高的: '{best_token}'")
print(f"拼接后: '{prompt}{best_token}'")

# ==================== 6. 额外信息 ====================

# 看看概率分布的集中程度
top1_prob = top_probs[0].item()
top5_prob = top_probs[:5].sum().item()
top10_prob = top_probs[:10].sum().item()

print(f"\n概率分布统计:")
print(f"  Top 1 占: {top1_prob*100:.1f}%")
print(f"  Top 5 占: {top5_prob*100:.1f}%")
print(f"  Top 10 占: {top10_prob*100:.1f}%")
print(f"  剩余 {next_token_logits.shape[0] - 10} 个 token 共占: {(1-top10_prob)*100:.1f}%")

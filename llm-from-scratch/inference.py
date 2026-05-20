"""
inference.py —— 对比推理过程中有无 KV Cache 的区别
从零开始理解大模型（七）配套代码
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

print("加载模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "The meaning of life is"
max_new_tokens = 30

# ==================== 方式 1：不用 KV Cache（每次重算全部） ====================

print(f"\n{'=' * 60}")
print(f"方式 1：无 KV Cache（每步重算所有 token）")
print(f"{'=' * 60}")

input_ids = tokenizer.encode(prompt, return_tensors="pt")
generated = input_ids.clone()
prompt_len = input_ids.shape[1]

start = time.time()
step_times = []

with torch.no_grad():
    for step in range(max_new_tokens):
        step_start = time.time()

        # 每次把 全部已有 token 送入模型（无缓存）
        outputs = model(generated)
        next_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)

        token_text = tokenizer.decode(next_token[0])
        total_tokens = generated.shape[1]
        print(f"  Step {step+1:>2}: '{token_text}' "
              f"({step_time:.0f}ms, 处理了 {total_tokens} 个 token)")

        if next_token.item() == tokenizer.eos_token_id:
            break

no_cache_total = time.time() - start
no_cache_text = tokenizer.decode(generated[0])

print(f"\n  总耗时: {no_cache_total*1000:.0f}ms")
print(f"  平均每步: {sum(step_times)/len(step_times):.0f}ms")
print(f"  输出: '{no_cache_text}'")

# ==================== 方式 2：使用 KV Cache ====================

print(f"\n{'=' * 60}")
print(f"方式 2：有 KV Cache（每步只处理新 token）")
print(f"{'=' * 60}")

input_ids = tokenizer.encode(prompt, return_tensors="pt")
generated_ids = input_ids.clone()

start = time.time()
step_times_cached = []
past_kv = None

with torch.no_grad():
    for step in range(max_new_tokens):
        step_start = time.time()

        if past_kv is None:
            # Prefill：第一次，处理全部输入
            outputs = model(input_ids, use_cache=True)
            phase = "Prefill"
            tokens_processed = input_ids.shape[1]
        else:
            # Decode：后续，只处理最后一个新 token
            outputs = model(next_token_ids, past_key_values=past_kv, use_cache=True)
            phase = "Decode"
            tokens_processed = 1

        past_kv = outputs.past_key_values
        next_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_logits)
        next_token_ids = next_token_id.unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

        step_time = (time.time() - step_start) * 1000
        step_times_cached.append(step_time)

        token_text = tokenizer.decode(next_token_id)
        print(f"  Step {step+1:>2}: '{token_text}' "
              f"({step_time:.0f}ms, {phase}, 处理了 {tokens_processed} 个 token)")

        if next_token_id.item() == tokenizer.eos_token_id:
            break

cache_total = time.time() - start
cache_text = tokenizer.decode(generated_ids[0])

print(f"\n  总耗时: {cache_total*1000:.0f}ms")
print(f"  Prefill: {step_times_cached[0]:.0f}ms")
print(f"  平均 Decode: {sum(step_times_cached[1:])/max(len(step_times_cached)-1,1):.0f}ms")
print(f"  输出: '{cache_text}'")

# ==================== 对比 ====================

print(f"\n{'=' * 60}")
print(f"对比")
print(f"{'=' * 60}")

speedup = no_cache_total / cache_total if cache_total > 0 else 0
print(f"\n  无 Cache 总耗时: {no_cache_total*1000:.0f}ms")
print(f"  有 Cache 总耗时: {cache_total*1000:.0f}ms")
print(f"  加速比: {speedup:.1f}x")

# KV Cache 大小估算

if past_kv is not None:
    n_layers = len(past_kv)
    # past_kv[layer] = (key, value), 各自形状 [batch, heads, seq_len, head_dim]
    k_shape = past_kv[0][0].shape
    cache_elements = sum(
        past_kv[l][0].numel() + past_kv[l][1].numel()
        for l in range(n_layers)
    )
    cache_bytes = cache_elements * 4  # float32
    print(f"\n  KV Cache 统计:")
    print(f"    层数: {n_layers}")
    print(f"    K/V 形状: {k_shape} (每层)")
    print(f"    缓存的 token 数: {k_shape[2]}")
    print(f"    总元素数: {cache_elements:,}")
    print(f"    显存占用: {cache_bytes/1024/1024:.2f} MB")

# ==================== 逐步耗时对比图 ====================

print(f"\n\n{'=' * 60}")
print("逐步耗时对比")
print("=" * 60)

n = min(len(step_times), len(step_times_cached))
print(f"\n  {'Step':<6} {'无Cache':>10} {'有Cache':>10} {'节省':>10}")
print("  " + "-" * 40)

for i in range(n):
    saved = step_times[i] - step_times_cached[i]
    print(f"  {i+1:<6} {step_times[i]:>8.0f}ms {step_times_cached[i]:>8.0f}ms "
          f"{saved:>+8.0f}ms")

print(f"\n  观察:")
print(f"  - 无 Cache: 每步处理的 token 数递增，耗时逐步增长")
print(f"  - 有 Cache: Step 1（Prefill）最慢，后续 Decode 几乎恒定")
print(f"  - 生成越长，有 Cache 的优势越大")

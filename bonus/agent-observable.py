"""
agent-observable.py - 可观测版 Agent
基于 agent.py (115行)，核心新增:

1. AgentLogger —— 每轮循环写 JSON Lines 日志
1. Trace —— 把日志串成因果链（Span 结构）
1. Replay —— 用保存的 messages 快照复现某一轮

用法:

# 正常运行（自动生成日志文件）

python agent-observable.py "列出当前目录的文件"

# 查看日志

cat agent.log.jsonl | python -m json.tool --no-ensure-ascii

# 复现某一轮（详细模式，保存 messages 快照）

python agent-observable.py --verbose "你的任务"
python agent-observable.py --replay agent.trace.json --round 3
"""

import os
import json
import subprocess
import sys
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ==================== 工具定义（和 agent.py 完全一样）====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command on the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    }
]

# ==================== 工具实现（和 agent.py 完全一样）====================

def execute_bash(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

def write_file(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {str(e)}"

available_functions = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file
}

# ==================== AgentLogger：JSON Lines 日志 ====================

class AgentLogger:
    """记录 Agent 每轮循环的关键信息，写入 .jsonl 文件"""

    def __init__(self, log_file="agent.log.jsonl"):
        self.log_file = log_file
        self.task_id = str(int(time.time()))

    def log_round(self, round_num, message_count, llm_content,
                  tool_name=None, tool_args=None, tool_result=None,
                  usage=None, duration_ms=None):
        entry = {
            "task_id": self.task_id,
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message_count": message_count,
            "llm_content": llm_content or "",
            "has_tool_calls": tool_name is not None,
        }

        if tool_name:
            entry["tool"] = {
                "name": tool_name,
                "args": tool_args,
                "result_length": len(tool_result) if tool_result else 0,
                "result_preview": (tool_result[:200] if tool_result else "")
            }

        if usage:
            entry["tokens"] = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens
            }

        if duration_ms is not None:
            entry["duration_ms"] = duration_ms

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# ==================== Trace：因果链 ====================

class Trace:
    """一次 Agent 任务的完整追踪，由多个 Span 组成"""

    def __init__(self, task):
        self.trace_id = str(int(time.time()))
        self.task = task
        self.spans = []
        self.start_time = time.time()

    def add_span(self, round_num, llm_content, tool_name=None,
                 tool_args=None, tool_result=None,
                 duration_ms=None, tokens=None,
                 messages_snapshot=None):
        span = {
            "span_id": f"{self.trace_id}-{round_num}",
            "round": round_num,
            "llm_content": llm_content or "",
            "tool": None,
            "duration_ms": duration_ms,
            "tokens": tokens,
        }

        if tool_name:
            span["tool"] = {
                "name": tool_name,
                "args": tool_args,
                "result_length": len(tool_result) if tool_result else 0,
                "result_preview": (tool_result[:200] if tool_result else "")
            }

        # 详细模式下保存完整 messages，用于 replay
        if messages_snapshot is not None:
            span["messages_snapshot"] = messages_snapshot

        self.spans.append(span)

    def dump(self):
        """输出完整 Trace 数据"""
        total_duration = int((time.time() - self.start_time) * 1000)
        total_tokens = 0
        for s in self.spans:
            if s["tokens"]:
                total_tokens += s["tokens"]["input"] + s["tokens"]["output"]

        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "total_rounds": len(self.spans),
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "spans": self.spans
        }

    def save(self, filepath="agent.trace.json"):
        """保存 Trace 到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.dump(), f, ensure_ascii=False, indent=2)
        print(f"\n[Trace] 已保存到 {filepath}", flush=True)

    def print_summary(self):
        """打印 Trace 摘要"""
        data = self.dump()
        print(f"\n{'='*55}")
        print(f"Trace 摘要  (task_id: {data['trace_id']})")
        print(f"{'='*55}")
        print(f"任务: {data['task']}")
        print(f"总轮次: {data['total_rounds']}")
        print(f"总耗时: {data['total_duration_ms']}ms")
        print(f"总 Token: {data['total_tokens']}")
        print(f"{'-'*55}")
        print(f"{'轮次':<6} {'工具':<16} {'耗时':>8} {'Token':>8}")
        print(f"{'-'*55}")
        for s in self.spans:
            tool_name = s["tool"]["name"] if s["tool"] else "(无工具)"
            dur = f"{s['duration_ms']}ms" if s['duration_ms'] else "-"
            tok = ""
            if s["tokens"]:
                tok = str(s["tokens"]["input"] + s["tokens"]["output"])
            print(f"{s['round']:<6} {tool_name:<16} {dur:>8} {tok:>8}")
        print(f"{'='*55}")

# ==================== Replay：复现某一轮 ====================

def replay_round(trace_file, round_num, temperature=0):
    """
    从保存的 Trace 文件中取出某一轮的 messages 快照，
    重新发送给 LLM，看是否得到相同的结果。

    前提：Trace 必须在 --verbose 模式下生成（包含 messages_snapshot）。
    """
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    target_span = None
    for span in trace_data["spans"]:
        if span["round"] == round_num:
            target_span = span
            break

    if not target_span:
        print(f"[Replay] 未找到轮次 {round_num}")
        return

    if "messages_snapshot" not in target_span:
        print(f"[Replay] 轮次 {round_num} 没有 messages_snapshot。")
        print(f"         请用 --verbose 模式重新运行任务以保存完整 messages。")
        return

    messages = target_span["messages_snapshot"]
    print(f"[Replay] 重放轮次 {round_num}，messages 数量: {len(messages)}")
    print(f"[Replay] temperature={temperature}")
    print(f"{'-'*40}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=temperature
    )

    message = response.choices[0].message

    if message.content:
        print(f"[Replay] LLM 回复: {message.content}")

    if message.tool_calls:
        for tc in message.tool_calls:
            func_name = tc.function.name
            func_args = tc.function.arguments
            print(f"[Replay] 工具调用: {func_name}({func_args})")

    print(f"{'-'*40}")

    # 和原始结果对比
    original_content = target_span["llm_content"]
    original_tool = target_span["tool"]["name"] if target_span["tool"] else None
    replay_tool = message.tool_calls[0].function.name if message.tool_calls else None

    if original_tool == replay_tool:
        print(f"[Replay] 工具调用一致: {replay_tool or '(无)'}")
    else:
        print(f"[Replay] ⚠ 工具调用不一致！原始: {original_tool}, 重放: {replay_tool}")

    return message

# ==================== Agent 核心循环（可观测版）====================

def run_agent(user_message, max_iterations=10, verbose=False):
    logger = AgentLogger()
    trace = Trace(user_message)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can interact with the system. Be concise."},
        {"role": "user", "content": user_message}
    ]

    for i in range(max_iterations):
        round_num = i + 1
        start = time.time()

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )
        message = response.choices[0].message
        duration_ms = int((time.time() - start) * 1000)

        messages.append(message)

        # 详细模式下保存 messages 快照（用于 replay）
        # 注意：只在需要时保存，因为快照可能很大
        snapshot = None
        if verbose:
            snapshot = [
                {"role": m["role"], "content": m.get("content", "")}
                if isinstance(m, dict) else
                {"role": m.role, "content": m.content or ""}
                for m in messages[:-1]  # 不包含刚加入的 assistant 消息
            ]

        if not message.tool_calls:
            # 没有工具调用，任务结束
            logger.log_round(
                round_num, len(messages), message.content,
                usage=response.usage, duration_ms=duration_ms
            )
            trace.add_span(
                round_num, message.content,
                duration_ms=duration_ms,
                tokens={"input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens},
                messages_snapshot=snapshot
            )
            print(f"[轮次 {round_num}] {message.content[:100]}...")

            trace.print_summary()
            if verbose:
                trace.save()
            return message.content

        # 有工具调用
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"[轮次 {round_num}] [工具] {function_name}({function_args})")

            function_response = available_functions[function_name](**function_args)

            # 记录日志
            logger.log_round(
                round_num, len(messages), message.content or "",
                tool_name=function_name,
                tool_args=function_args,
                tool_result=function_response,
                usage=response.usage,
                duration_ms=duration_ms
            )

            # 记录 Trace
            trace.add_span(
                round_num, message.content or "",
                tool_name=function_name,
                tool_args=function_args,
                tool_result=function_response,
                duration_ms=duration_ms,
                tokens={"input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens},
                messages_snapshot=snapshot
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_response
            })

    trace.print_summary()
    if verbose:
        trace.save()
    return "Max iterations reached"

# ==================== 入口 ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python agent-observable.py '你的任务'")
        print("  python agent-observable.py --verbose '你的任务'    # 保存完整 messages 用于 replay")
        print("  python agent-observable.py --replay agent.trace.json --round 3  # 重放第 3 轮")
        sys.exit(1)

    # 解析参数
    args = sys.argv[1:]

    # Replay 模式
    if "--replay" in args:
        replay_idx = args.index("--replay")
        trace_file = args[replay_idx + 1]
        round_idx = args.index("--round")
        round_num = int(args[round_idx + 1])
        replay_round(trace_file, round_num)
        sys.exit(0)

    # 正常运行模式
    verbose = False
    if "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    task = " ".join(args)
    result = run_agent(task, verbose=verbose)
    print(f"\n{result}")

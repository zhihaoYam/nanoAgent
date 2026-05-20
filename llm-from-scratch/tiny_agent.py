"""
tiny_agent.py —— 50 行最小 Agent 实现
从零开始理解大模型（十）配套代码

一个能执行 bash 命令和读写文件的最小 Agent。
展示 Agent = LLM + 工具 + 循环 的完整结构。

用法：
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，可指向任意兼容 API

python tiny_agent.py "帮我看看当前目录下有什么文件"
python tiny_agent.py "创建一个 hello.txt，内容是 Hello World"
python tiny_agent.py "查看系统的 Python 版本"

需要：pip install openai
"""

import json
import os
import subprocess
import sys

from openai import OpenAI

# ==================== 1. LLM 客户端 ====================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "sk-xxx"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
MODEL = os.environ.get("MODEL", "gpt-4o-mini")

# ==================== 2. 工具定义 ====================

# 这段 JSON Schema 会随每次请求发给模型
# 模型读到这个“说明书”后，就“知道”自己能执行命令和读写文件

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command on the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]

# ==================== 3. 工具实现 ====================

# 模型输出 JSON 说“我想调 execute_bash”，这里真正执行

def execute_bash(command):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return output if output else "(命令执行成功，无输出)"
    except Exception as e:
        return f"Error: {e}"


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已成功写入 {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具名 → 函数的映射表
available_tools = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
}

# ==================== 4. Agent 核心循环 ====================


def run_agent(user_message, max_steps=10):
    """
    Agent 的完整循环：
    1. 把用户消息发给 LLM
    2. LLM 输出文本 → 返回给用户，结束
    3. LLM 输出工具调用 → 执行工具 → 把结果加入消息历史 → 回到 1
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": user_message},
    ]

    for step in range(max_steps):
        print(f"\n\033[36m[Step {step + 1}]\033[0m 调用 LLM... ", end="", flush=True)

        # ---- 让大模型“想” ----
        # 这一步背后发生的事情：
        #   分词(第2篇) → Embedding(第3篇) → Attention(第4篇)
        #   → Transformer×N层(第5篇) → 逐token输出(第7篇)
        #   上下文窗口在增长(第8篇)，模型越大越准(第9篇)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # 把模型的回复加入消息历史
        messages.append(assistant_msg)

        # ---- 判断：输出文本还是工具调用？ ----
        if not assistant_msg.tool_calls:
            # 模型输出了普通文本 → 任务结束
            print("回复文本")
            print(f"\n\033[32m{assistant_msg.content}\033[0m")
            return assistant_msg.content

        # ---- 模型输出了工具调用 → 让代码去“做” ----
        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"调用工具 {func_name}({func_args})")

            # 执行工具
            if func_name in available_tools:
                result = available_tools[func_name](**func_args)
            else:
                result = f"未知工具: {func_name}"

            # 打印工具返回（截断过长的输出）
            display = result[:200] + "..." if len(result) > 200 else result
            print(f"  \033[33m→ {display}\033[0m")

            # 把工具结果加入消息历史，让模型在下一轮看到
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    print("\n达到最大步数，停止。")
    return None


# ==================== 5. 入口 ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tiny_agent.py '你的任务'")
        print("示例: python tiny_agent.py '帮我看看当前目录下有什么文件'")
        print()
        print("需要设置环境变量:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export OPENAI_BASE_URL='https://api.openai.com/v1'  # 可选")
        print("  export MODEL='gpt-4o-mini'  # 可选，默认 gpt-4o-mini")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    print(f"任务: {task}")
    run_agent(task)

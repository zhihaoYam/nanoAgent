import os
import json
import subprocess
from openai import OpenAI
from datetime import datetime

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def execute_bash(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"


functions = {"execute_bash": execute_bash, "read_file": read_file, "write_file": write_file}


def save_messages_to_file(messages, filename=None):
    """Save all messages to a JSON file with timestamp"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_messages_{timestamp}.json"
    
    # Convert messages to serializable format
    serializable_messages = []
    for msg in messages:
        # Handle both dict and ChatCompletionMessage objects
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_call_id = msg.get("tool_call_id")
        else:
            # ChatCompletionMessage object
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
            tool_call_id = None
        
        serializable_msg = {"role": role, "content": content}
        
        if tool_call_id:
            serializable_msg["tool_call_id"] = tool_call_id
        
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            serializable_msg["tool_calls"] = []
            for tool_call in msg.tool_calls:
                serializable_msg["tool_calls"].append({
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        serializable_messages.append(serializable_msg)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "messages": serializable_messages
        }, f, indent=2, ensure_ascii=False)
    
    return filename


def run_agent(user_message, max_iterations=15):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": user_message},
    ]
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "qwen3.5-35b-a3b"),
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return message.content, messages
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {name}({args})")
            if name not in functions:
                result = f"Error: Unknown tool '{name}'"
            else:
                result = functions[name](**args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    return "Max iterations reached", messages


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    result, messages = run_agent(task)
    
    # Save all messages to file
    filename = save_messages_to_file(messages)
    print(f"Messages saved to: {filename}")
    
    print(result)

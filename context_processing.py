import os
import json
from datetime import datetime

def save_memory(user_query: str, assistant_answer: str, route: str = "", memory_file: str = ""):
    """将新一轮对话追加保存到本地 JSONL"""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": user_query,
        "assistant": assistant_answer,
        "route": route
    }
    with open(memory_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_recent_memory(max_turns: int, memory_file: str = ""):
    """读取最近 max_turns 轮历史"""
    if not os.path.exists(memory_file):
        return []

    records = []
    with open(memory_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-max_turns:]


def format_history_for_prompt(history_records):
    """根据历史对话格式化成字符串，喂给提示词"""
    if not history_records:
        return "无"
    formatted = []
    for i, item in enumerate(history_records, 1):
        user_text = item.get("user", "")
        assistant_text = item.get("assistant", "")
        formatted.append(
            f"[历史对话{i}]\n"
            f"用户：{user_text}\n"
            f"助手：{assistant_text}"
        )
    return "\n\n".join(formatted)
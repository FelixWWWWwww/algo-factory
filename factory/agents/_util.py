"""Agent 共享工具：健壮的 LLM JSON 解析。"""
import re
import json

try:
    import json_repair
except Exception:  # pragma: no cover
    json_repair = None


def parse_json(text: str):
    """从 LLM 原始输出中提取并解析 JSON（剥离 ```json 包裹 + json_repair 兜底）。

    解析失败返回 None，调用方据此走 fallback。
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    cleaned = m.group(1).strip() if m else text.strip()
    for loader in (json_repair.loads if json_repair else None, json.loads):
        if loader is None:
            continue
        try:
            return loader(cleaned)
        except Exception:
            continue
    return None

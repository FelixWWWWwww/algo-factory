"""
factory/llm/logger.py
记录所有 LLM 调用（prompt / response / tokens）
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMCallLogger:
    """LLM 调用日志记录器"""

    def __init__(self, log_dir: str = "logs/llm"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_call(
            self,
            agent_name: str,
            prompt: str,
            response: str,
            model: str = "unknown",
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            **extra_info
    ) -> Path:
        """
        记录一次 LLM 调用。

        Args:
            agent_name: Agent 名称（如 "Planner", "ModelSelection"）
            prompt: 输入 prompt
            response: LLM 返回的内容
            model: 模型名（如 "gpt-3.5-turbo"）
            prompt_tokens: prompt token 数
            completion_tokens: completion token 数
            **extra_info: 其他信息（如 temperature, top_p 等）

        Returns:
            日志文件路径
        """

        timestamp = datetime.now().isoformat()
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{agent_name}.json"

        log_entry = {
            "timestamp": timestamp,
            "agent": agent_name,
            "model": model,
            "prompt": prompt,
            "response": response,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            **extra_info
        }

        log_file = self.log_dir / filename
        log_file.write_text(
            json.dumps(log_entry, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        logger.info(
            f"[LLM Log] {agent_name} → {log_file.name} "
            f"(tokens={log_entry['tokens']['total']})"
        )

        return log_file


# 全局实例
_llm_logger: Optional[LLMCallLogger] = None


def get_llm_logger() -> LLMCallLogger:
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = LLMCallLogger()
    return _llm_logger

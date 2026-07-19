"""
factory/config.py
全局配置（Day 2 关闭 Mock，接通真实 LLM）
"""

import os
from pathlib import Path

# ===== LLM 配置 =====
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() in ("true", "1")

# ===== 数据路径 =====
DATA_DIR = Path(__file__).parent.parent / "data"
SYNTH_DIR = DATA_DIR / "synth"
EXAMPLES_DIR = DATA_DIR / "examples"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# ===== 算法参数 =====
DEFAULT_CONTAMINATION = 0.02
DEFAULT_N_ESTIMATORS = 100
DEFAULT_TEST_SIZE = 0.3

# ===== 验证参数 =====
VALIDATION_CONFIG_DIR = DATA_DIR / "configs" / "validation"
VALIDATION_TIMEOUT_SEC = 60

def get_llm_client():
    """获取 LLM 客户端（真实或 Mock）"""
    if USE_MOCK_LLM:
        from factory.llm.mock_client import MockClient
        return MockClient()

    from factory.llm.openai_client import OpenAIClient
    return OpenAIClient()

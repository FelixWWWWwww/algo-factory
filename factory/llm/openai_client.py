# factory/llm/openai_client.py
import os
from importlib import import_module
from typing import Any, Optional, Dict, List
from .client import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI 兼容客户端。

    支持通过显式参数或环境变量配置 API key、base_url 和 model。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 OpenAI 客户端。

        api_key 优先级：显式参数 -> DEEPSEEK_API_KEY -> OPENAI_API_KEY
        base_url 优先级：显式参数 -> OPENAI_BASE_URL -> LLM_BASE_URL
        model 优先级：显式参数 -> LLM_MODEL -> OPENAI_MODEL -> 默认值
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未找到 LLM API Key，请设置 DEEPSEEK_API_KEY / OPENAI_API_KEY，或显式传入 api_key"
            )

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        if not self.base_url:
            raise ValueError(
                "未指定 base_url，请传入 base_url 或设置 OPENAI_BASE_URL / LLM_BASE_URL"
            )

        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-v4-pro"
        openai_module = import_module("openai")
        self.client = openai_module.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(
            self,
            messages: List[Dict[str, str]],
            schema: Optional[type] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            **kwargs
    ) -> Dict[str, Any]:
        """
        调用 OpenAI API

        Args:
            messages: 对话消息列表
            schema: Pydantic 模型（暂时忽略，后续在 structured.py 处理）
            temperature: 创意度（0-2，越低越严谨）
            max_tokens: 最大输出 token 数
            **kwargs: 其他参数

        Returns:
            {'message': 'LLM 返回的文本', 'usage': {'prompt_tokens': xxx, 'completion_tokens': xxx}}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        usage = getattr(response, "usage", None)
        return {
            "message": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0)
            }
        }

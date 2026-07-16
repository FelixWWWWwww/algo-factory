# factory/llm/openai_client.py
from importlib import import_module
from typing import Any, Optional, Dict, List
from .client import LLMClient


class OpenAIClient(LLMClient):
    """真实 OpenAI API 客户端

    支持两种使用方式：
    1. OpenAI 官方 API
    2. DeepSeek 兼容 OpenAI SDK 的 API
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: str = "gpt-3.5-turbo"
    ):
        """
        初始化 OpenAI 客户端

        Args:
            api_key: API Key（如果为 None，会从环境变量 OPENAI_API_KEY 读取）
            base_url: API 基础 URL（默认是 OpenAI 官方，或用 DeepSeek 等兼容服务）
            model: 模型名称（默认 gpt-3.5-turbo）

        Example:
            >>> # 使用 OpenAI 官方
            >>> client = OpenAIClient(api_key="sk-xxx")

            >>> # 使用 DeepSeek（兼容 OpenAI SDK）
            >>> client = OpenAIClient(
            ...     api_key="sk-xxx",
            ...     base_url="https://api.deepseek.com",
            ...     model="deepseek-chat"
            ... )
        """
        self.model = model
        openai_module = import_module("openai")
        self.client = openai_module.OpenAI(
            api_key=api_key,
            base_url=base_url
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # 提取响应
            message = response.choices[0].message.content

            return {
                "message": message,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }

        except Exception as e:
            # 错误处理
            return {
                "error": str(e),
                "message": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

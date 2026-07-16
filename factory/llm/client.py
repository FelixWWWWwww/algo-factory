# factory/llm/client.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class LLMClient(ABC):
    """LLM 客户端抽象基类

    所有 LLM 实现（真实 API、Mock）都要继承这个类
    并实现 chat() 方法，确保接口一致。
    """

    @abstractmethod
    def chat(
            self,
            messages: List[Dict[str, str]],
            schema: Optional[type] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        调用 LLM 进行对话

        Args:
            messages: 对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
            schema: Pydantic 模型（可选），用于结构化输出
            **kwargs: 其他参数（如 temperature, max_tokens 等）

        Returns:
            返回 LLM 的响应，格式为字典

        Example:
            >>> client = OpenAIClient()
            >>> response = client.chat([
            ...     {"role": "user", "content": "你好"}
            ... ])
            >>> print(response)
            {'message': 'Hello!', 'tokens_used': 10}
        """
        pass

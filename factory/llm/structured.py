# factory/llm/structured.py
import re
import json
from typing import Any, Dict, Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError
import logging

try:
    import json_repair
except Exception:  # pragma: no cover
    json_repair = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class StructuredOutput:
    """结构化输出处理器

    功能：
    1. 调用 LLM
    2. 修复 JSON 格式错误（json_repair）
    3. 用 Pydantic 验证数据
    4. 自动重试（最多 3 次）
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient 实例（OpenAI 或 Mock）
        """
        self.llm_client = llm_client

    @retry(
        stop=stop_after_attempt(3),  # 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=2, max=10)  # 指数退避：2s -> 4s -> 8s
    )
    def call_structured(
            self,
            prompt: str,
            schema: Type[T],
            model: Optional[str] = None
    ) -> T:
        """
        调用 LLM 获得结构化输出

        Args:
            prompt: 提示词
            schema: Pydantic 模型（定义输出格式）
            model: 指定模型（可选）

        Returns:
            验证后的 Pydantic 模型实例

        Raises:
            ValueError: 3 次重试后仍然失败

        Example:
            >>> from pydantic import BaseModel
            >>> class TaskCard(BaseModel):
            ...     task_type: str
            ...     target: str
            ...     metrics: list
            >>>
            >>> output = StructuredOutput(client).call_structured(
            ...     prompt="分析任务：...",
            ...     schema=TaskCard
            ... )
            >>> print(output.task_type)
        """
        # Step 1: 调用 LLM
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\n请直接返回 JSON，不要其他文字。"
            }
        ]

        response = self.llm_client.chat(messages)

        if "error" in response:
            raise ValueError(f"LLM 调用失败: {response['error']}")

        raw_output = response.get("message", "")

        # Step 2: 清理 JSON（去掉 ```json ... ``` 包裹）
        cleaned_output = self._clean_json_output(raw_output)

        # Step 3: 用 json_repair 修复常见错误
        try:
            if json_repair is not None:
                data = json_repair.loads(cleaned_output)
            else:
                data = json.loads(cleaned_output)
        except Exception as e:
            logger.error(f"json_repair 失败: {e}")
            # 最后的兜底：尝试原始 json.loads
            data = json.loads(cleaned_output)

        # Step 4: 用 Pydantic 验证
        try:
            validated = schema(**data)
            logger.info(f"结构化输出验证成功: {schema.__name__}")
            return validated
        except ValidationError as e:
            logger.error(f"Pydantic 验证失败: {e}")
            # 重试会自动触发（tenacity）
            raise

    @staticmethod
    def _clean_json_output(output: str) -> str:
        """
        清理 LLM 的输出，去掉 ```json ... ``` 包裹

        Example:
            Input: "```json\n{\"a\": 1}\n```"
            Output: "{\"a\": 1}"
        """
        # 方法 1: 用正则表达式找到 JSON 代码块
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output)
        if match:
            return match.group(1).strip()

        # 方法 2: 如果没有代码块，直接返回（假设已经是 JSON）
        return output.strip()


def call_structured(
        *,
        prompt: str,
        pydantic_model: Type[T],
        use_mock: bool = False,
        model: Optional[str] = None,
        **extra_fields: Any
) -> T:
    """Backward-compatible wrapper used by older call sites.

    The wrapper creates the appropriate client, validates the LLM output with
    ``StructuredOutput``, then injects any locally-known fields such as
    ``document_path`` or ``timestamp``.
    """
    if use_mock:
        from .mock_client import MockClient
        llm_client = MockClient()
    else:
        from .openai_client import OpenAIClient
        llm_client = OpenAIClient(model=model)

    structured = StructuredOutput(llm_client)
    result = structured.call_structured(prompt=prompt, schema=pydantic_model, model=model)

    if not extra_fields:
        return result

    if hasattr(result, "model_copy"):
        return result.model_copy(update=extra_fields)

    data = result.dict()
    data.update(extra_fields)
    return pydantic_model(**data)

from typing import Any, List, Optional


class _Message:
    content: Optional[str]


class _Choice:
    message: _Message


class _Usage:
    prompt_tokens: int
    completion_tokens: int


class _ChatCompletions:
    def create(
        self,
        *,
        model: str,
        messages: List[dict[str, str]],
        temperature: float = ...,
        max_tokens: int = ...,
        **kwargs: Any,
    ) -> Any: ...


class _Chat:
    completions: _ChatCompletions


class OpenAI:
    chat: _Chat

    def __init__(
        self,
        *,
        api_key: Optional[str] = ...,
        base_url: Optional[str] = ...,
        **kwargs: Any,
    ) -> None: ...

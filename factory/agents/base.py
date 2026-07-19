"""
factory/agents/base.py
所有 Agent 的抽象基类
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from factory.state import TaskState
from factory.llm.client import LLMClient


class Agent(ABC):
    """
    所有 Agent 的抽象基类。

    使用模式：
    ```python
    class MyAgent(Agent):
        def _run(self, state: TaskState) -> TaskState:
            # 实现你的逻辑
            state.some_field = "new_value"
            return state

    agent = MyAgent(name="my_agent", llm_client=llm_client)
    state = agent.run(state)
    ```
    """

    def __init__(self, name: str, llm_client: LLMClient = None):
        self.name = name
        self.llm_client = llm_client
        self.start_time = None
        self.end_time = None

    def run(self, state: TaskState) -> TaskState:
        """
        公共运行接口。
        包含：日志记录 → 调用 _run() → 异常捕获 → 耗时统计
        """
        self.start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{self.name}] 开始执行")
        print(f"{'=' * 60}")

        try:
            # 调用子类实现的具体逻辑
            state = self._run(state)

            self.end_time = time.time()
            duration = self.end_time - self.start_time

            print(f"\n[{self.name}] 完成（耗时 {duration:.2f}s）")
            print(f"   状态更新字段：{self._get_changed_fields(state)}")

            # 记录到错误历史（成功）
            state.error_history.append({
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "duration": duration
            })

            return state

        except Exception as e:
            self.end_time = time.time()
            duration = self.end_time - self.start_time

            print(f"\n[{self.name}] 失败：{type(e).__name__}: {e}")

            # 记录到错误历史
            state.error_history.append({
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
                "duration": duration
            })

            # Mock 模式下，不崩溃，返回原状态
            if getattr(state, '_use_mock', True):
                print(f"   [Mock 模式] 继续执行")
                return state
            else:
                raise

    @abstractmethod
    def _run(self, state: TaskState) -> TaskState:
        """
        子类必须实现的具体逻辑。
        """
        pass

    def _get_changed_fields(self, state: TaskState) -> list:
        """
        简单的"变更追踪"，用于打印日志。
        实际产品可用 __dict__ diff。
        """
        return ["task_card", "retrieved_context", "plans", "code", "metrics", "validation_results"]

"""
Interpreter Agent：自然语言需求 → 结构化 TaskCard
"""

from factory.agents.base import Agent
from factory.state import TaskState


class InterpreterAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Interpreter", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        """
        Mock 模式：直接返回预定义的 TaskCard
        真实模式（Day 2）：用 LLM 抽取结构
        """
        if state._use_mock:
            # Mock 数据
            state.task_card = {
                "task_type": "anomaly_detection",
                "target": "industrial_sensors",
                "constraints": [
                    "extreme_imbalance (< 3% anomalies)",
                    "high_dimensional (> 100 features)"
                ],
                "metrics": ["pr_auc", "f1", "precision", "recall"],
                "data_hint": "time_series_sensor_data"
            }
            return state

        # Day 2+ 真实实现在这里
        raise NotImplementedError("真实 Interpreter 在 Day 2 实现")

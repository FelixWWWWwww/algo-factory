"""
Validator Agent：执行代码 + 评估
"""

from factory.agents.base import Agent
from factory.state import TaskState


class ValidatorAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Validator", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        """
        Mock 模式：为每个方案返回假的评估结果
        真实模式（Day 3）：在沙箱中执行代码，收集真实指标
        """
        if state._use_mock:
            # Mock 评估结果
            mock_results = [
                {
                    "plan_id": "plan_1",
                    "algorithm": "IForest",
                    "pr_auc": 0.84,
                    "f1": 0.72,
                    "precision": 0.75,
                    "recall": 0.70,
                    "execution_time": 0.35,
                    "status": "success"
                },
                {
                    "plan_id": "plan_2",
                    "algorithm": "LOF",
                    "pr_auc": 0.81,
                    "f1": 0.68,
                    "precision": 0.70,
                    "recall": 0.67,
                    "execution_time": 0.52,
                    "status": "success"
                },
                {
                    "plan_id": "plan_3",
                    "algorithm": "OCSVM",
                    "pr_auc": 0.83,
                    "f1": 0.71,
                    "precision": 0.73,
                    "recall": 0.69,
                    "execution_time": 0.48,
                    "status": "success"
                }
            ]

            state.validation_results = mock_results

            # 自动选优（按 PR-AUC）
            best = max(mock_results, key=lambda x: x["pr_auc"])
            state.best_model = best["algorithm"]
            state.metrics = {
                k: v for k, v in best.items()
                if k not in ["plan_id", "algorithm", "status"]
            }

            return state

        raise NotImplementedError("真实 Validator 在 Day 3 实现")

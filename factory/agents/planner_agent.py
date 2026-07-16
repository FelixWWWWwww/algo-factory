"""
Planner Agent：TaskCard + 检索上下文 → 多个候选方案
"""

from factory.agents.base import Agent
from factory.state import TaskState


class PlannerAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Planner", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        """
        Mock 模式：返回 3 个预定义的方案
        """
        if state._use_mock:
            state.plans = [
                {
                    "id": "plan_1",
                    "name": "IForest + StandardScaler",
                    "algorithm": "IForest",
                    "contamination": 0.02,
                    "pipeline_steps": [
                        "load_data",
                        "eda",
                        "preprocess_with_standard_scaler",
                        "train_iforest",
                        "evaluate"
                    ],
                    "rationale": "IForest 对高维不平衡数据鲁棒，无需复杂调参"
                },
                {
                    "id": "plan_2",
                    "name": "LOF + novelty=True",
                    "algorithm": "LOF",
                    "contamination": 0.02,
                    "pipeline_steps": [
                        "load_data",
                        "eda",
                        "preprocess_with_robust_scaler",
                        "train_lof_novelty",
                        "evaluate"
                    ],
                    "rationale": "LOF novelty 模式对新样本有更好的判别能力"
                },
                {
                    "id": "plan_3",
                    "name": "One-Class SVM + RBF",
                    "algorithm": "OCSVM",
                    "contamination": 0.02,
                    "pipeline_steps": [
                        "load_data",
                        "eda",
                        "preprocess_with_standard_scaler",
                        "train_ocsvm",
                        "evaluate"
                    ],
                    "rationale": "OCSVM RBF 核对离群点敏感度最高"
                }
            ]
            return state

        raise NotImplementedError("真实 Planner 在 Day 2 实现")

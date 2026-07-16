"""
Retriever Agent：TaskCard → 知识图谱检索 → 上下文
"""

from factory.agents.base import Agent
from factory.state import TaskState


class RetrieverAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Retriever", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        """
        Mock 模式：返回预定义的检索结果
        真实模式（Day 2）：从 GraphStore 检索
        """
        if state._use_mock:
            state.retrieved_context = {
                "similar_capabilities": [
                    {
                        "name": "IForest on sensors",
                        "success_rate": 0.0,  # Day 1 初值都是 0
                        "pr_auc": 0.84
                    }
                ],
                "applicable_algorithms": ["IForest", "LOF", "OCSVM"],
                "lessons": [
                    {
                        "title": "contamination 参数设置",
                        "content": "推荐值 0.01-0.05，与真实异常占比一致"
                    },
                    {
                        "title": "高维空间下 LOF 退化",
                        "content": "维度 > 100 时应使用 novelty=True 或改用 IForest"
                    }
                ],
                "failure_cases": [
                    {
                        "title": "用 Accuracy 选模导致废模型",
                        "fix_suggestion": "使用 PR-AUC 作为主指标"
                    }
                ]
            }
            return state

        raise NotImplementedError("真实 Retriever 在 Day 2 实现")

"""
T2.4 Retriever Agent：TaskCard → 知识图谱检索 → RetrievedContext

优先从注入的 GraphStore 按 task_type 检索；无图谱/命中为空时回退内置上下文。
关键：始终纳入 FailureCase（accuracy 选模废模型），供 Planner 规避。
"""

from factory.agents.base import Agent
from factory.state import TaskState, RetrievedContext

_APPLICABLE = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
_LESSONS = [
    "contamination 建议 0.01–0.05，应与真实异常占比一致",
    "高维(>100维)时 LOF 易退化，改用 novelty=True 或 IsolationForest",
    "距离/密度类(LOF/OCSVM)上线前必须 StandardScaler",
    "极不平衡场景禁用 accuracy 选优，主指标用 PR-AUC / F1(anomaly)",
]
_FAILURES = [
    {
        "title": "用 accuracy 选模导致废模型",
        "root_cause": "正常样本占比 >95%，accuracy 被多数类主导虚高",
        "fix_suggestion": "改用 PR-AUC 作为主指标",
    }
]


class RetrieverAgent(Agent):
    def __init__(self, llm_client=None, graph_store=None):
        super().__init__(name="Retriever", llm_client=llm_client)
        self.graph_store = graph_store

    def _run(self, state: TaskState) -> TaskState:
        task_type = getattr(state.task_card, "task_type", "anomaly_detection")
        state.retrieved_context = self._from_graph(task_type) or self._fallback()
        return state

    def _from_graph(self, task_type: str):
        if self.graph_store is None:
            return None
        try:
            hits = self.graph_store.query_by_task_type(task_type)
            if not hits:
                return None
            caps = [{"id": k, **(v if isinstance(v, dict) else {})} for k, v in hits.items()]
            return RetrievedContext(
                similar_capabilities=caps,
                failure_cases=_FAILURES,
                lessons=_LESSONS,
            )
        except Exception:
            return None

    def _fallback(self) -> RetrievedContext:
        return RetrievedContext(
            similar_capabilities=[{
                "name": "IForest on industrial sensors",
                "applicable_algorithms": _APPLICABLE,
                "success_rate": 0.0,
            }],
            failure_cases=_FAILURES,
            lessons=_LESSONS,
        )

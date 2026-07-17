"""
T2.4 Retriever Agent：TaskCard → 知识图谱检索 → RetrievedContext

真实检索：从图谱读取「该 task_type 下失败过的算法」与 Lesson，
使 Planner 能据此规避历史失败（闭环学习的读取侧）。图空则回退内置先验。
"""
from factory.agents.base import Agent
from factory.state import TaskState, RetrievedContext

_APPLICABLE = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
_LESSONS = [
    "contamination 建议 0.01–0.05，应与真实异常占比一致",
    "距离/密度类(LOF/OCSVM)上线前必须 StandardScaler",
    "极不平衡场景禁用 accuracy 选优，主指标用 PR-AUC / F1(anomaly)",
]
_FAILURES = [
    {"title": "用 accuracy 选模导致废模型",
     "root_cause": "正常样本占比 >95%，accuracy 被多数类主导虚高",
     "fix_suggestion": "改用 PR-AUC 作为主指标"},
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
            failed = {}          # algorithm -> reason
            lessons = []
            for _, attrs in self.graph_store.graph.nodes(data=True):
                if attrs.get("task_type") != task_type:
                    continue
                if attrs.get("type") == "FailureCase" and attrs.get("algorithm"):
                    failed[attrs["algorithm"]] = attrs.get("reason", "")
                elif attrs.get("type") == "Lesson" and attrs.get("content"):
                    lessons.append(attrs["content"])
            if not failed and not lessons:
                return None       # 图里还没有可用经验 → 用内置先验兜底
            failure_cases = [
                {"algorithm": a, "reason": r,
                 "fix_suggestion": "该算法在本场景历史失败，建议规避或降级"}
                for a, r in failed.items()
            ]
            return RetrievedContext(
                similar_capabilities=[{"applicable_algorithms": _APPLICABLE}],
                failure_cases=failure_cases or _FAILURES,
                lessons=(lessons + _LESSONS),
            )
        except Exception:
            return None

    def _fallback(self) -> RetrievedContext:
        return RetrievedContext(
            similar_capabilities=[{"applicable_algorithms": _APPLICABLE, "success_rate": 0.0}],
            failure_cases=_FAILURES,
            lessons=_LESSONS,
        )

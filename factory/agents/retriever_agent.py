"""
T2.4 Retriever Agent：TaskCard + 数据画像 → 知识图谱检索 → RetrievedContext

画像感知的失败复用（升级版）：
  某算法在图谱里失败过，只有当【当前数据画像】与【当时失败的画像】足够相似时，
  才把它当作"需规避"上报给 Planner。这样 LOF 只会在"像致密簇"的数据上被降级，
  换个数据集（画像不同）照样有机会当冠军——避免"一朝被蛇咬，十年怕井绳"。
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

SIM_THRESHOLD = 0.70   # 画像相似度阈值：≥ 该值才认为"同类数据"，才复用该失败经验


def _close(x, y, scale):
    """两个数值的归一化接近度 [0,1]；差得越远越接近 0。"""
    try:
        return max(0.0, 1.0 - abs(float(x) - float(y)) / (scale + 1e-9))
    except Exception:
        return 0.5


def profile_similarity(a: dict, b: dict) -> float:
    """两个数据画像的相似度 [0,1]。异常紧致度权重加倍（它是算法成败的关键判别特征）。"""
    nf_scale = max(float(a.get("n_features") or 1), float(b.get("n_features") or 1), 1)
    parts = [
        (1.0, _close(a.get("n_features", 0), b.get("n_features", 0), nf_scale)),
        (1.0, _close(a.get("anomaly_ratio", 0) or 0, b.get("anomaly_ratio", 0) or 0, 0.08)),
        (1.0, 1.0 if bool(a.get("scale_disparity")) == bool(b.get("scale_disparity")) else 0.0),
        (2.0, _close(a.get("anomaly_compactness", 1.0) if a.get("anomaly_compactness") is not None else 1.0,
                     b.get("anomaly_compactness", 1.0) if b.get("anomaly_compactness") is not None else 1.0, 0.8)),
        (1.0, 1.0 if bool(a.get("has_time_column")) == bool(b.get("has_time_column")) else 0.0),
    ]
    total_w = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total_w


def _applies(cur_profile: dict, failed_sig: dict) -> bool:
    """该失败经验是否适用于当前数据。无画像（旧数据/缺失）→ 向后兼容视为适用。"""
    if not failed_sig or not cur_profile:
        return True
    return profile_similarity(cur_profile, failed_sig) >= SIM_THRESHOLD


class RetrieverAgent(Agent):
    def __init__(self, llm_client=None, graph_store=None):
        super().__init__(name="Retriever", llm_client=llm_client)
        self.graph_store = graph_store

    def _run(self, state: TaskState) -> TaskState:
        task_type = getattr(state.task_card, "task_type", "anomaly_detection")
        ctx = self._from_graph(task_type, state.data_profile or {})
        state.retrieved_context = ctx or self._fallback()
        return state

    def _from_graph(self, task_type: str, cur_profile: dict):
        if self.graph_store is None:
            return None
        try:
            failed = {}   # algorithm -> reason（仅当画像相似）
            for _, attrs in self.graph_store.graph.nodes(data=True):
                if attrs.get("task_type") != task_type:
                    continue
                if attrs.get("type") == "FailureCase" and attrs.get("algorithm"):
                    if _applies(cur_profile, attrs.get("profile") or {}):
                        failed[attrs["algorithm"]] = attrs.get("reason", "")
            if not failed:
                return None    # 没有"同类数据"上的失败 → 交给内置先验兜底
            # 只取相似失败算法对应的教训
            lessons = []
            for _, attrs in self.graph_store.graph.nodes(data=True):
                if (attrs.get("task_type") == task_type and attrs.get("type") == "Lesson"
                        and attrs.get("algorithm") in failed and attrs.get("content")):
                    lessons.append(attrs["content"])
            failure_cases = [
                {"algorithm": a, "reason": r,
                 "fix_suggestion": "该算法在相似数据画像上历史失败，建议规避或降级"}
                for a, r in failed.items()
            ]
            return RetrievedContext(
                similar_capabilities=[{"applicable_algorithms": _APPLICABLE}],
                failure_cases=failure_cases,
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

"""验证画像感知学习：失败经验只在"数据画像相似"时才复用，避免误伤好算法。"""
from factory.state import TaskState, TaskCard
from factory.agents import RetrieverAgent
from factory.agents.retriever_agent import profile_similarity
from factory.graph.store import GraphStore


def _graph_with_lof_failure(compactness):
    gs = GraphStore()
    gs.add_node("failurecase:t:v2", type="FailureCase", task_type="anomaly_detection",
                algorithm="LocalOutlierFactor", reason="pr_auc=0.04 < 0.6",
                profile={"n_features": 6, "anomaly_ratio": 0.05, "scale_disparity": False,
                         "anomaly_compactness": compactness, "has_time_column": False})
    return gs


def _avoided(state, gs):
    s = RetrieverAgent(None, gs)._run(state)
    return {fc.get("algorithm") for fc in s.retrieved_context.failure_cases}


def test_similarity_distinguishes_compact_vs_spread():
    compact = {"n_features": 6, "anomaly_ratio": 0.05, "scale_disparity": False, "anomaly_compactness": 0.11}
    spread = {"n_features": 6, "anomaly_ratio": 0.03, "scale_disparity": False, "anomaly_compactness": 3.5}
    assert profile_similarity(compact, {**compact, "anomaly_compactness": 0.12}) >= 0.7
    assert profile_similarity(compact, spread) < 0.7


def test_lof_avoided_on_similar_compact_data():
    gs = _graph_with_lof_failure(0.11)
    s = TaskState(user_query="x"); s.task_card = TaskCard(task_type="anomaly_detection")
    s.data_profile = {"n_features": 6, "anomaly_ratio": 0.05, "scale_disparity": False,
                      "anomaly_compactness": 0.12, "has_time_column": False}
    assert "LocalOutlierFactor" in _avoided(s, gs)          # 相似 → 规避


def test_lof_gets_fresh_chance_on_different_data():
    gs = _graph_with_lof_failure(0.11)
    s = TaskState(user_query="x"); s.task_card = TaskCard(task_type="anomaly_detection")
    s.data_profile = {"n_features": 6, "anomaly_ratio": 0.03, "scale_disparity": False,
                      "anomaly_compactness": 3.5, "has_time_column": False}
    assert "LocalOutlierFactor" not in _avoided(s, gs)      # 画像不同 → 不误伤

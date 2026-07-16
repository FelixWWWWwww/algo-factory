"""
T2.4–T2.6 集成测试：
  T2.4  Interpreter / Retriever / Planner Agent 产出结构化结果
  T2.5  Coder Agent 生成可运行代码（并实际执行验证）
  T2.6  train + evaluate 节点跑出真实 PR-AUC（IForest/LOF/OCSVM 三算法）
       + threshold 与 y_pred 一致性（回归此前失败的 bug）
"""

import numpy as np
import pytest

from factory.llm import MockClient
from factory.state import TaskState, TaskCard, Plan, RetrievedContext, CodeVersion
from factory.agents import InterpreterAgent, RetrieverAgent, PlannerAgent, CoderAgent
from factory.nodes import (
    data_ingestion_node, eda_node, preprocess_node,
    split_node, train_node, evaluate_node, make_synthetic_dataset,
)

ALGOS = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]


# ────────────────────────── T2.4 ──────────────────────────

def test_interpreter_produces_taskcard():
    llm = MockClient()
    state = TaskState(user_query="对工业传感器数据进行异常检测")
    state = InterpreterAgent(llm).run(state)
    assert isinstance(state.task_card, TaskCard)
    assert state.task_card.task_type == "anomaly_detection"
    assert "pr_auc" in state.task_card.metrics
    assert 0 < state.contamination <= 0.5


def test_retriever_includes_failure_case():
    state = TaskState(user_query="异常检测")
    state = InterpreterAgent(MockClient()).run(state)
    state = RetrieverAgent(MockClient()).run(state)
    assert isinstance(state.retrieved_context, RetrievedContext)
    titles = [fc.get("title", "") for fc in state.retrieved_context.failure_cases]
    assert any("accuracy" in t for t in titles), "Retriever 必须纳入 accuracy 废模型的 FailureCase"


def test_planner_produces_three_plans():
    state = TaskState(user_query="异常检测")
    state = InterpreterAgent(MockClient()).run(state)
    state = PlannerAgent(MockClient()).run(state)
    assert len(state.plans) == 3
    algos = {p.algorithm for p in state.plans}
    assert algos == set(ALGOS)
    for p in state.plans:
        assert isinstance(p, Plan) and p.rationale  # 每个方案都有自然语言理由


# ────────────────────────── T2.5 ──────────────────────────

def test_coder_generates_runnable_code(tmp_path):
    state = TaskState(user_query="异常检测")
    for agent in (InterpreterAgent(MockClient()), PlannerAgent(MockClient())):
        state = agent.run(state)
    state = CoderAgent(MockClient(), out_dir=str(tmp_path / "examples")).run(state)

    assert len(state.code_versions) == 3
    for cv in state.code_versions:
        assert isinstance(cv, CodeVersion)
        assert "def run(" in cv.code and "RESULT_JSON" in cv.code
        assert "== -1" in cv.code  # 标签映射存在


def test_generated_code_actually_executes(tmp_path):
    """把生成的代码 exec 出来，在合成数据上真跑，验证 IPC 协议 + 指标。"""
    csv = tmp_path / "syn.csv"
    make_synthetic_dataset(n_samples=1500, n_features=6, anomaly_ratio=0.03,
                           random_state=7, save_path=str(csv))

    state = TaskState(user_query="异常检测")
    for agent in (InterpreterAgent(MockClient()), PlannerAgent(MockClient())):
        state = agent.run(state)
    state = CoderAgent(MockClient(), out_dir=str(tmp_path / "examples")).run(state)

    for cv in state.code_versions:
        ns = {}
        exec(compile(cv.code, cv.plan_name, "exec"), ns)
        result = ns["run"](str(csv))
        assert "pr_auc" in result, f"{cv.plan_name} 未产出 pr_auc"
        assert "accuracy" not in result, "不得输出 accuracy"
        assert 0.0 <= result["pr_auc"] <= 1.0


# ────────────────────────── T2.6 ──────────────────────────

def _run_node_pipeline(csv, algorithm):
    state = TaskState(user_query="异常检测")
    state = data_ingestion_node(state, str(csv))
    state = eda_node(state, MockClient())
    state = preprocess_node(state, algorithm=algorithm)
    state = split_node(state)
    state = train_node(state, algorithm=algorithm)
    state = evaluate_node(state)
    return state


@pytest.mark.parametrize("algorithm", ALGOS)
def test_node_pipeline_real_pr_auc(tmp_path, algorithm):
    csv = tmp_path / "syn.csv"
    make_synthetic_dataset(n_samples=1500, n_features=6, anomaly_ratio=0.03,
                           random_state=11, save_path=str(csv))
    state = _run_node_pipeline(csv, algorithm)
    assert "pr_auc" in state.final_metrics, f"{algorithm} 未产出 pr_auc"
    assert state.final_metrics["pr_auc"] is not None
    # 合成数据高度可分，主算法 PR-AUC 应显著高于随机
    assert state.final_metrics["pr_auc"] > 0.5, (
        f"{algorithm} PR-AUC={state.final_metrics['pr_auc']} 偏低"
    )


def test_threshold_consistent_with_ypred(tmp_path):
    """回归：scores > threshold 必须与 y_pred 一致（此前 -offset_ 的 bug）。"""
    csv = tmp_path / "syn.csv"
    make_synthetic_dataset(n_samples=1500, n_features=6, anomaly_ratio=0.03,
                           random_state=3, save_path=str(csv))
    state = _run_node_pipeline(csv, "IsolationForest")
    scores = np.array(state.anomaly_scores)
    y_pred = np.array(state.y_pred)
    above = (scores > state.threshold).astype(int)
    agreement = (above == y_pred).mean()
    assert agreement > 0.99, f"threshold 与 y_pred 一致率仅 {agreement:.2%}"


def test_eval_never_uses_accuracy_for_selection(tmp_path):
    csv = tmp_path / "syn.csv"
    make_synthetic_dataset(n_samples=1500, n_features=6, anomaly_ratio=0.03,
                           random_state=5, save_path=str(csv))
    state = _run_node_pipeline(csv, "IsolationForest")
    # final_metrics（选优用）不得含 accuracy
    assert not any("accuracy" in k for k in state.final_metrics)

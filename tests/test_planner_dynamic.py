"""验证 Planner 动态选型：mock 用内置默认；real（LLM）完全由模型决定算法（废除固定池）。"""
import json
from factory.state import TaskState, TaskCard
from factory.agents import PlannerAgent
from factory.llm import MockClient


class _StubLLM:
    """假装 real LLM：按 planner_prompt 返回一个方案数组（含非预设算法）。"""
    def chat(self, messages, **kw):
        plans = [
            {"name": "ECOD 方案", "algorithm": "ECOD", "import_path": "pyod.models.ecod.ECOD",
             "preprocessing": ["StandardScaler"], "rationale": "无参数、稳健",
             "fit_reason": "高维不平衡", "expected_metric": 0.8},
            {"name": "EllipticEnvelope 方案", "algorithm": "EllipticEnvelope",
             "preprocessing": ["StandardScaler"], "rationale": "高斯假设",
             "fit_reason": "低维数值特征", "expected_metric": 0.7},
        ]
        return {"message": json.dumps(plans, ensure_ascii=False)}


def test_planner_mock_uses_defaults():
    s = TaskState(user_query="异常检测")                 # _use_mock 默认 True
    s.task_card = TaskCard(task_type="anomaly_detection")
    s = PlannerAgent(MockClient())._run(s)
    assert {p.algorithm for p in s.plans} == {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"}


def test_planner_real_uses_llm_selection():
    s = TaskState(user_query="检测信用卡欺诈")
    s._use_mock = False                                  # 走 real 分支
    s.task_card = TaskCard(task_type="anomaly_detection")
    s.data_profile = {"n_samples": 1500, "n_features": 30, "has_labels": True,
                      "anomaly_ratio": 0.02, "scale_disparity": True}
    s = PlannerAgent(_StubLLM())._run(s)
    # 算法完全由 LLM 决定，不再是固定三件套
    assert [p.algorithm for p in s.plans] == ["ECOD", "EllipticEnvelope"], [p.algorithm for p in s.plans]

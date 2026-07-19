"""验证 Coder：mock 用模板；real（LLM）从零现写、支持非预设算法，且不回退成 IForest 模板。"""
from factory.state import TaskState, TaskCard, Plan
from factory.agents import CoderAgent
from factory.llm import MockClient

_ELLIPTIC_CODE = '''import json, numpy as np, pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
def run(data_path):
    df = pd.read_csv(data_path)
    y = df.pop("label").values if "label" in df.columns else None
    X = df.values.astype(float)
    m = EllipticEnvelope(contamination=0.05, random_state=42).fit(X)
    scores = -m.decision_function(X); y_pred = (m.predict(X) == -1).astype(int)
    r = {"n_anomalies_detected": int(y_pred.sum())}
    if y is not None:
        r.update({"pr_auc": round(float(average_precision_score(y, scores)), 4),
                  "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4)})
    print("RESULT_JSON:" + json.dumps(r)); return r
'''


class _StubLLM:
    def chat(self, messages, **kw):
        return {"message": _ELLIPTIC_CODE}


def test_coder_mock_uses_template():
    s = TaskState(user_query="x"); s.task_card = TaskCard()
    s.plans = [Plan(name="LOF 方案", algorithm="LocalOutlierFactor", contamination=0.05)]
    s = CoderAgent(MockClient())._run(s)                 # mock → 模板
    assert "LocalOutlierFactor" in s.code_versions[0].code


def test_coder_real_generates_from_llm():
    s = TaskState(user_query="检测欺诈"); s._use_mock = False; s.task_card = TaskCard()
    s.plans = [Plan(name="EllipticEnvelope 方案", algorithm="EllipticEnvelope", contamination=0.05)]
    s = CoderAgent(_StubLLM())._run(s)
    code = s.code_versions[0].code
    assert "EllipticEnvelope" in code and "def run(" in code   # 用了 LLM 现写的新算法代码
    assert "IsolationForest" not in code                       # 不是回退的 IForest 模板

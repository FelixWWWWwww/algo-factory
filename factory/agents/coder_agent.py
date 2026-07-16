"""
T2.5 Coder Agent：Plan → 可运行 Python 代码 + IPC 协议

代码生成协议约定：
  - 函数签名  def run(data_path: str) -> dict
  - 末尾打印  print("RESULT_JSON:" + json.dumps(metrics))
  - sklearn 标签 -1/1 必须映射为 1/0
优先采用 LLM 产出（若合法），否则用内置正确模板；代码落盘 data/examples/{task_id}/。
"""

import os
import ast
from factory.agents.base import Agent
from factory.state import TaskState, CodeVersion

_MODEL_SNIPPET = {
    "IsolationForest": (
        "from sklearn.ensemble import IsolationForest",
        "IsolationForest(contamination=CONTAM, random_state=42)",
        False,  # 是否需要标准化
    ),
    "LocalOutlierFactor": (
        "from sklearn.neighbors import LocalOutlierFactor",
        "LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=CONTAM)",
        True,
    ),
    "OneClassSVM": (
        "from sklearn.svm import OneClassSVM",
        "OneClassSVM(kernel='rbf', gamma='scale', nu=CONTAM)",
        True,
    ),
}


class CoderAgent(Agent):
    def __init__(self, llm_client=None, out_dir="data/examples"):
        super().__init__(name="Coder", llm_client=llm_client)
        self.out_dir = out_dir

    def _run(self, state: TaskState) -> TaskState:
        versions = []
        for i, plan in enumerate(state.plans, 1):
            algo = getattr(plan, "algorithm", "IsolationForest")
            contamination = getattr(plan, "contamination", 0.05) or 0.05
            code = self._llm_code(algo) or self._template(algo, contamination)
            versions.append(CodeVersion(version=f"v{i}", code=code, plan_name=plan.name))
            self._dump(state.task_id, f"v{i}_{algo}.py", code)
        state.code_versions = versions
        if versions:
            state.final_code = versions[0].code
        return state

    # 采用 LLM 代码，仅当它语法合法且符合 IPC 协议
    def _llm_code(self, algorithm: str):
        if self.llm_client is None:
            return None
        try:
            resp = self.llm_client.chat(
                [{"role": "user", "content": f"生成 {algorithm} 异常检测代码(代码)，"
                                             "含 def run(data_path)->dict 与 RESULT_JSON 输出。"}]
            )
            code = resp.get("message", "")
            if (code and "def run(" in code and "RESULT_JSON" in code
                    and algorithm in code and _is_valid(code)):
                return code
        except Exception:
            pass
        return None

    def _template(self, algorithm: str, contamination: float) -> str:
        imp, ctor, needs_scale = _MODEL_SNIPPET.get(
            algorithm, _MODEL_SNIPPET["IsolationForest"]
        )
        ctor = ctor.replace("CONTAM", repr(float(contamination)))
        scale_line = (
            "    X_score = StandardScaler().fit_transform(X)\n"
            if needs_scale else "    X_score = X\n"
        )
        return f'''import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
{imp}


def run(data_path: str) -> dict:
    """生成代码：{algorithm} 异常检测。若含 label 列(1=异常/0=正常)则计算监督指标。"""
    df = pd.read_csv(data_path)
    y_true = df.pop("label").values if "label" in df.columns else None
    X = df.values.astype(float)
{scale_line}
    model = {ctor}
    model.fit(X_score)
    scores = -model.decision_function(X_score)          # 越大越可疑
    y_pred = (model.predict(X_score) == -1).astype(int)  # -1→1(异常), 1→0(正常)

    result = {{"n_anomalies_detected": int(y_pred.sum())}}
    if y_true is not None and len(np.unique(y_true)) > 1:
        result.update({{
            "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        }})
    else:
        result["note"] = "无标签，仅输出检出数供 Top-K 人工审阅"

    print("RESULT_JSON:" + json.dumps(result))
    return result


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
'''

    def _dump(self, task_id: str, fname: str, code: str):
        try:
            d = os.path.join(self.out_dir, task_id)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
                f.write(code)
        except Exception:
            pass


def _is_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

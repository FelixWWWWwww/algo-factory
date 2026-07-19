"""
T2.5 Coder Agent：Plan → 可运行 Python 代码 + IPC 协议

- real 模式：调 LLM（coder_prompt.jinja2）**从零生成**完整脚本，不塞模板。
- mock 模式：无真实生成能力，直接用内置模板产出（模板此时是 LLM 的离线替身）。
- 模板 `_template` 仍保留，但在 real 路径中只作为「验证/修复耗尽 N 次后」的兜底拦截器
  （由 pipeline._validate_with_repair 调用），Coder 本身不再首选模板。

IPC 协议：def run(data_path)->dict + print("RESULT_JSON:"+json.dumps(...)) + -1/1→1/0 映射。
"""
import os
import ast
import json
import re
import logging
from factory.agents.base import Agent
from factory.state import TaskState, CodeVersion
from factory.llm.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)

_MODEL_SNIPPET = {
    "IsolationForest": (
        "from sklearn.ensemble import IsolationForest",
        "IsolationForest(contamination=CONTAM, random_state=42)", False),
    "LocalOutlierFactor": (
        "from sklearn.neighbors import LocalOutlierFactor",
        "LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=CONTAM)", True),
    "OneClassSVM": (
        "from sklearn.svm import OneClassSVM",
        "OneClassSVM(kernel='rbf', gamma='scale', nu=CONTAM)", True),
}


def _strip_fences(text: str) -> str:
    """去掉 LLM 输出里的 ```python ... ``` 包裹。"""
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    return (m.group(1) if m else text).strip()


class CoderAgent(Agent):
    def __init__(self, llm_client=None, out_dir="data/examples"):
        super().__init__(name="Coder", llm_client=llm_client)
        self.out_dir = out_dir
        try:
            self.prompt_mgr = get_prompt_manager()
        except Exception as e:
            logger.warning(f"[coder] prompt_manager 初始化失败: {e}")
            self.prompt_mgr = None

    def _run(self, state: TaskState) -> TaskState:
        use_mock = getattr(state, "_use_mock", True)
        versions = []
        for i, plan in enumerate(state.plans, 1):
            algo = getattr(plan, "algorithm", "IsolationForest")
            contamination = getattr(plan, "contamination", 0.05) or 0.05
            if use_mock:
                code = self._template(algo, contamination)          # 离线：模板即产出
            else:
                code = self._llm_generate(plan, state)              # 真·现写
                if not code:
                    code = "# LLM 未产出有效代码，交由修复循环处理\n"
            versions.append(CodeVersion(version=f"v{i}", code=code, plan_name=plan.name))
            self._dump(state.task_id, f"v{i}_{algo}.py", code)
        state.code_versions = versions
        if versions:
            state.final_code = versions[0].code
        return state

    # ── real：LLM 按方案从零生成 ─────────────────────────────────────
    def _llm_generate(self, plan, state: TaskState):
        if self.llm_client is None or self.prompt_mgr is None:
            return None
        try:
            J = lambda o: json.dumps(o, ensure_ascii=False, indent=2)
            sc = getattr(state.task_card, "suspected_columns", {}) or {}
            label_col = sc.get("label") or "label"
            prompt = self.prompt_mgr.render_template(
                "coder_prompt.jinja2",
                plan_json=J(plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)),
                data_profile_json=J(state.data_profile or {}),
                constraints_json=J(list(getattr(state.task_card, "constraints", []) or [])),
                label_col=label_col,
            )
            resp = self.llm_client.chat([
                {"role": "system", "content": self.prompt_mgr.get_system_prompt()},
                {"role": "user", "content": prompt},
            ])
            return _strip_fences(resp.get("message", "")) or None
        except Exception as e:
            logger.warning(f"[coder] LLM 生成失败: {e}")
            return None

    # ── 模板：mock 产出 + real 路径的最终兜底拦截器 ──────────────────
    def _template(self, algorithm: str, contamination: float) -> str:
        imp, ctor, needs_scale = _MODEL_SNIPPET.get(algorithm, _MODEL_SNIPPET["IsolationForest"])
        ctor = ctor.replace("CONTAM", repr(float(contamination)))
        scale_line = ("    X_score = StandardScaler().fit_transform(X)\n"
                      if needs_scale else "    X_score = X\n")
        return f'''import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
{imp}


def run(data_path: str) -> dict:
    """兜底模板：{algorithm} 异常检测。"""
    df = pd.read_csv(data_path)
    y_true = df.pop("label").values if "label" in df.columns else None
    X = df.values.astype(float)
{scale_line}
    model = {ctor}
    model.fit(X_score)
    scores = -model.decision_function(X_score)
    y_pred = (model.predict(X_score) == -1).astype(int)

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

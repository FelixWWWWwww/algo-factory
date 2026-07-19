"""
T2.4 Planner Agent：TaskCard + data_profile + 检索上下文 → 候选方案(Plan)

- real 模式：调 LLM（planner_prompt.jinja2）依据【数据画像】动态推荐 2-3 个方案，废除固定算法池。
- mock / LLM 失败：回退到内置默认方案（原三件套），保证离线可跑、测试全绿。
- 无论哪条路径：对「本场景历史失败过的算法」降级排后并标注（闭环学习行动侧）。
"""
import json
import logging
from factory.agents.base import Agent
from factory.agents._util import parse_json
from factory.state import TaskState, Plan
from factory.llm.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)

# 内置默认方案（仅作 mock / LLM 失败时的兜底，不再是唯一来源）
_CANDIDATES = [
    ("Isolation Forest 方案", "IsolationForest", 0.76,
     "基于随机分裂隔离离群点，对高维和量纲不敏感、训练快，异常检测稳健首选。",
     ["load_data", "eda", "preprocess(跳过标准化)", "train_iforest", "evaluate(PR-AUC)"]),
    ("LOF 方案", "LocalOutlierFactor", 0.73,
     "基于局部密度识别离群点，擅长发现局部异常簇；须先 StandardScaler 且 novelty=True。",
     ["load_data", "eda", "preprocess(StandardScaler)", "train_lof", "evaluate(PR-AUC)"]),
    ("One-Class SVM 方案", "OneClassSVM", 0.71,
     "学习正常样本边界，适合半监督设定；对量纲极敏感，须先标准化。",
     ["load_data", "eda", "preprocess(StandardScaler)", "train_ocsvm", "evaluate(PR-AUC)"]),
]


def _num(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


class PlannerAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Planner", llm_client=llm_client)
        try:
            self.prompt_mgr = get_prompt_manager()
        except Exception as e:
            logger.warning(f"[planner] prompt_manager 初始化失败: {e}")
            self.prompt_mgr = None

    def _run(self, state: TaskState) -> TaskState:
        contamination = state.contamination or 0.05
        use_mock = getattr(state, "_use_mock", True)

        plans = None
        if not use_mock and self.llm_client is not None:
            plans = self._llm_plans(state, contamination)      # 真·动态选型
            if plans:
                logger.info(f"[planner] LLM 动态选型 → {[p.algorithm for p in plans]}")
        if not plans:
            plans = self._default_plans(contamination)          # 兜底：内置三件套

        plans = self._apply_demotion(plans, self._known_failures(state))
        state.plans = plans
        return state

    # ── real：LLM 依据 data_profile 动态推荐 ─────────────────────────
    def _llm_plans(self, state: TaskState, contamination: float):
        if self.prompt_mgr is None:
            return None
        try:
            J = lambda o: json.dumps(o, ensure_ascii=False, indent=2)
            tc = state.task_card.model_dump() if hasattr(state.task_card, "model_dump") else dict(state.task_card)
            ctx = state.retrieved_context.model_dump() if hasattr(state.retrieved_context, "model_dump") else {}
            prompt = self.prompt_mgr.render_template(
                "planner_prompt.jinja2",
                task_card_json=J(tc), task_card=tc,
                data_profile_json=J(state.data_profile or {}),
                retrieved_context=ctx,
            )
            resp = self.llm_client.chat([
                {"role": "system", "content": self.prompt_mgr.get_system_prompt()},
                {"role": "user", "content": prompt},
            ])
            raw = parse_json(resp.get("message", ""))
            if not isinstance(raw, list) or not raw:
                return None
            plans = []
            for p in raw[:3]:
                if not isinstance(p, dict) or not p.get("algorithm"):
                    continue
                rationale = (str(p.get("rationale", "")) + " " + str(p.get("fit_reason", ""))).strip()
                plans.append(Plan(
                    name=p.get("name") or f"{p['algorithm']} 方案",
                    algorithm=p["algorithm"],
                    import_path=p.get("import_path", ""),
                    pipeline_steps=p.get("preprocessing") or [],
                    rationale=rationale or "LLM 动态推荐",
                    expected_metric=_num(p.get("expected_metric"), 0.7),
                    contamination=contamination,
                ))
            return plans or None
        except Exception as e:
            logger.warning(f"[planner] LLM 选型失败，回退默认: {e}")
            return None

    # ── mock / 兜底：内置默认方案 ────────────────────────────────────
    _KNOWN_PATH = {
        "IsolationForest": "sklearn.ensemble.IsolationForest",
        "LocalOutlierFactor": "sklearn.neighbors.LocalOutlierFactor",
        "OneClassSVM": "sklearn.svm.OneClassSVM",
    }

    def _default_plans(self, contamination: float):
        return [
            Plan(name=name, algorithm=algo, import_path=self._KNOWN_PATH.get(algo, ""),
                 pipeline_steps=list(steps), rationale=rat,
                 expected_metric=exp, contamination=contamination)
            for name, algo, exp, rat, steps in _CANDIDATES
        ]

    # ── 闭环学习：历史失败算法降级排后 ───────────────────────────────
    def _known_failures(self, state: TaskState) -> set:
        out = set()
        try:
            for fc in state.retrieved_context.failure_cases:
                a = fc.get("algorithm") if isinstance(fc, dict) else None
                if a:
                    out.add(a)
        except Exception:
            pass
        return out

    def _apply_demotion(self, plans, failed_algos):
        if not failed_algos:
            return plans
        tagged = []
        for p in plans:
            demoted = p.algorithm in failed_algos
            if demoted and "历史失败" not in p.rationale:
                p.rationale = "⚠️ 历史失败已规避（本场景该算法曾验证不达标，降级排后）：" + p.rationale
            tagged.append((demoted, p))
        tagged.sort(key=lambda t: t[0])   # 未失败在前
        return [p for _, p in tagged]

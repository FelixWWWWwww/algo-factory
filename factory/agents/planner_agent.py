"""
T2.4 Planner Agent：TaskCard + 检索上下文 → 多个候选方案(Plan)

闭环学习的"行动侧"：若某算法在本场景历史失败（来自 retrieved_context），
则将其**降级并排到最后**，rationale 明确标注"历史失败已规避"，
从而体现"从失败中学习、下次规避"。
"""
import logging
from factory.agents.base import Agent
from factory.agents._util import parse_json
from factory.state import TaskState, Plan
from factory.llm.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)

# (name, algorithm, expected_pr_auc, rationale, pipeline_steps)
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


class PlannerAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Planner", llm_client=llm_client)
        try:
            self.prompt_mgr = get_prompt_manager()
        except Exception as e:
            logger.warning(f"[planner] prompt_manager 初始化失败，禁用 LLM 理由: {e}")
            self.prompt_mgr = None

    def _run(self, state: TaskState) -> TaskState:
        contamination = state.contamination or 0.05
        failed_algos = self._known_failures(state)      # 历史失败算法集合
        llm_note = self._llm_rationale(state)
        built = []
        for name, algo, exp, rat, steps in _CANDIDATES:
            rationale = rat + (f" [LLM补充] {llm_note}" if llm_note else "")
            demoted = algo in failed_algos
            if demoted:
                rationale = "⚠️ 历史失败已规避（本场景该算法曾验证不达标，降级排后）：" + rationale
            built.append((demoted, Plan(
                name=name, algorithm=algo, pipeline_steps=list(steps),
                rationale=rationale, expected_metric=exp, contamination=contamination,
            )))
        # 历史失败的算法排到最后（降级）；未失败的优先
        built.sort(key=lambda t: t[0])
        state.plans = [p for _, p in built]
        if failed_algos:
            logger.info(f"[planner] 依据图谱经验规避/降级算法: {sorted(failed_algos)}")
        return state

    def _known_failures(self, state: TaskState) -> set:
        """从检索上下文提取本场景历史失败过的算法名。"""
        out = set()
        try:
            for fc in state.retrieved_context.failure_cases:
                a = fc.get("algorithm") if isinstance(fc, dict) else None
                if a:
                    out.add(a)
        except Exception:
            pass
        return out

    def _llm_rationale(self, state: TaskState) -> str:
        if self.llm_client is None or self.prompt_mgr is None:
            return ""
        try:
            df = getattr(state, "raw_df", None)
            n_samples = len(df) if df is not None else 0
            n_features = int(df.shape[1]) if df is not None else 0
            prompt = self.prompt_mgr.get_planner_prompt(
                user_query=state.user_query, n_samples=n_samples, n_features=n_features,
                anomaly_ratio=state.anomaly_ratio or 0.0,
                has_labels=getattr(state, "y_true", None) is not None,
                retrieved_context=str(state.retrieved_context),
            )
            resp = self.llm_client.chat([
                {"role": "system", "content": self.prompt_mgr.get_system_prompt()},
                {"role": "user", "content": prompt},
            ])
            data = parse_json(resp.get("message", ""))
            if isinstance(data, dict):
                return str(data.get("rationale", "")).strip()
        except Exception as e:
            logger.warning(f"[planner] LLM 补充理由失败，忽略: {e}")
        return ""

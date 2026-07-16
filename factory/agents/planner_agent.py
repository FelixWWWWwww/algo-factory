"""
T2.4 Planner Agent：TaskCard + 检索上下文 → 多个候选方案(Plan)

产出 3 个候选方案（IForest / LOF / OCSVM），供后续多方案自动比较。
可选调用 LLM 补充自然语言 rationale（命中"自然语言解释设计依据"加分项）。
"""

from factory.agents.base import Agent
from factory.agents._util import parse_json
from factory.state import TaskState, Plan

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

from factory.llm.prompt_manager import get_prompt_manager


class PlannerAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Planner", llm_client=llm_client)
        self.prompt_mgr = get_prompt_manager()

    def _run(self, state: TaskState) -> TaskState:
        # ... 基础方案生成 ...

        # 用 LLM 生成补充理由
        llm_rationale = self._llm_rationale(state)

        # 写入 plans
        # ...
        return state

    def _llm_rationale(self, state: TaskState) -> str:
        """用 LLM 补充理由"""
        if self.llm_client is None:
            return ""

        try:
            # 用 Jinja2 渲染 planner_prompt
            prompt = self.prompt_mgr.get_planner_prompt(
                user_query=state.user_query,
                n_samples=len(state.X_raw) if hasattr(state, 'X_raw') and state.X_raw is not None else 0,
                n_features=state.X_raw.shape[1] if hasattr(state, 'X_raw') and state.X_raw is not None else 0,
                anomaly_ratio=state.anomaly_ratio or 0.0,
                has_labels=state.y_raw is not None,
                retrieved_context=str(state.retrieved_context)
            )

            # 调用 LLM
            response = self.llm_client.chat([
                {"role": "system", "content": self.prompt_mgr.get_system_prompt()},
                {"role": "user", "content": prompt}
            ])

            # 记录 LLM 调用
            from factory.llm.logger import get_llm_logger
            get_llm_logger().log_call(
                agent_name="Planner",
                prompt=prompt,
                response=response.get("message", ""),
                model=getattr(self.llm_client, 'model', 'unknown')
            )

            # 返回 rationale
            from factory.agents._util import parse_json
            data = parse_json(response.get("message", ""))
            if isinstance(data, list) and len(data) > 0:
                return str(data[0].get("rationale", "")).strip()

        except Exception as e:
            logger.warning(f"[Planner] LLM 补充失败: {e}")

        return ""


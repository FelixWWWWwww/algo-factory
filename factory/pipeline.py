# factory/pipeline.py
"""主编排：能力理解→检索→规划→生成→沙箱验证→修复→择优→图谱回写→报告。"""
from __future__ import annotations
import os, json, logging
from pathlib import Path

from factory.state import TaskState
from factory.llm import MockClient
from factory.agents import InterpreterAgent, RetrieverAgent, PlannerAgent, CoderAgent
from factory.agents.curator_agent import CuratorAgent
from factory.agents.coder_agent import CoderAgent as _Coder, _is_valid
from factory.graph.store import GraphStore
from factory.sandbox.validator import validate, load_config
from factory.nodes import make_synthetic_dataset
from factory.report import generate_report

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, use_mock: bool = True, max_retries: int = 2, work_dir: str = "data"):
        self.use_mock = use_mock
        self.max_retries = max_retries
        self.work_dir = work_dir
        self.llm = self._make_llm()
        self.graph_store = GraphStore()

    def _make_llm(self):
        if self.use_mock:
            return MockClient()
        from factory.llm import OpenAIClient
        return OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        )

    # ───────────────────────── 主流程 ─────────────────────────
    def run(self, query: str, data_path: str | None = None) -> TaskState:
        state = TaskState(user_query=query)
        state._use_mock = self.use_mock

        # a→c 理解 / 检索 / 规划
        state = InterpreterAgent(self.llm).run(state)
        state = RetrieverAgent(self.llm, graph_store=self.graph_store).run(state)
        state = PlannerAgent(self.llm).run(state)

        # 数据（未提供则合成）
        data_path = data_path or self._ensure_data(state)

        # d 生成代码
        state = CoderAgent(self.llm, out_dir=os.path.join(self.work_dir, "examples")).run(state)

        # e→f 逐方案：验证 + 修复循环
        config = load_config()
        for plan, cv in zip(state.plans, state.code_versions):
            report = self._validate_with_repair(plan, cv, data_path, config, state)
            status = "passed" if report.status == "passed" else "failed"
            state.add_validation_result(
                cv.version, plan.name, status,
                metrics=report.metrics,
                error_message=None if status == "passed" else report.message,
            )
            plan.actual_metric = (report.metrics or {}).get("pr_auc")
            plan.validation_status = "success" if status == "passed" else "failed"

        # 择优（T3.5）
        self._select_best(state)

        # g 图谱回写（T3.6）
        state = CuratorAgent(
            self.llm, graph_store=self.graph_store,
            graph_path=os.path.join(self.work_dir, "knowledge_graph.json"),
        ).run(state)

        # 报告（T3.7）
        state.final_status = "completed"
        try:
            generate_report(state)
        except Exception as e:
            logger.warning(f"[pipeline] 报告生成失败: {e}")
        return state

    # ───────────────────────── 子步骤 ─────────────────────────
    def _ensure_data(self, state: TaskState) -> str:
        d = os.path.join(self.work_dir, "synth")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{state.task_id}.csv")
        ratio = min(max(state.contamination or 0.03, 0.01), 0.05)
        make_synthetic_dataset(1500, 6, ratio, save_path=path)
        return path

    def _validate_with_repair(self, plan, cv, data_path, config, state):
        """T3.4：验证失败 → 带负面约束重生成 → 再验证，最多 max_retries 轮。"""
        report = validate(cv.code, data_path, config)
        retries = 0
        while report.status != "passed" and retries < self.max_retries:
            retries += 1
            logger.info(f"[repair] {plan.algorithm} 第 {retries} 轮修复（{report.failed_layer}）")
            cv.validation_error = report.message
            cv.code = self._repair(plan, report)
            report = validate(cv.code, data_path, config)
        cv.validation_metrics = report.metrics
        return report

    def _repair(self, plan, report) -> str:
        """带负面约束提示 Coder 重生成；不可用时退回模板。"""
        try:
            prompt = (
                f"上一版 {plan.algorithm} 异常检测代码(代码)验证失败："
                f"[{report.failed_layer}] {report.message}。请修复并避免相同错误，"
                "务必保留 def run(data_path)->dict 与 RESULT_JSON 输出。"
            )
            code = self.llm.chat([{"role": "user", "content": prompt}]).get("message", "")
            if code and "def run(" in code and "RESULT_JSON" in code and _is_valid(code):
                return code
        except Exception:
            pass
        return _Coder(self.llm)._template(plan.algorithm, plan.contamination or 0.05)

    def _select_best(self, state: TaskState):
        pairs = list(zip(state.plans, state.validation_results))
        passed = [(p, vr) for p, vr in pairs if vr.status == "passed"]
        pool = passed or pairs
        if not pool:
            return
        best_plan, best_vr = max(pool, key=lambda t: (t[1].metrics or {}).get("pr_auc", -1))
        best_plan.is_best = True
        state.best_model = best_plan.algorithm
        state.metrics = best_vr.metrics or {}
        state.final_metrics = {k: v for k, v in (best_vr.metrics or {}).items()
                               if not str(k).startswith("accuracy")}
        for cv in state.code_versions:
            if cv.plan_name == best_plan.name:
                state.final_code = cv.code

    # ───────────────────────── 落盘 ─────────────────────────
    def dump_state(self, state: TaskState, output_dir: str = "logs") -> str:
        """只序列化 JSON-safe 字段（跳过 raw_df / 模型 / ndarray）。"""
        os.makedirs(output_dir, exist_ok=True)
        safe = {
            "task_id": state.task_id,
            "user_query": state.user_query,
            "task_card": state.task_card.model_dump() if hasattr(state.task_card, "model_dump") else {},
            "plans": [p.model_dump() for p in state.plans],
            "validation_results": [vr.model_dump() for vr in state.validation_results],
            "best_model": state.best_model,
            "final_metrics": state.final_metrics,
            "eda_summary": state.eda_summary,
            "train_info": state.train_info,
            "final_status": state.final_status,
        }
        path = os.path.join(output_dir, f"{state.task_id}.json")
        Path(path).write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
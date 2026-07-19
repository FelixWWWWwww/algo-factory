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
from factory.nodes import (make_synthetic_dataset, profile_node, data_ingestion_node, eda_node,
                           preprocess_node, split_node, train_node, evaluate_node)
from factory.report import generate_report

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, use_mock: bool = True, max_retries: int = 2, work_dir: str = "data"):
        self.use_mock = use_mock
        self.max_retries = max_retries
        self.work_dir = work_dir
        self.llm = self._make_llm()
        self.graph_store = GraphStore()
        # 加载已持久化的图谱：让本次运行能读到历史失败经验（闭环学习的关键）
        self._graph_path = os.path.join(work_dir, "knowledge_graph.json")
        if os.path.exists(self._graph_path):
            try:
                self.graph_store.load(self._graph_path)
            except Exception as e:
                logger.warning(f"[pipeline] 图谱加载失败，从空图开始: {e}")

    def _make_llm(self):
        if self.use_mock:
            return MockClient()
        from factory.llm import OpenAIClient
        # 不硬塞默认值：全部交给 OpenAIClient 的环境变量逻辑
        #   api_key : DEEPSEEK_API_KEY -> OPENAI_API_KEY
        #   base_url: OPENAI_BASE_URL  -> LLM_BASE_URL
        #   model   : LLM_MODEL -> OPENAI_MODEL -> 默认 deepseek-v4-pro
        return OpenAIClient()

    # ───────────────────────── 主流程 ─────────────────────────
    def run(self, query: str, data_path: str | None = None) -> TaskState:
        state = TaskState(user_query=query)
        state._use_mock = self.use_mock

        # a 理解需求
        state = InterpreterAgent(self.llm).run(state)

        # 数据就绪（未提供则合成）→ 数据画像（供 Planner 动态选型）
        data_path = data_path or self._ensure_data(state)
        profile_node(state, data_path)

        # b 检索经验 / c 动态规划（Planner 现在能看到 data_profile）
        state = RetrieverAgent(self.llm, graph_store=self.graph_store).run(state)
        state = PlannerAgent(self.llm).run(state)

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
        # 富化报告：用最优算法跑一遍 node 流水线，填充 EDA/异常分数/Top-K
        self._enrich_artifacts(state, data_path)

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

    def _validate_with_repair(self, plan, cv, data_path, config, state=None):
        """T3.4：验证失败 → 纯 LLM 修复最多 N 轮；N 次耗尽后才启用模板兜底拦截器。"""
        report = validate(cv.code, data_path, config)
        retries = 0
        while report.status != "passed" and retries < self.max_retries:
            retries += 1
            logger.info(f"[repair] {plan.algorithm} 第 {retries} 轮 LLM 修复（{report.failed_layer}）")
            cv.validation_error = report.message
            cv.code = self._llm_repair(plan, report, cv.code)   # 纯 LLM，不塞模板
            report = validate(cv.code, data_path, config)
        if report.status != "passed":
            # 连续 N 次仍失败 → 底层拦截器启用模板兜底
            logger.info(f"[repair] {plan.algorithm} 修复 {self.max_retries} 次仍失败 → 启用模板兜底")
            cv.code = _Coder(self.llm)._template(plan.algorithm, plan.contamination or 0.05)
            report = validate(cv.code, data_path, config)
        cv.validation_metrics = report.metrics
        return report

    def _llm_repair(self, plan, report, prev_code: str) -> str:
        """纯 LLM 带负面约束重写；产出不可用则保留上一版（不在修复阶段塞模板）。"""
        try:
            prompt = (
                f"上一版 {plan.algorithm} 异常检测代码(代码)验证失败："
                f"[{report.failed_layer}] {report.message}。请修复并避免相同错误，"
                "务必保留 def run(data_path)->dict 与 RESULT_JSON 输出。"
            )
            code = self.llm.chat([{"role": "user", "content": prompt}]).get("message", "")
            if (code and "def run(" in code and "RESULT_JSON" in code
                    and plan.algorithm in code and _is_valid(code)):
                return code
        except Exception:
            pass
        return prev_code

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
    def _enrich_artifacts(self, state: TaskState, data_path: str):
        """用最优算法跑一遍 node 流水线，为报告填充 EDA 摘要 / 异常分数 / Top-K。
        不覆盖沙箱择优的 final_metrics（结束后复位）。任何失败都吞掉，不影响主流程。"""
        best_algo = state.best_model or "IsolationForest"
        best_plan = next((p for p in state.plans if getattr(p, "is_best", False)), None) \
            or next((p for p in state.plans if p.algorithm == best_algo), None)
        best_import = getattr(best_plan, "import_path", "") if best_plan else ""
        saved_final = dict(state.final_metrics or {})
        try:
            data_ingestion_node(state, data_path)
            eda_node(state, self.llm)
            preprocess_node(state, algorithm=best_algo)
            split_node(state)
            train_node(state, algorithm=best_algo, import_path=best_import)
            evaluate_node(state)
        except Exception as e:
            logger.warning(f"[pipeline] 报告富化失败（不影响主流程）: {e}")
        finally:
            state.raw_df = None
            state.trained_model = None
            if saved_final:
                state.final_metrics = saved_final

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

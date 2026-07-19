"""
ModelSelection Agent: explain why the best model was selected.
"""

from __future__ import annotations

from typing import List, Dict, Any

from factory.agents.base import Agent
from factory.state import TaskState


class ModelSelectionAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="ModelSelection", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        comparison: List[Dict[str, Any]] = list(getattr(state, "model_comparison", []) or [])
        best_model = getattr(state, "best_model_name", "") or getattr(state, "best_model", "")

        if not best_model or not comparison:
            state.model_selection_explanation = {
                "best_model": best_model,
                "rationale": "缺少模型对比结果，无法生成解释。",
            }
            return state

        ranked = sorted(
            comparison,
            key=lambda item: item.get("pr_auc") if item.get("pr_auc") is not None else item.get("detection_rate", 0),
            reverse=True,
        )
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        prompt = (
            "你是异常检测建模助手，请用简洁中文解释为什么选择当前最优模型。\n"
            f"最优模型: {best_model}\n"
            f"最佳结果: {best}\n"
            f"次优结果: {second if second else '无'}\n"
            "请输出 2-4 句，重点说明 PR-AUC、检出率、以及是否适合当前数据特征。"
        )

        rationale = self._fallback_rationale(best_model, best, second)
        if self.llm_client is not None:
            try:
                response = self.llm_client.chat([{"role": "user", "content": prompt}])
                if not response.get("error"):
                    message = (response.get("message") or "").strip()
                    if message:
                        rationale = message
            except Exception:
                pass

        state.model_selection_explanation = {
            "best_model": best_model,
            "rationale": rationale,
            "best_result": best,
            "second_result": second,
        }
        state.best_model = best_model
        return state

    @staticmethod
    def _fallback_rationale(best_model: str, best: Dict[str, Any], second: Dict[str, Any] | None) -> str:
        metric = best.get("pr_auc")
        if metric is None:
            metric = best.get("detection_rate")
            metric_name = "检出率"
        else:
            metric_name = "PR-AUC"

        parts = [f"当前选择 {best_model}，因为它在候选模型中表现最好，{metric_name} 为 {metric}。"]
        if second:
            second_metric = second.get("pr_auc")
            if second_metric is None:
                second_metric = second.get("detection_rate")
                second_metric_name = "检出率"
            else:
                second_metric_name = "PR-AUC"
            parts.append(
                f"次优模型 {second.get('algorithm', 'unknown')} 的 {second_metric_name} 为 {second_metric}，"
                f"与最优模型相比略低。"
            )
        parts.append("因此优先采用当前模型作为最终方案。")
        return " ".join(parts)

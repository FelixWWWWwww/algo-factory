"""
T2.4 Interpreter Agent：自然语言需求 → 结构化 TaskCard

真实模式：调用 llm_client.chat() 抽取结构（离线可用 MockClient）
兜底：LLM 缺失/解析失败时，回退关键字规则，保证永远产出合法 TaskCard
"""

from factory.agents.base import Agent
from factory.agents._util import parse_json
from factory.state import TaskState, TaskCard

_DEFAULT_METRICS = ["pr_auc", "f1", "recall", "precision"]


class InterpreterAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Interpreter", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        card = self._llm_extract(state.user_query) or self._fallback(state.user_query)
        state.task_card = card
        # 把 contamination 同步到全局，供后续节点统一使用
        if card.contamination:
            state.contamination = card.contamination
        return state

    def _llm_extract(self, query: str):
        if self.llm_client is None:
            return None
        prompt = (
            "你是异常检测任务解析器。请把用户需求解析为 JSON 任务卡（task）。\n"
            "字段：task_type / target / constraints(list) / metrics(list) / "
            "data_hint / contamination(float)。\n"
            f"用户需求：{query}\n只输出 JSON。"
        )
        try:
            resp = self.llm_client.chat([{"role": "user", "content": prompt}])
            data = parse_json(resp.get("message", ""))
            if not isinstance(data, dict):
                return None
            return TaskCard(
                task_type=data.get("task_type", "anomaly_detection"),
                target=data.get("target") or query,
                constraints=data.get("constraints") or [],
                metrics=data.get("metrics") or list(_DEFAULT_METRICS),
                data_hint=data.get("data_hint", ""),
                contamination=float(data.get("contamination") or 0.05),
            )
        except Exception:
            return None

    def _fallback(self, query: str) -> TaskCard:
        return TaskCard(
            task_type="anomaly_detection",
            target=query,
            constraints=["极度不平衡", "禁止过采样/SMOTE", "禁用 accuracy 选优"],
            metrics=list(_DEFAULT_METRICS),
            data_hint="",
            contamination=0.05,
        )

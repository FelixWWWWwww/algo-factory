# factory/agents/curator_agent.py
"""T3.6 Curator Agent：验证结果 → 知识图谱回写（能力沉淀闭环的 g 环节）。"""
import logging
from factory.agents.base import Agent
from factory.state import TaskState
from factory.graph.store import GraphStore

logger = logging.getLogger(__name__)


class CuratorAgent(Agent):
    def __init__(self, llm_client=None, graph_store: GraphStore = None,
                 graph_path: str = "data/knowledge_graph.json"):
        super().__init__(name="Curator", llm_client=llm_client)
        self.graph_store = graph_store or GraphStore()
        self.graph_path = graph_path

    def _run(self, state: TaskState) -> TaskState:
        gs = self.graph_store
        task_type = getattr(state.task_card, "task_type", "anomaly_detection")
        cap_id = f"capability:{task_type}"
        gs.add_node(cap_id, type="Capability", task_type=task_type,
                    target=getattr(state.task_card, "target", ""))

        for vr in state.validation_results:
            run_id = f"validationrun:{state.task_id}:{vr.version}"
            gs.add_node(run_id, type="ValidationRun", task_type=task_type,
                        timestamp=vr.timestamp, status=vr.status,
                        pr_auc=(vr.metrics or {}).get("pr_auc"),
                        f1=(vr.metrics or {}).get("f1"))
            gs.add_edge(cap_id, run_id, "VALIDATED_IN")

            algo_id = f"algorithm:{vr.plan_name}"
            gs.add_node(algo_id, type="Algorithm", name=vr.plan_name)
            gs.add_edge(cap_id, algo_id, "USES_ALGORITHM")

            if vr.status == "failed":
                fc_id = f"failurecase:{state.task_id}:{vr.version}"
                gs.add_node(fc_id, type="FailureCase", reason=vr.error_message or "unknown")
                gs.add_edge(run_id, fc_id, "CAUSED_LESSON")

        try:
            gs.save(self.graph_path)
            gs.export_graphml(self.graph_path.replace(".json", ".graphml"))
            logger.info(f"[curator] 图谱已回写: {self.graph_path}")
        except Exception as e:
            logger.warning(f"[curator] 图谱保存失败: {e}")
        return state
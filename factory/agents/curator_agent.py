# factory/agents/curator_agent.py
"""T3.6 Curator Agent：验证结果 → 知识图谱回写（能力沉淀闭环的 g 环节）。

写入三类"可被下次检索"的知识：
  - ValidationRun：每次验证记录（含 algorithm / status / pr_auc）
  - FailureCase ：失败审计痕迹（带 task_type + algorithm，供 Retriever 精确查询）
  - Lesson      ：稳定的可复用教训（按 task_type+algorithm 去重，跨运行累积）
"""
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

        # plan_name -> algorithm 映射（ValidationResult 只存 plan_name）
        algo_of = {p.name: p.algorithm for p in state.plans}

        for vr in state.validation_results:
            algo = algo_of.get(vr.plan_name, vr.plan_name)
            run_id = f"validationrun:{state.task_id}:{vr.version}"
            gs.add_node(run_id, type="ValidationRun", task_type=task_type, algorithm=algo,
                        timestamp=vr.timestamp, status=vr.status,
                        pr_auc=(vr.metrics or {}).get("pr_auc"),
                        f1=(vr.metrics or {}).get("f1"))
            gs.add_edge(cap_id, run_id, "VALIDATED_IN")

            algo_id = f"algorithm:{algo}"
            gs.add_node(algo_id, type="Algorithm", name=algo)
            gs.add_edge(cap_id, algo_id, "USES_ALGORITHM")

            if vr.status == "failed":
                # 审计痕迹：每次运行一条（带 algorithm/task_type，可精确检索）
                fc_id = f"failurecase:{state.task_id}:{vr.version}"
                gs.add_node(fc_id, type="FailureCase", task_type=task_type, algorithm=algo,
                            reason=vr.error_message or "unknown")
                gs.add_edge(run_id, fc_id, "CAUSED_LESSON")
                # 可复用教训：按 (task_type, algorithm) 去重，跨运行累积
                lesson_id = f"lesson:{task_type}:{algo}"
                gs.add_node(lesson_id, type="Lesson", task_type=task_type, algorithm=algo,
                            content=f"{algo} 在 {task_type} 场景曾验证失败（{vr.error_message or 'unknown'}），"
                                    f"下次应规避或降级。")
                gs.add_edge(fc_id, lesson_id, "CAUSED_LESSON")
                gs.add_edge(cap_id, lesson_id, "HAS_LESSON")

        try:
            gs.save(self.graph_path)
            gs.export_graphml(self.graph_path.replace(".json", ".graphml"))
            logger.info(f"[curator] 图谱已回写: {self.graph_path}")
        except Exception as e:
            logger.warning(f"[curator] 图谱保存失败: {e}")
        return state

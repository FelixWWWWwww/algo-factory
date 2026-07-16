"""
factory/pipeline.py
异常检测任务的执行流水线（状态机）
"""

import uuid
from datetime import datetime
from pathlib import Path
from factory.state import TaskState
from factory.llm.client import LLMClient
from factory.llm.mock_client import MockClient
from factory.agents.base import Agent
from factory.agents.interpreter_agent import InterpreterAgent
from factory.agents.retriever_agent import RetrieverAgent
from factory.agents.planner_agent import PlannerAgent
from factory.agents.coder_agent import CoderAgent
from factory.agents.validator_agent import ValidatorAgent
from factory.agents.curator_agent import CuratorAgent


class Pipeline:
    """
    异常检测算法工厂的执行流水线。

    执行流程（Day 1 Mock 模式）：
    1. Interpreter：自然语言需求 → 结构化 TaskCard
    2. Retriever：TaskCard → 检索知识图谱 → retrieved_context
    3. Planner：TaskCard + context → 多个候选方案 plans
    4. Coder：plans → 生成代码（每个方案一份）
    5. Validator：代码 → 执行 + 评估（仅 Mock 返回假结果）
    6. Curator：验证结果 → 图谱回写（Day 3 才真实回写）

    执行逻辑：
    - Day 1 Mock：各 Agent 返回预定义的假数据
    - Day 2-3 真实：各 Agent 接通真实 LLM + 模型 + 图谱
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

        # 初始化 LLM 客户端
        if use_mock:
            self.llm = MockClient()
        else:
            from factory.llm.openai_client import OpenAIClient
            self.llm = OpenAIClient()

        # 初始化各 Agent
        self.agents: dict[str, Agent] = {
            "interpreter": InterpreterAgent(llm_client=self.llm),
            "retriever": RetrieverAgent(llm_client=self.llm),
            "planner": PlannerAgent(llm_client=self.llm),
            "coder": CoderAgent(llm_client=self.llm),
            "validator": ValidatorAgent(llm_client=self.llm),
            "curator": CuratorAgent(llm_client=self.llm),
        }

    def run(self, user_query: str) -> TaskState:
        """
        主执行方法。

        Args:
            user_query: 用户输入的自然语言需求
                       如 "对工业传感器数据进行异常检测"

        Returns:
            state: 最终的 TaskState，包含所有中间结果
        """

        # 初始化状态
        state = TaskState(
            task_id=str(uuid.uuid4())[:8],
            user_query=user_query,
            _use_mock=self.use_mock
        )

        print(f"\n🚀 任务启动")
        print(f"   Task ID: {state.task_id}")
        print(f"   Query: {user_query}")
        print(f"   Mock 模式: {self.use_mock}")

        # ===== 执行流水线 =====

        # Step 1: Interpreter
        state = self.agents["interpreter"].run(state)

        # Step 2: Retriever
        state = self.agents["retriever"].run(state)

        # Step 3: Planner
        state = self.agents["planner"].run(state)

        # Step 4: Coder（可能生成多份代码）
        state = self.agents["coder"].run(state)

        # Step 5: Validator（执行代码，获取指标）
        state = self.agents["validator"].run(state)

        # Step 6: Curator（图谱回写）
        state = self.agents["curator"].run(state)

        # ===== 任务完成 =====
        state.final_status = "completed"

        print(f"\n{'=' * 60}")
        print(f"✅ 所有 Agent 执行完成")
        print(f"{'=' * 60}")
        print(f"\n最终结果摘要：")
        print(f"  - Task ID: {state.task_id}")
        print(f"  - Best Model: {state.best_model}")
        print(f"  - PR-AUC: {state.metrics.get('pr_auc', 'N/A')}")
        error_count = 0
        for record in state.error_history:
            status = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
            if status == "error":
                error_count += 1
        print(f"  - 错误数: {error_count}")

        return state

    def dump_state(self, state: TaskState, output_dir: str = "logs"):
        """
        将 TaskState 序列化到 JSON 文件。
        """
        Path(output_dir).mkdir(exist_ok=True)

        output_file = Path(output_dir) / f"{state.task_id}.json"

        # TaskState 是 Pydantic BaseModel，可直接 .model_dump_json()
        json_str = state.model_dump_json(indent=2, exclude_none=True)
        output_file.write_text(json_str, encoding="utf-8")

        print(f"\n📝 状态已保存: {output_file}")
        return output_file

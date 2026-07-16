"""
Curator Agent：验证结果 → 图谱回写（更新知识）
"""

from factory.agents.base import Agent
from factory.state import TaskState


class CuratorAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Curator", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        """
        Mock 模式：打印一下会做什么，但不真的回写
        真实模式（Day 3）：将 ValidationRun 节点和 Lesson 写入图谱
        """
        if state._use_mock:
            print(f"\n  [Curator Mock] 将执行以下操作：")
            print(f"    1. 创建 ValidationRun 节点")
            print(f"    2. 更新 {state.best_model} 的 success_rate")
            print(f"    3. 失败 case 记录为 FailureCase 节点")
            print(f"    4. 提取教训，更新 Lesson 节点")
            return state

        raise NotImplementedError("真实 Curator 在 Day 3 实现")

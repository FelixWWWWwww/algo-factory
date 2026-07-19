"""
factory/dag_pipeline.py
基于数据节点的 DAG 流水线（Day 2 核心）
"""

import logging
from pathlib import Path
from factory.state import TaskState
from factory.llm.mock_client import MockClient
from factory.llm.openai_client import OpenAIClient
from factory.nodes.ingestion import data_ingestion_node, eda_node
from factory.nodes.preprocess import preprocess_node
from factory.nodes.split import split_node
from factory.nodes.train import train_node
from factory.nodes.evaluate import evaluate_node
from factory.nodes.model_selection import model_selection_node

logger = logging.getLogger(__name__)


class DAGPipeline:
    """
    数据驱动的 DAG 流水线。

    流程（Day 2 Mock）：
      1. 接入 → ingestion_node（读 CSV）
      2. EDA → 内置（计算 anomaly_ratio）
      3. 预处理 → preprocess_node（标准化）
      4. 切分 → split_node（train/test）
      5. 多算法训练 → model_selection_node（并行 fit 三个算法）
      6. 评估 → evaluate_node（计算 PR-AUC）

    条件分叉：
      - 有 y_test → 计算 PR-AUC / F1
      - 无标签 → Top-K 人工审阅
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.graph = {}  # DAG 的边信息（可视化用）
        self.llm_client = MockClient() if use_mock else OpenAIClient()

    def run(self, data_path: str) -> TaskState:
        """
        主流水线。

        Args:
            data_path: CSV 文件路径

        Returns:
            state: 最终的 TaskState
        """

        # 初始化状态
        state = TaskState(
            task_id="dag_task_001",
            user_query=f"异常检测: {data_path}",
            _use_mock=self.use_mock
        )

        logger.info(f"\n🚀 DAG Pipeline 启动")
        logger.info(f"   Data: {data_path}")
        logger.info(f"   Mock: {self.use_mock}")

        # ===== DAG 节点执行 =====

        # 1. 接入
        logger.info(f"\n→ [1/6] 数据接入...")
        state = data_ingestion_node(state, csv_path=data_path)
        if state.raw_df is None:
            logger.error("[DAG] 数据接入失败，中止")
            return state

        # 2. EDA
        logger.info(f"→ [2/6] EDA 分析...")
        state = eda_node(state, self.llm_client)
        if state.anomaly_ratio is not None:
            logger.info(f"   异常占比: {state.anomaly_ratio:.2%}")

        # 3. 预处理
        logger.info(f"→ [3/6] 数据预处理...")
        state = preprocess_node(state, algorithm="OneClassSVM")
        if state.X_processed is None:
            logger.error("[DAG] 预处理失败")
            return state

        # 4. 切分
        logger.info(f"→ [4/6] 数据切分...")
        state = split_node(
            state,
            test_size=0.3,
            random_state=42
        )

        # 5. 多算法训练 + 选优（DAG 的关键节点）
        logger.info(f"→ [5/6] 多算法训练与选优...")
        state = model_selection_node(
            state,
            algorithms=["IForest", "LOF", "OCSVM"]
        )
        if not getattr(state, "best_model_name", ""):
            logger.error("[DAG] 没有成功的模型")
            return state

        # 5.5 Agent 解释选择理由（NEW in T2.8）
        logger.info(f"→ [5.5/6] Agent 生成选择理由...")
        from factory.agents.model_selection_agent import ModelSelectionAgent
        model_selection_agent = ModelSelectionAgent(llm_client=self.llm_client)
        state = model_selection_agent.run(state)

        # 6. 用最优模型评估
        logger.info(f"→ [6/6] 评估...")
        state = train_node(state, algorithm=state.best_model_name)
        state = evaluate_node(state, topk=20)

        # ===== 完成 =====
        state.final_status = "completed"
        logger.info(f"\n✅ DAG Pipeline 完成")
        logger.info(f"   最优模型: {state.best_model_name}")
        logger.info(f"   PR-AUC: {state.eval_metrics.get('pr_auc', 'N/A')}")

        return state

    def dump_state(self, state: TaskState, output_dir: str = "logs"):
        """保存状态到 JSON"""
        import json

        Path(output_dir).mkdir(exist_ok=True)
        output_file = Path(output_dir) / f"{state.task_id}_dag.json"

        data = state.model_dump(exclude_none=True)

        for key in ["raw_df", "X_processed", "X_train", "X_test", "y_true", "y_train", "y_test"]:
            value = data.get(key)
            if value is None:
                continue
            if hasattr(value, "to_dict"):
                data[key] = value.to_dict(orient="records")
            elif hasattr(value, "tolist"):
                data[key] = value.tolist()

        json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        output_file.write_text(json_str, encoding="utf-8")
        logger.info(f"📝 状态已保存: {output_file}")
        return output_file

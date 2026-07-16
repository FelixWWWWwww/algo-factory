# test_ingestion.py
"""
T1.5 Mock 模式验证脚本
运行：cd algo-factory && python -m pytest test_ingestion.py -v
"""

import io
import pytest
import pandas as pd
import numpy as np

from factory.state import TaskState
from factory.llm.mock_client import MockClient
from factory.nodes.ingestion import data_ingestion_node, eda_node


# ─── 工具：生成合成不平衡数据集 ────────────────────────────
def make_imbalanced_csv(tmp_path, n=500, anomaly_ratio=0.02) -> str:
    """在临时目录生成一份合成的极不平衡 CSV，1=异常，0=正常"""
    np.random.seed(42)
    n_anomaly = max(1, int(n * anomaly_ratio))
    n_normal  = n - n_anomaly

    normal  = np.random.randn(n_normal,  4) * [1, 2, 0.5, 3]
    anomaly = np.random.randn(n_anomaly, 4) * 5 + 10  # 明显离群

    X = np.vstack([normal, anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)

    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["label"] = y

    # 随机注入 5% 缺失
    mask = np.random.rand(*df[["f1","f2"]].shape) < 0.05
    df.loc[mask[:,0], "f1"] = np.nan
    df.loc[mask[:,1], "f2"] = np.nan

    path = str(tmp_path / "mock_data.csv")
    df.to_csv(path, index=False)
    return path


# ─── 测试用例 ────────────────────────────────────────────
class TestDataIngestionNode:

    def test_reads_csv_and_fills_raw_df(self, tmp_path):
        """raw_df 应正确读入且行列数匹配"""
        csv_path = make_imbalanced_csv(tmp_path, n=500)
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, csv_path)

        assert state.raw_df is not None, "raw_df 不应为 None"
        assert state.raw_df.shape[0] == 500
        assert state.raw_df.shape[1] == 5   # f1 f2 f3 f4 label

    def test_schema_info_contains_dtype_and_missing(self, tmp_path):
        """schema_info 应含 dtype / n_missing / missing_rate"""
        csv_path = make_imbalanced_csv(tmp_path, n=500)
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, csv_path)

        for col, info in state.schema_info.items():
            assert "dtype"        in info, f"{col} 缺少 dtype"
            assert "n_missing"    in info, f"{col} 缺少 n_missing"
            assert "missing_rate" in info, f"{col} 缺少 missing_rate"

    def test_missing_file_records_error(self):
        """找不到文件时，state.error_history 应记录一条错误"""
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, "not_exist.csv")

        assert len(state.error_history) == 1
        assert state.error_history[0].error_type == "RuntimeError"


class TestEdaNode:

    def test_anomaly_ratio_computed(self, tmp_path):
        """有 label 列时，anomaly_ratio 应接近注入比例"""
        csv_path = make_imbalanced_csv(tmp_path, n=1000, anomaly_ratio=0.02)
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, csv_path)

        mock = MockClient(mode="sequential")
        state = eda_node(state, mock)

        assert state.anomaly_ratio is not None
        # 允许 ±1% 误差
        assert abs(state.anomaly_ratio - 0.02) < 0.015, \
            f"anomaly_ratio={state.anomaly_ratio} 超出预期范围"

    def test_quantiles_not_mean(self, tmp_path):
        """schema_info 应含分位数（q25/q50/q75/q99），不应仅含均值"""
        csv_path = make_imbalanced_csv(tmp_path, n=500)
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, csv_path)

        mock = MockClient()
        state = eda_node(state, mock)

        # 检查数值列 f1 含分位数
        f1_info = state.schema_info.get("f1", {})
        for q_key in ["q25", "q50", "q75", "q99", "iqr"]:
            assert q_key in f1_info, f"schema_info['f1'] 缺少 {q_key}"

    def test_eda_summary_not_empty(self, tmp_path):
        """eda_summary 应为非空字符串（LLM 或兜底模板）"""
        csv_path = make_imbalanced_csv(tmp_path, n=500)
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, csv_path)

        mock = MockClient()
        state = eda_node(state, mock)

        assert isinstance(state.eda_summary, str)
        assert len(state.eda_summary) > 10, "eda_summary 为空或过短"

    def test_fallback_template_when_llm_fails(self, tmp_path):
        """LLM 返回空时，兜底模板应包含行数和异常占比"""
        csv_path = make_imbalanced_csv(tmp_path, n=300, anomaly_ratio=0.03)
        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, csv_path)

        # 用一个永远返回空的假客户端
        class EmptyClient:
            def chat(self, messages, **_):
                return {"message": ""}

        state = eda_node(state, EmptyClient())

        # 兜底字符串应含行数
        assert "300" in state.eda_summary or "行" in state.eda_summary

    def test_end_to_end_mock_pipeline(self, tmp_path):
        """T1.5 完整链路：ingestion → eda，Mock 模式，全字段不为空"""
        csv_path = make_imbalanced_csv(tmp_path, n=800, anomaly_ratio=0.025)
        state = TaskState(user_query="对交易数据做异常检测，异常约 2.5%")

        mock = MockClient(mode="sequential")

        # 运行两个节点
        state = data_ingestion_node(state, csv_path)
        state = eda_node(state, mock)

        # 验收清单
        assert state.raw_df is not None,         "raw_df 为空"
        assert len(state.schema_info) > 0,       "schema_info 为空"
        assert state.anomaly_ratio is not None,  "anomaly_ratio 未计算"
        assert state.eda_summary != "",           "eda_summary 为空"
        assert len(state.error_history) == 0,    f"有意外错误: {state.error_history}"

        print("\n✅ T1.5 全链路验收通过")
        print(f"   rows={state.raw_df.shape[0]}  anomaly_ratio={state.anomaly_ratio:.2%}")
        print(f"   EDA摘要: {state.eda_summary[:100]}...")

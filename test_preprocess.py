# test_preprocess.py
"""
T1.6 Mock 模式验证
运行：python -m pytest test_preprocess.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, RobustScaler

from factory.state import TaskState
from factory.nodes.ingestion import data_ingestion_node
from factory.nodes.preprocess import preprocess_node


# ─── 工具函数 ─────────────────────────────────────────────────
def make_csv(tmp_path, n=400, anomaly_ratio=0.02,
             add_cat=False, add_missing=False) -> str:
    """生成合成数据，可控制：类别列、缺失值、异常占比"""
    np.random.seed(0)
    n_anom = max(1, int(n * anomaly_ratio))
    n_norm = n - n_anom

    X_norm = np.random.randn(n_norm, 3) * [1, 2, 0.5]
    X_anom = np.random.randn(n_anom, 3) * 5 + 8
    X = np.vstack([X_norm, X_anom])
    y = [0] * n_norm + [1] * n_anom

    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["label"] = y

    if add_cat:
        df["device_type"] = np.random.choice(["A", "B", "C"], n)

    if add_missing:
        # 随机把 5% 的 f1 设为 NaN
        mask = np.random.rand(n) < 0.05
        df.loc[mask, "f1"] = np.nan

    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    return path


def _load(tmp_path, **kw) -> TaskState:
    """快捷：生成 CSV → 跑 ingestion_node → 返回 state"""
    path = make_csv(tmp_path, **kw)
    state = TaskState(user_query="测试")
    return data_ingestion_node(state, path)


# ─── 测试 ─────────────────────────────────────────────────────

class TestLabelSeparation:

    def test_y_true_separated_when_label_exists(self, tmp_path):
        """有 label 列时，y_true 应分离出来，X_processed 不含 label"""
        state = preprocess_node(_load(tmp_path), algorithm="IsolationForest")

        assert state.y_true is not None
        assert 1 in state.y_true             # 有异常样本
        assert "label" not in state.feature_names

    def test_y_true_is_none_without_label(self, tmp_path):
        """无 label 列时，y_true 应为 None（无监督路径）"""
        path = make_csv(tmp_path)
        # 删掉 label 列
        df = pd.read_csv(path).drop(columns=["label"])
        path2 = str(tmp_path / "no_label.csv")
        df.to_csv(path2, index=False)

        state = TaskState(user_query="测试")
        state = data_ingestion_node(state, path2)
        state = preprocess_node(state, algorithm="IsolationForest")

        assert state.y_true is None


class TestScalingBranch:

    def test_iforest_no_scaler(self, tmp_path):
        """IsolationForest → scaler 应为 None（树模型不需要标准化）"""
        state = preprocess_node(_load(tmp_path), algorithm="IsolationForest")
        assert state.scaler is None
        assert state.preprocessing_info["scaling_applied"] is False

    def test_lof_has_scaler(self, tmp_path):
        """LocalOutlierFactor → 必须有 fitted scaler"""
        state = preprocess_node(_load(tmp_path), algorithm="LocalOutlierFactor")
        assert state.scaler is not None
        assert isinstance(state.scaler, (StandardScaler, RobustScaler))
        assert state.preprocessing_info["scaling_applied"] is True

    def test_ocsvm_has_scaler(self, tmp_path):
        """OneClassSVM → 必须有 fitted scaler"""
        state = preprocess_node(_load(tmp_path), algorithm="OneClassSVM")
        assert state.scaler is not None

    def test_same_data_two_branches(self, tmp_path):
        """同一份数据，标准化/未标准化两条路径产出不同的 X_processed"""
        state_tree = preprocess_node(_load(tmp_path), algorithm="IsolationForest")
        state_dist = preprocess_node(_load(tmp_path), algorithm="LocalOutlierFactor")

        X_tree = state_tree.X_processed
        X_dist = state_dist.X_processed

        # 树模型版本的 X 与原始同量纲；距离模型版本均值应接近 0
        assert not np.allclose(X_tree, X_dist), "两条路径结果不应相同"
        col_means = X_dist.mean(axis=0)
        assert np.all(np.abs(col_means) < 0.1), \
            f"StandardScaler 后各列均值应≈0，实际={col_means}"


class TestMissingImputation:

    def test_no_nan_after_imputation(self, tmp_path):
        """填充后 X_processed 不应含任何 NaN"""
        state = preprocess_node(
            _load(tmp_path, add_missing=True),
            algorithm="IsolationForest"
        )
        assert not np.isnan(state.X_processed).any(), "仍有 NaN 未被填充"

    def test_median_not_mean_used(self, tmp_path):
        """填充值应为中位数（preprocessing_info 有记录）"""
        state = preprocess_node(
            _load(tmp_path, add_missing=True),
            algorithm="IsolationForest"
        )
        # 有缺失的列（f1）应被记录
        medians = state.preprocessing_info.get("num_medians_used", {})
        assert "f1" in medians, "f1 有缺失但未记录中位数填充"


class TestCategoricalEncoding:

    def test_cat_col_encoded_to_dummies(self, tmp_path):
        """类别列 device_type 应被 One-Hot 展开，不再是字符串"""
        state = preprocess_node(
            _load(tmp_path, add_cat=True),
            algorithm="IsolationForest"
        )
        # feature_names 应含 device_type_A / _B / _C 之类
        cat_encoded = [c for c in state.feature_names if "device_type" in c]
        assert len(cat_encoded) >= 2, \
            f"类别列未被正确 One-Hot：{state.feature_names}"

    def test_X_processed_all_numeric(self, tmp_path):
        """编码后 X_processed 应为纯数值矩阵"""
        state = preprocess_node(
            _load(tmp_path, add_cat=True),
            algorithm="LocalOutlierFactor"
        )
        assert state.X_processed.dtype in [np.float32, np.float64]


class TestImbalancePreserved:

    def test_anomaly_ratio_unchanged(self, tmp_path):
        """预处理不应改变正负样本比例（严禁过采样）"""
        ratio = 0.02
        state = preprocess_node(
            _load(tmp_path, n=500, anomaly_ratio=ratio),
            algorithm="IsolationForest"
        )
        actual_ratio = (state.y_true == 1).sum() / len(state.y_true)
        assert abs(actual_ratio - ratio) < 0.005, \
            f"异常比例被篡改：期望≈{ratio:.2%}，实际={actual_ratio:.2%}"

    def test_row_count_unchanged(self, tmp_path):
        """预处理不应增删行数（严禁 SMOTE）"""
        state = _load(tmp_path, n=300)
        n_before = len(state.raw_df)
        state = preprocess_node(state, algorithm="IsolationForest")
        n_after = len(state.X_processed)
        assert n_before == n_after, f"行数变化：{n_before} → {n_after}"


class TestEndToEnd:

    def test_full_pipeline_t15_t16(self, tmp_path):
        """T1.5 → T1.6 完整链路，Mock 模式，关键字段全不为空"""
        from factory.llm.mock_client import MockClient
        from factory.nodes.ingestion import eda_node

        path = make_csv(tmp_path, n=600, anomaly_ratio=0.025,
                        add_cat=True, add_missing=True)
        state = TaskState(user_query="传感器异常检测")

        # T1.5
        state = data_ingestion_node(state, path)
        state = eda_node(state, MockClient())

        # T1.6（LOF 需要标准化）
        state = preprocess_node(state, algorithm="LocalOutlierFactor")

        # 验收
        assert state.X_processed is not None,   "X_processed 为空"
        assert state.y_true is not None,        "y_true 为空"
        assert state.scaler is not None,        "LOF 路径 scaler 为空"
        assert len(state.feature_names) > 0,   "feature_names 为空"
        assert not np.isnan(state.X_processed).any(), "X_processed 含 NaN"
        assert len(state.error_history) == 0,  f"有错误: {state.error_history}"

        print("\n✅ T1.5 → T1.6 全链路验收通过")
        print(f"   X shape={state.X_processed.shape}")
        print(f"   异常占比={( state.y_true==1).sum()/len(state.y_true):.2%}")
        print(f"   Scaler={state.preprocessing_info['scaler_type']}")
        print(f"   特征列={state.feature_names}")

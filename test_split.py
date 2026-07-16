# test_split.py
"""
T1.7 验证脚本
运行：python -m pytest test_split.py -v
"""

import numpy as np
import pandas as pd
import pytest

from factory.state import TaskState
from factory.nodes.ingestion import data_ingestion_node
from factory.nodes.preprocess import preprocess_node
from factory.nodes.split import split_node, make_synthetic_dataset


# ─── 工具：用合成数据快速建好 T1.5+T1.6 后的 state ─────────────────
def _state_after_preprocess(
    tmp_path,
    n=800,
    anomaly_ratio=0.025,
    algorithm="IsolationForest",
    has_label=True,
) -> TaskState:
    df = make_synthetic_dataset(n, anomaly_ratio=anomaly_ratio)
    if not has_label:
        df = df.drop(columns=["label"])
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)

    state = TaskState(user_query="测试")
    state = data_ingestion_node(state, path)
    state = preprocess_node(state, algorithm=algorithm)
    return state


# ══════════════════════════════════════════════════════
# 一、split_node 测试
# ══════════════════════════════════════════════════════

class TestSplitNode:

    def test_stratify_keeps_anomaly_ratio(self, tmp_path):
        """分层切分后，训练集和测试集的异常比例应一致（允许 ±0.5%）"""
        state = _state_after_preprocess(tmp_path, n=1000, anomaly_ratio=0.03)
        state = split_node(state, test_size=0.2)

        train_ratio = state.split_info["train_anomaly_ratio"]
        test_ratio  = state.split_info["test_anomaly_ratio"]
        drift       = state.split_info["ratio_drift"]

        assert drift < 0.005, (
            f"训练集异常比={train_ratio:.3f} vs 测试集={test_ratio:.3f}，"
            f"漂移={drift:.4f}，超过 0.5% 阈值"
        )

    def test_test_set_has_anomalies(self, tmp_path):
        """测试集必须有异常，否则 Recall 无法计算"""
        state = _state_after_preprocess(tmp_path, n=1000, anomaly_ratio=0.03)
        state = split_node(state, test_size=0.2)

        n_test_anom = state.split_info["n_test_anomaly"]
        assert n_test_anom > 0, "测试集 0 异常，Recall 无法计算"

    def test_row_count_adds_up(self, tmp_path):
        """训练集 + 测试集行数 = 总行数"""
        state = _state_after_preprocess(tmp_path, n=800)
        n_before = len(state.X_processed)
        state = split_node(state, test_size=0.2)

        assert state.split_info["n_train"] + state.split_info["n_test"] == n_before

    def test_no_label_skips_split(self, tmp_path):
        """无标签时，X_train = 全量数据，X_test 为 None"""
        state = _state_after_preprocess(tmp_path, n=500, has_label=False)
        state = split_node(state)

        assert state.X_test is None,  "无标签时 X_test 应为 None"
        assert state.y_test is None,  "无标签时 y_test 应为 None"
        assert len(state.X_train) == 500
        assert state.split_info["mode"] == "unsupervised_no_split"

    def test_scaler_transform_not_refit_on_test(self, tmp_path):
        """LOF 路径：测试集数据来自 T1.6 已缩放的 X_processed，
        split_node 只做切片，不对测试集再次 fit scaler"""
        state = _state_after_preprocess(
            tmp_path, n=600, algorithm="LocalOutlierFactor"
        )
        scaler_before = state.scaler  # T1.6 的 scaler

        state = split_node(state, test_size=0.2)

        # split_node 不应替换 scaler
        assert state.scaler is scaler_before, "split_node 不应重新 fit scaler"

    def test_reproducible_with_same_seed(self, tmp_path):
        """相同 random_state 两次切分结果完全一致"""
        s1 = _state_after_preprocess(tmp_path, n=500)
        s2 = _state_after_preprocess(tmp_path, n=500)

        s1 = split_node(s1, random_state=42)
        s2 = split_node(s2, random_state=42)

        np.testing.assert_array_equal(
            s1.y_test, s2.y_test,
            err_msg="相同种子切分结果不一致"
        )

    def test_error_recorded_when_X_processed_none(self):
        """X_processed 为空时应记录错误，不崩溃"""
        state = TaskState(user_query="测试")
        state = split_node(state)

        assert len(state.error_history) == 1
        assert "X_processed" in state.error_history[0].error_message


# ══════════════════════════════════════════════════════
# 二、make_synthetic_dataset 测试
# ══════════════════════════════════════════════════════

class TestMakeSyntheticDataset:

    def test_basic_shape(self):
        """输出 DataFrame 行列数正确"""
        df = make_synthetic_dataset(n_samples=500, n_features=6, anomaly_ratio=0.02)
        assert df.shape == (500, 7)  # 6 特征 + 1 label

    def test_anomaly_ratio_approximate(self):
        """实际异常占比应在目标值 ±0.5% 以内"""
        target = 0.02
        df = make_synthetic_dataset(n_samples=2000, anomaly_ratio=target)
        actual = df["label"].mean()
        assert abs(actual - target) < 0.005, f"实际比例 {actual:.3f} 偏离目标 {target}"

    def test_label_is_binary(self):
        """label 列只有 0 和 1"""
        df = make_synthetic_dataset(n_samples=500)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_reproducible(self):
        """相同种子两次生成结果完全一致"""
        df1 = make_synthetic_dataset(n_samples=300, random_state=7)
        df2 = make_synthetic_dataset(n_samples=300, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """不同种子结果不同"""
        df1 = make_synthetic_dataset(n_samples=300, random_state=1)
        df2 = make_synthetic_dataset(n_samples=300, random_state=2)
        assert not df1.equals(df2)

    def test_ratio_too_high_raises(self):
        """anomaly_ratio > 0.15 应抛出 ValueError"""
        with pytest.raises(ValueError, match="超出合理范围"):
            make_synthetic_dataset(anomaly_ratio=0.20)

    def test_ratio_too_low_raises(self):
        """anomaly_ratio < 0.005 应抛出 ValueError"""
        with pytest.raises(ValueError, match="超出合理范围"):
            make_synthetic_dataset(anomaly_ratio=0.001)

    def test_save_to_csv(self, tmp_path):
        """指定 save_path 后文件应被写入"""
        path = str(tmp_path / "synthetic.csv")
        make_synthetic_dataset(n_samples=200, save_path=path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 200

    def test_anomalies_are_outliers(self):
        """异常样本的 L2 范数均值应显著大于正常样本（确实是离群点）"""
        df = make_synthetic_dataset(n_samples=2000, anomaly_ratio=0.05)
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values

        norm_normal = np.linalg.norm(X[df["label"] == 0], axis=1).mean()
        norm_anomal = np.linalg.norm(X[df["label"] == 1], axis=1).mean()

        assert norm_anomal > norm_normal * 1.5, (
            f"异常样本范数 {norm_anomal:.2f} 应显著大于正常样本 {norm_normal:.2f}"
        )


# ══════════════════════════════════════════════════════
# 三、T1.5 → T1.6 → T1.7 全链路
# ══════════════════════════════════════════════════════

class TestFullPipeline:

    def test_t15_t16_t17_end_to_end(self, tmp_path):
        """合成数据 → ingestion → preprocess → split，全字段验收"""
        from factory.llm.mock_client import MockClient
        from factory.nodes.ingestion import eda_node

        # 生成合成数据
        df = make_synthetic_dataset(n_samples=1000, n_features=5, anomaly_ratio=0.025)
        path = str(tmp_path / "end2end.csv")
        df.to_csv(path, index=False)

        state = TaskState(user_query="工厂传感器异常检测，极不平衡")

        # T1.5
        state = data_ingestion_node(state, path)
        state = eda_node(state, MockClient())

        # T1.6（LOF 需要标准化）
        state = preprocess_node(state, algorithm="LocalOutlierFactor")

        # T1.7
        state = split_node(state, test_size=0.2)

        # ── 验收清单 ──────────────────────────────────
        assert state.X_train is not None,          "X_train 为空"
        assert state.X_test  is not None,          "X_test 为空"
        assert state.y_test  is not None,          "y_test 为空"
        assert state.split_info["n_test_anomaly"] > 0, "测试集无异常"
        assert state.split_info["ratio_drift"] < 0.005, \
            f"异常比例漂移过大: {state.split_info['ratio_drift']}"
        assert len(state.error_history) == 0,      f"有错误: {state.error_history}"

        # 特征维度一致
        assert state.X_train.shape[1] == state.X_test.shape[1], \
            "训练集与测试集特征维度不匹配"

        print("\n✅ T1.5 → T1.6 → T1.7 全链路验收通过")
        info = state.split_info
        print(f"   训练集: {info['n_train']} 行，异常 {info['n_train_anomaly']} 个（{info['train_anomaly_ratio']:.2%}）")
        print(f"   测试集: {info['n_test']} 行，异常 {info['n_test_anomaly']} 个（{info['test_anomaly_ratio']:.2%}）")
        print(f"   比例漂移: {info['ratio_drift']:.4f}")

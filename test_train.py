# test_train.py
"""
T1.8 验证脚本
运行：python -m pytest test_train.py -v
"""

import numpy as np
import pytest
from factory.state import TaskState
from factory.nodes.split import make_synthetic_dataset
from factory.nodes.ingestion import data_ingestion_node
from factory.nodes.preprocess import preprocess_node
from factory.nodes.split import split_node
from factory.nodes.train import train_node, _zscore_baseline


# ─── 工具：一键建好 T1.5→T1.7 完成后的 state ──────────────────────────
def _state_ready_to_train(
    tmp_path,
    n=800,
    anomaly_ratio=0.025,
    algorithm="IsolationForest",
) -> TaskState:
    df = make_synthetic_dataset(n, anomaly_ratio=anomaly_ratio)
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)

    state = TaskState(user_query="测试")
    state = data_ingestion_node(state, path)
    state = preprocess_node(state, algorithm=algorithm)
    state = split_node(state, test_size=0.2)
    return state


# ══════════════════════════════════════════════════════
# 一、基本产出验证
# ══════════════════════════════════════════════════════

class TestTrainNodeOutputs:

    def test_anomaly_scores_length_matches_test_set(self, tmp_path):
        """anomaly_scores 长度应 = X_test 行数（对测试集打分）"""
        state = _state_ready_to_train(tmp_path)
        n_test = len(state.X_test)
        state = train_node(state)

        assert len(state.anomaly_scores) == n_test, (
            f"scores 长度 {len(state.anomaly_scores)} ≠ X_test 行数 {n_test}"
        )

    def test_y_pred_length_matches_test_set(self, tmp_path):
        """y_pred 长度应 = X_test 行数"""
        state = _state_ready_to_train(tmp_path)
        n_test = len(state.X_test)
        state = train_node(state)

        assert len(state.y_pred) == n_test

    def test_y_pred_only_contains_0_and_1(self, tmp_path):
        """y_pred 必须只含 0 和 1（已完成 -1/1 → 0/1 映射）"""
        state = train_node(_state_ready_to_train(tmp_path))
        unique_vals = set(state.y_pred)
        assert unique_vals.issubset({0, 1}), (
            f"y_pred 含非法值 {unique_vals}，"
            "可能忘记做 sklearn -1/1 → 0/1 映射"
        )

    def test_threshold_is_float(self, tmp_path):
        """threshold 应为浮点数"""
        state = train_node(_state_ready_to_train(tmp_path))
        assert isinstance(state.threshold, float)

    def test_n_anomalies_detected_consistent_with_y_pred(self, tmp_path):
        """n_anomalies_detected 应等于 y_pred 中 1 的个数"""
        state = train_node(_state_ready_to_train(tmp_path))
        assert state.n_anomalies_detected == sum(state.y_pred)

    def test_trained_model_not_none(self, tmp_path):
        """正常流程下 trained_model 不为 None"""
        state = train_node(_state_ready_to_train(tmp_path))
        assert state.trained_model is not None

    def test_train_info_has_required_keys(self, tmp_path):
        """train_info 必须包含关键字段"""
        state = train_node(_state_ready_to_train(tmp_path))
        required = {
            "algorithm", "contamination", "n_anomalies_detected",
            "threshold", "elapsed_sec", "used_fallback"
        }
        missing = required - set(state.train_info.keys())
        assert not missing, f"train_info 缺少字段: {missing}"


# ══════════════════════════════════════════════════════
# 二、标签映射（最关键的坑）
# ══════════════════════════════════════════════════════

class TestLabelMapping:

    def test_sklearn_minus1_mapped_to_1(self, tmp_path):
        """sklearn -1（异常）必须映射为 1，不能出现 -1 在 y_pred 中"""
        state = train_node(_state_ready_to_train(tmp_path))
        assert -1 not in state.y_pred, (
            "y_pred 中出现了 -1，说明 sklearn 原始输出未被映射！"
            "评估 Precision/Recall 时会完全错误。"
        )

    def test_scores_higher_means_more_anomalous(self, tmp_path):
        """异常分数越高应越可疑：y_pred=1 的样本平均分数 > y_pred=0"""
        state = train_node(_state_ready_to_train(tmp_path, n=1000))
        scores = np.array(state.anomaly_scores)
        y_pred = np.array(state.y_pred)

        if y_pred.sum() == 0 or (y_pred == 0).sum() == 0:
            pytest.skip("无法比较：检出全 0 或全 1")

        mean_anom   = scores[y_pred == 1].mean()
        mean_normal = scores[y_pred == 0].mean()
        assert mean_anom > mean_normal, (
            f"异常样本平均分 {mean_anom:.4f} 应 > 正常样本平均分 {mean_normal:.4f}，"
            "可能 decision_function 忘记取负"
        )

    def test_threshold_separates_classes(self, tmp_path):
        """scores > threshold 的样本应对应 y_pred=1"""
        state = train_node(_state_ready_to_train(tmp_path))
        scores   = np.array(state.anomaly_scores)
        y_pred   = np.array(state.y_pred)
        threshold = state.threshold

        above = (scores > threshold).astype(int)
        # 允许极小数值误差（浮点精度），一致性应 >99%
        agreement = (above == y_pred).mean()
        assert agreement > 0.99, (
            f"threshold 与 y_pred 一致率仅 {agreement:.2%}，"
            "scores 和 threshold 可能来自不同变换"
        )


# ══════════════════════════════════════════════════════
# 三、contamination 从状态读取
# ══════════════════════════════════════════════════════

class TestContaminationFromState:

    def test_uses_anomaly_ratio_when_available(self, tmp_path):
        """state.anomaly_ratio 存在时，应优先于 state.contamination"""
        state = _state_ready_to_train(tmp_path)
        state.anomaly_ratio = 0.03   # 覆盖实测值
        state.contamination = 0.10   # 这个不该被用

        state = train_node(state)
        assert abs(state.train_info["contamination"] - 0.03) < 1e-6

    def test_falls_back_to_contamination_when_ratio_none(self, tmp_path):
        """anomaly_ratio 为 None 时，使用 state.contamination"""
        state = _state_ready_to_train(tmp_path)
        state.anomaly_ratio = None
        state.contamination = 0.04

        state = train_node(state)
        assert abs(state.train_info["contamination"] - 0.04) < 1e-6

    def test_contamination_clipped_to_valid_range(self, tmp_path):
        """contamination 超出 (0, 0.5] 时应被截断，不报错"""
        state = _state_ready_to_train(tmp_path)
        state.anomaly_ratio = 0.99   # 非法值

        state = train_node(state)   # 不应崩溃
        assert state.train_info["contamination"] <= 0.5


# ══════════════════════════════════════════════════════
# 四、无标签路径（对 X_train 打分）
# ══════════════════════════════════════════════════════

class TestNoLabelPath:

    def test_scores_on_train_when_no_test(self, tmp_path):
        """无标签时（X_test=None），应对 X_train 打分"""
        df = make_synthetic_dataset(500)
        df = df.drop(columns=["label"])   # 去掉标签
        path = str(tmp_path / "no_label.csv")
        df.to_csv(path, index=False)

        state = TaskState(user_query="无标签测试")
        state = data_ingestion_node(state, path)
        state = preprocess_node(state)
        state = split_node(state)         # 无标签路径，X_test=None
        state = train_node(state)

        assert len(state.anomaly_scores) == len(state.X_train)
        assert state.train_info["scored_on"] == "X_train（无标签路径）"


# ══════════════════════════════════════════════════════
# 五、z-score 兜底基线
# ══════════════════════════════════════════════════════

class TestZscoreBaseline:

    def test_baseline_returns_valid_scores(self):
        """z-score 基线应返回正确形状的分数"""
        np.random.seed(0)
        X = np.random.randn(200, 5)
        X[0] = [10, 10, 10, 10, 10]   # 注入明显异常

        scores, y_pred, threshold = _zscore_baseline(X)

        assert len(scores) == 200
        assert len(y_pred) == 200
        assert threshold == 3.0

    def test_baseline_detects_obvious_outlier(self):
        """明显离群点（|z|>10）应被基线检测到"""
        np.random.seed(1)
        X = np.random.randn(300, 4)
        X[0] = [20, 20, 20, 20]   # 超级离群点

        scores, y_pred, _ = _zscore_baseline(X)
        assert y_pred[0] == 1, "明显离群点未被 z-score 基线检测到"

    def test_baseline_no_nan_output(self):
        """输出不应含 NaN（防止后续评估崩溃）"""
        X = np.random.randn(100, 3)
        scores, y_pred, _ = _zscore_baseline(X)
        assert not np.isnan(scores).any()

    def test_baseline_handles_zero_std_column(self):
        """某列方差为 0（常数列）时不应除零报错"""
        X = np.ones((100, 3))   # 全是常数，std=0
        X[:, 0] = np.random.randn(100)

        scores, y_pred, _ = _zscore_baseline(X)   # 不崩溃即通过
        assert len(scores) == 100


# ══════════════════════════════════════════════════════
# 六、T1.5 → T1.8 全链路
# ══════════════════════════════════════════════════════

class TestFullPipeline:

    def test_t15_to_t18_end_to_end(self, tmp_path):
        """合成数据跑完 T1.5→T1.8 全链路，关键字段验收"""
        from factory.llm.mock_client import MockClient
        from factory.nodes.ingestion import eda_node

        df = make_synthetic_dataset(1000, n_features=6, anomaly_ratio=0.025)
        path = str(tmp_path / "e2e.csv")
        df.to_csv(path, index=False)

        state = TaskState(user_query="端到端测试")

        state = data_ingestion_node(state, path)
        state = eda_node(state, MockClient())
        state = preprocess_node(state, algorithm="IsolationForest")
        state = split_node(state, test_size=0.2)
        state = train_node(state)

        # 验收清单
        assert len(state.anomaly_scores) > 0,       "anomaly_scores 为空"
        assert len(state.y_pred) > 0,               "y_pred 为空"
        assert -1 not in state.y_pred,              "y_pred 含 -1，标签未映射"
        assert state.threshold is not None,         "threshold 为空"
        assert state.n_anomalies_detected >= 0,     "n_anomalies_detected 异常"
        assert state.trained_model is not None,     "trained_model 为空"
        assert not state.train_info["used_fallback"], "不应走兜底路径"
        assert len(state.error_history) == 0,       f"有错误: {state.error_history}"

        print("\n✅ T1.5 → T1.8 全链路验收通过")
        info = state.train_info
        print(f"   算法: {info['algorithm']}")
        print(f"   contamination: {info['contamination']:.4f}")
        print(f"   检出异常: {info['n_anomalies_detected']}/{info['n_score_samples']}"
              f"  ({info['detection_rate']:.2%})")
        print(f"   threshold: {info['threshold']:.4f}")
        print(f"   耗时: {info['elapsed_sec']}s")

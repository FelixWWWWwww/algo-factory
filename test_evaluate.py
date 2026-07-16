# test_evaluate.py
"""
T1.9 验证脚本
运行：python -m pytest test_evaluate.py -v
"""

import numpy as np
import pytest
from factory.state import TaskState
from factory.nodes.split import make_synthetic_dataset
from factory.nodes.ingestion import data_ingestion_node
from factory.nodes.preprocess import preprocess_node
from factory.nodes.split import split_node
from factory.nodes.train import train_node
from factory.nodes.evaluate import evaluate_node


# ─── 工具：快速建好 T1.5→T1.8 后的 state ─────────────────────────────
def _state_ready_to_eval(
    tmp_path,
    n=1000,
    anomaly_ratio=0.03,
    algorithm="IsolationForest",
) -> TaskState:
    df = make_synthetic_dataset(n, anomaly_ratio=anomaly_ratio, random_state=0)
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)

    state = TaskState(user_query="测试")
    state = data_ingestion_node(state, path)
    state = preprocess_node(state, algorithm=algorithm)
    state = split_node(state, test_size=0.2, random_state=0)
    state = train_node(state, random_state=0)
    return state


# ══════════════════════════════════════════════════════════
# 一、有标签路径：指标合法性
# ══════════════════════════════════════════════════════════

class TestLabeledPath:

    def test_pr_auc_in_range(self, tmp_path):
        """PR-AUC 应在 [0, 1] 范围内"""
        state = evaluate_node(_state_ready_to_eval(tmp_path))
        pr_auc = state.eval_metrics["pr_auc"]
        assert 0.0 <= pr_auc <= 1.0, f"PR-AUC={pr_auc} 超出 [0,1]"

    def test_all_core_metrics_present(self, tmp_path):
        """eval_metrics 必须含 pr_auc / f1 / recall / precision"""
        state = evaluate_node(_state_ready_to_eval(tmp_path))
        for key in ["pr_auc", "f1", "recall", "precision"]:
            assert key in state.eval_metrics, f"缺少指标: {key}"

    def test_accuracy_marked_unreliable(self, tmp_path):
        """accuracy 应以 __UNRELIABLE 后缀存在，而不是裸 accuracy"""
        state = evaluate_node(_state_ready_to_eval(tmp_path))
        assert "accuracy" not in state.eval_metrics, \
            "accuracy 不应裸露出现，会被误用于选优"
        assert "accuracy__UNRELIABLE" in state.eval_metrics

    def test_accuracy_not_in_final_metrics(self, tmp_path):
        """final_metrics（选优依据）不应含 accuracy"""
        state = evaluate_node(_state_ready_to_eval(tmp_path))
        assert "accuracy" not in state.final_metrics
        assert "accuracy__UNRELIABLE" not in state.final_metrics

    def test_final_metrics_has_pr_auc(self, tmp_path):
        """final_metrics 中 pr_auc 应存在且为主指标"""
        state = evaluate_node(_state_ready_to_eval(tmp_path))
        assert "pr_auc" in state.final_metrics

    def test_topk_indices_produced(self, tmp_path):
        """有标签时也应产出 topk_indices（供可解释性模块使用）"""
        state = evaluate_node(_state_ready_to_eval(tmp_path), topk=10)
        assert len(state.topk_indices) == 10

    def test_topk_are_highest_scoring(self, tmp_path):
        """topk_indices 对应的分数应是 anomaly_scores 中最高的"""
        state = evaluate_node(_state_ready_to_eval(tmp_path), topk=5)
        scores = np.array(state.anomaly_scores)
        topk_scores = scores[state.topk_indices]
        min_topk = topk_scores.min()
        non_topk_scores = np.delete(scores, state.topk_indices)
        assert min_topk >= non_topk_scores.max() - 1e-9, \
            "topk_indices 不是得分最高的样本"


# ══════════════════════════════════════════════════════════
# 二、PR-AUC vs ROC-AUC 对比（核心原理验证）
# ══════════════════════════════════════════════════════════

class TestPrAucVsRocAuc:

    def test_waste_model_exposed_by_pr_auc(self):
        """废模型（全判正常）：accuracy 高，PR-AUC 低——验证 PR-AUC 的区分力"""
        from sklearn.metrics import average_precision_score, accuracy_score

        # 极不平衡：998 正常，2 异常
        y_true  = np.array([0]*998 + [1]*2)
        # 废模型：全判正常（y_pred 全 0，scores 全一样低）
        scores_waste = np.zeros(1000)
        y_pred_waste = np.zeros(1000, dtype=int)

        acc_waste    = accuracy_score(y_true, y_pred_waste)
        pr_auc_waste = average_precision_score(y_true, scores_waste)

        assert acc_waste > 0.99,   f"废模型 accuracy 应 >99%，实际={acc_waste:.3f}"
        assert pr_auc_waste < 0.05, f"废模型 PR-AUC 应 <5%，实际={pr_auc_waste:.3f}"

    def test_good_model_pr_auc_higher_than_waste(self, tmp_path):
        """合成数据上，IForest 的 PR-AUC 应显著高于随机基线"""
        state = evaluate_node(_state_ready_to_eval(tmp_path, n=2000))
        y_test = np.array(state.y_test)

        # 随机基线 PR-AUC ≈ 正类占比（anomaly_ratio）
        baseline_pr_auc = y_test.mean()
        model_pr_auc    = state.eval_metrics["pr_auc"]

        assert model_pr_auc > baseline_pr_auc, (
            f"模型 PR-AUC={model_pr_auc:.4f} 应 > 随机基线 {baseline_pr_auc:.4f}"
        )


# ══════════════════════════════════════════════════════════
# 三、边界：测试集 0 异常
# ══════════════════════════════════════════════════════════

class TestZeroAnomalyEdgeCase:

    def test_zero_anomaly_no_crash(self, tmp_path):
        """测试集 0 异常时不应崩溃，应降级到 Top-K"""
        state = _state_ready_to_eval(tmp_path)

        # 强制把 y_test 全设成 0（模拟极端情况）
        state.y_test = [0] * len(state.y_test)

        # 不崩溃即通过
        state = evaluate_node(state)
        assert "warning" in state.eval_info or "note" in state.eval_metrics

    def test_zero_anomaly_topk_still_produced(self, tmp_path):
        """即使测试集 0 异常，Top-K 列表也应产出（供人工审阅）"""
        state = _state_ready_to_eval(tmp_path)
        state.y_test = [0] * len(state.y_test)
        state = evaluate_node(state, topk=10)

        assert len(state.topk_indices) > 0

    def test_zero_anomaly_no_pr_auc_in_final_metrics(self, tmp_path):
        """测试集 0 异常时，final_metrics 不应有 pr_auc（避免 None 参与比较）"""
        state = _state_ready_to_eval(tmp_path)
        state.y_test = [0] * len(state.y_test)
        state = evaluate_node(state)

        # final_metrics 应为空或不含 pr_auc
        assert state.final_metrics.get("pr_auc") is None \
            or "pr_auc" not in state.final_metrics


# ══════════════════════════════════════════════════════════
# 四、无标签路径
# ══════════════════════════════════════════════════════════

class TestUnlabeledPath:

    def test_unlabeled_produces_score_distribution(self, tmp_path):
        """无标签时应产出分数分布统计（分位数）"""
        state = _state_ready_to_eval(tmp_path)
        state.y_test = None   # 模拟无标签

        state = evaluate_node(state, topk=15)

        for key in ["score_p50", "score_p90", "score_p95", "score_p99"]:
            assert key in state.eval_metrics, f"缺少分布统计: {key}"

    def test_unlabeled_topk_count_correct(self, tmp_path):
        """无标签时 Top-K 数量应等于参数 topk"""
        state = _state_ready_to_eval(tmp_path)
        state.y_test = None
        state = evaluate_node(state, topk=20)

        assert len(state.topk_indices) == 20

    def test_unlabeled_final_metrics_empty(self, tmp_path):
        """无标签时 final_metrics 应为空（没有可靠指标）"""
        state = _state_ready_to_eval(tmp_path)
        state.y_test = None
        state = evaluate_node(state)

        assert state.final_metrics == {}

    def test_score_distribution_ordered(self, tmp_path):
        """分位数应单调递增：p50 ≤ p90 ≤ p95 ≤ p99"""
        state = _state_ready_to_eval(tmp_path)
        state.y_test = None
        state = evaluate_node(state)

        m = state.eval_metrics
        assert m["score_p50"] <= m["score_p90"] <= m["score_p95"] <= m["score_p99"]


# ══════════════════════════════════════════════════════════
# 五、T1.5 → T1.9 Day 1 全链路
# ══════════════════════════════════════════════════════════

class TestDay1FullPipeline:

    def test_t15_to_t19_mock_end_to_end(self, tmp_path):
        """Day 1 完整链路：5 个节点，Mock 模式，全字段验收"""
        from factory.llm.mock_client import MockClient
        from factory.nodes.ingestion import eda_node

        df = make_synthetic_dataset(
            n_samples=1200, n_features=6,
            anomaly_ratio=0.025, random_state=42
        )
        path = str(tmp_path / "day1.csv")
        df.to_csv(path, index=False)

        state = TaskState(user_query="Day 1 全链路端到端测试")

        # T1.5
        state = data_ingestion_node(state, path)
        state = eda_node(state, MockClient())
        # T1.6
        state = preprocess_node(state, algorithm="IsolationForest")
        # T1.7
        state = split_node(state, test_size=0.2, random_state=42)
        # T1.8
        state = train_node(state, random_state=42)
        # T1.9
        state = evaluate_node(state, topk=20)

        # ── 验收清单 ──────────────────────────────────────
        assert state.eval_metrics.get("pr_auc") is not None,  "pr_auc 为空"
        assert "accuracy" not in state.final_metrics,         "accuracy 混入 final_metrics"
        assert len(state.topk_indices) == 20,                 "topk_indices 数量错误"
        assert len(state.error_history) == 0,                 f"有错误: {state.error_history}"
        assert state.eval_info["path"] == "labeled",          "应走有标签路径"
        assert 0 <= state.eval_metrics["pr_auc"] <= 1,        "PR-AUC 超出范围"

        print("\n🏁 Day 1 全链路验收通过")
        m = state.eval_metrics
        print(f"   PR-AUC   = {m['pr_auc']:.4f}  ← 主指标")
        print(f"   F1       = {m['f1']:.4f}")
        print(f"   Recall   = {m['recall']:.4f}")
        print(f"   Precision= {m['precision']:.4f}")
        print(f"   Accuracy = {m['accuracy__UNRELIABLE']:.4f}  ← 不可靠，仅展示")
        print(f"   Top-20 最可疑行号: {state.topk_indices[:5]}...")
        print(f"   EDA摘要: {state.eda_summary[:60]}...")

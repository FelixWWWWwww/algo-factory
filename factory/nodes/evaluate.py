# factory/nodes/evaluate.py
"""
T1.9: 评估节点（核心难点）

两条路径：
  有标签 → 计算 PR-AUC（主）/ ROC-AUC / F1 / Recall / Precision
           accuracy 展示但标注"不可靠"，严禁用于选优
  无标签 → Top-K 最可疑样本列表 + 分数分布统计

边界处理：
  - 测试集 0 异常 → PR-AUC 无法计算，降级 Top-K + 记录警告
  - y_pred 全 0   → Precision 分母为零，zero_division=0 返回 0.0
  - PR-AUC 计算崩溃 → 回退固定阈值下的 F1

关键约定：
  - anomaly_scores 越大越可疑（T1.8 已取负）
  - y_test / y_pred 中 1=异常，0=正常（T1.8 已映射）
  - final_metrics 只写"可靠指标"（pr_auc / f1 / recall / precision）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)


def evaluate_node(
    state: "TaskState",
    topk: int = 20,
) -> "TaskState":
    """评估节点：根据有无标签走两条路径。

    Args:
        state: 全局状态（读 anomaly_scores / y_test / y_pred）
        topk:  无标签路径下，输出 Top-K 最可疑样本数量

    写入状态：
        state.eval_metrics   — 指标字典
        state.topk_indices   — Top-K 行索引（无标签时）
        state.eval_info      — 评估记录
        state.final_metrics  — 可靠指标子集（供 get_best_result 使用）
    """
    if not state.anomaly_scores:
        logger.error("[evaluate] anomaly_scores 为空，请先运行 train_node")
        state.add_error("evaluate", "RuntimeError", "anomaly_scores 为空")
        return state

    scores = np.array(state.anomaly_scores, dtype=float)

    # ── 路径分叉 ─────────────────────────────────────────────────────────
    if state.y_test is None:
        return _evaluate_unlabeled(state, scores, topk)
    else:
        return _evaluate_labeled(state, scores, topk)


# ══════════════════════════════════════════════════════════
# 路径 A：有标签
# ══════════════════════════════════════════════════════════

def _evaluate_labeled(state: "TaskState", scores: np.ndarray, topk: int) -> "TaskState":
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )

    y_test = np.array(state.y_test, dtype=int)
    y_pred = np.array(state.y_pred, dtype=int)

    n_pos = int(y_test.sum())       # 测试集中真实异常数
    n_total = len(y_test)
    info: dict = {
        "path":           "labeled",
        "n_test":         n_total,
        "n_test_anomaly": n_pos,
        "topk":           topk,
    }

    # ── 边界：测试集 0 异常 ──────────────────────────────────────────────
    if n_pos == 0:
        logger.warning(
            "[evaluate] ⚠️  测试集 0 个真实异常！"
            "Recall/PR-AUC 无法计算，降级为 Top-K 输出。"
            "建议：增大数据量或调高 anomaly_ratio（T1.7）。"
        )
        info["warning"] = "test_set_zero_anomaly"
        state.eval_info = info
        # 降级到 Top-K
        return _append_topk(state, scores, topk, info)

    # ── 主指标：PR-AUC ────────────────────────────────────────────────────
    # average_precision_score 对分数排序后逐阈值计算 Precision-Recall 曲线下面积
    # 与 ROC-AUC 不同，它对正类（异常）非常敏感
    try:
        pr_auc = float(average_precision_score(y_test, scores))
        info["pr_auc_method"] = "average_precision_score"
    except Exception as e:
        logger.warning(f"[evaluate] PR-AUC 计算失败，回退固定阈值 F1: {e}")
        pr_auc = float(f1_score(y_test, y_pred, zero_division=0))
        info["pr_auc_method"] = "fallback_threshold_f1"
        info["pr_auc_warning"] = str(e)

    # ── 辅助指标：ROC-AUC ────────────────────────────────────────────────
    try:
        roc_auc = float(roc_auc_score(y_test, scores))
    except Exception:
        roc_auc = None
        info["roc_auc_warning"] = "计算失败（可能测试集只有一个类别）"

    # ── 阈值处硬指标：Precision / Recall / F1 ───────────────────────────
    # 这些基于 y_pred（T1.8 按 contamination 切出的硬标签）
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))

    # ── accuracy：展示但明确标记为"不可靠"────────────────────────────────
    # 极不平衡时废模型（全判正常）accuracy 接近 1.0，毫无意义
    acc = float(accuracy_score(y_test, y_pred))

    # ── 写入 eval_metrics ────────────────────────────────────────────────
    state.eval_metrics = {
        # === 可靠指标（用于选优）===
        "pr_auc":    round(pr_auc, 4),
        "roc_auc":   round(roc_auc, 4) if roc_auc is not None else None,
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        # === 不可靠指标（仅供展示，不参与选优）===
        "accuracy__UNRELIABLE": round(acc, 4),
    }

    # final_metrics 只存可靠部分（get_best_result 和 T2.5 选优用这个）
    state.final_metrics = {
        "pr_auc":    state.eval_metrics["pr_auc"],
        "f1":        state.eval_metrics["f1"],
        "precision": state.eval_metrics["precision"],
        "recall":    state.eval_metrics["recall"],
    }
    if roc_auc is not None:
        state.final_metrics["roc_auc"] = state.eval_metrics["roc_auc"]

    # Top-K 也顺便算，T3.2 可解释性要用
    topk_idx = np.argsort(scores)[::-1][:topk]
    state.topk_indices = topk_idx.tolist()
    info["topk_scores"] = scores[topk_idx].tolist()

    info["pr_auc_note"] = (
        "主指标。不平衡场景下比 ROC-AUC 更有区分力，"
        "对误报（FP）极度敏感。"
    )
    info["accuracy_note"] = (
        "⚠️ 不可靠：极不平衡时全判正常可得近 100% accuracy，"
        "严禁用于模型选优。"
    )
    state.eval_info = info

    logger.info(
        f"[evaluate] 完成（有标签）。"
        f"PR-AUC={pr_auc:.4f}  "
        f"F1={f1:.4f}  "
        f"Recall={recall:.4f}  "
        f"Precision={precision:.4f}  "
        f"Accuracy={acc:.4f}（不可靠）"
    )
    return state


# ══════════════════════════════════════════════════════════
# 路径 B：无标签（Top-K 人工审阅）
# ══════════════════════════════════════════════════════════

def _evaluate_unlabeled(state: "TaskState", scores: np.ndarray, topk: int) -> "TaskState":
    """无标签路径：输出异常分数分布统计 + Top-K 最可疑样本行号。"""
    info = {
        "path":  "unlabeled",
        "note":  "无标签，无法计算 PR-AUC/F1。评估退化为 Top-K 人工审阅。",
        "topk":  topk,
    }
    return _append_topk(state, scores, topk, info)


def _append_topk(state: "TaskState", scores: np.ndarray, topk: int, info: dict) -> "TaskState":
    """计算 Top-K 和分数分布，写入 state。"""
    topk_actual = min(topk, len(scores))
    topk_idx = np.argsort(scores)[::-1][:topk_actual]

    state.topk_indices = topk_idx.tolist()

    # 分数分布（用分位数，不用均值——原因同 EDA 节点）
    state.eval_metrics = {
        "note":          info.get("note", "Top-K 模式"),
        "topk":          topk_actual,
        "score_p50":     round(float(np.percentile(scores, 50)), 4),
        "score_p90":     round(float(np.percentile(scores, 90)), 4),
        "score_p95":     round(float(np.percentile(scores, 95)), 4),
        "score_p99":     round(float(np.percentile(scores, 99)), 4),
        "score_max":     round(float(scores.max()), 4),
        "topk_min_score": round(float(scores[topk_idx].min()), 4),
    }
    state.final_metrics = {}   # 无标签时无可靠指标
    state.eval_info = info

    logger.info(
        f"[evaluate] 完成（无标签/零异常）。"
        f"Top-{topk_actual} 分数范围: "
        f"[{scores[topk_idx].min():.4f}, {scores[topk_idx].max():.4f}]"
    )
    return state

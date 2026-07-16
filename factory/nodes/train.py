# factory/nodes/train.py
"""
T1.8: 单模型训练节点

职责：
  1. 从 state.contamination（或 state.anomaly_ratio）读取异常比例
  2. 在 X_train 上 fit Isolation Forest（无监督）
  3. 对 X_test（或无切分时对 X_train）打分 → anomaly_scores
  4. 产出映射后的硬标签 y_pred（1=异常/0=正常）
  5. 记录 threshold 与 n_anomalies_detected
  6. fit 失败时回退 z-score / IQR 统计基线

关键约定：
  - sklearn IForest/LOF 的 predict() 返回 -1=异常 / 1=正常
    评估前必须映射：y_pred = (raw == -1).astype(int)
  - decision_function() 越小越异常，取负使"越大越可疑"
    anomaly_scores = -model.decision_function(X_test)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import IsolationForest

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)


def train_node(
    state: "TaskState",
    algorithm: str = "IsolationForest",
    n_estimators: int = 100,
    random_state: int = 42,
) -> "TaskState":
    """单模型训练节点：fit → score → 映射硬标签。

    Args:
        state:        全局状态（读 X_train/X_test，写 anomaly_scores/y_pred/...）
        algorithm:    当前仅支持 "IsolationForest"（T2.4 扩展多算法）
        n_estimators: IForest 树的数量
        random_state: 随机种子

    写入状态：
        state.trained_model        — fitted 模型
        state.anomaly_scores       — 测试集异常分数（越大越可疑）
        state.y_pred               — 测试集硬标签（1=异常/0=正常，已映射）
        state.threshold            — 判定阈值
        state.n_anomalies_detected — 检出异常数
        state.train_info           — 训练记录字典
    """
    if state.X_train is None:
        logger.error("[train] X_train 为空，请先运行 split_node")
        state.add_error("train", "RuntimeError", "X_train 为空")
        return state

    # ── 从状态读取 contamination ────────────────────────────────────────
    # 优先用 EDA 实测值（更准确），其次用全局设定值
    contamination = (
        state.anomaly_ratio
        if state.anomaly_ratio is not None
        else state.contamination
    )
    # sklearn 要求 contamination ∈ (0, 0.5]
    contamination = float(np.clip(contamination, 1e-4, 0.5))
    logger.info(f"[train] 使用 contamination={contamination:.4f}")

    # ── 确定评分集（有切分用 X_test，无切分用全量 X_train）───────────────
    X_score = state.X_test if state.X_test is not None else state.X_train
    score_on = "X_test" if state.X_test is not None else "X_train（无标签路径）"

    # ── 主路径：Isolation Forest ────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        scores, y_pred_raw, threshold, model = _fit_iforest(
            state.X_train, X_score,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        used_fallback = False
        fallback_reason = None
        logger.info(f"[train] IsolationForest fit 完成（{time.perf_counter()-t0:.2f}s）")

    except Exception as e:
        logger.warning(f"[train] IForest 失败，回退 z-score 基线: {e}")
        scores, y_pred_raw, threshold = _zscore_baseline(X_score)
        model = None
        used_fallback = True
        fallback_reason = str(e)

    # ── 标签映射：-1=异常/1=正常 → 1=异常/0=正常 ───────────────────────
    #   sklearn predict 返回 -1 或 1；z-score 基线直接返回 0/1
    if not used_fallback:
        y_pred = (y_pred_raw == -1).astype(int)
    else:
        y_pred = y_pred_raw.astype(int)

    n_detected = int(y_pred.sum())

    # ── 写入状态 ────────────────────────────────────────────────────────
    state.trained_model        = model
    state.anomaly_scores       = scores.tolist()
    state.y_pred               = y_pred.tolist()
    state.threshold            = float(threshold)
    state.n_anomalies_detected = n_detected

    state.train_info = {
        "algorithm":          algorithm if not used_fallback else "zscore_baseline（兜底）",
        "contamination":      contamination,
        "n_estimators":       n_estimators if not used_fallback else None,
        "random_state":       random_state,
        "scored_on":          score_on,
        "n_score_samples":    len(X_score),
        "n_anomalies_detected": n_detected,
        "detection_rate":     round(n_detected / len(X_score), 4),
        "threshold":          float(threshold),
        "elapsed_sec":        round(time.perf_counter() - t0, 3),
        "used_fallback":      used_fallback,
        "fallback_reason":    fallback_reason,
    }

    logger.info(
        f"[train] 完成。检出异常: {n_detected}/{len(X_score)} "
        f"({n_detected/len(X_score):.2%})  threshold={threshold:.4f}  "
        f"fallback={used_fallback}"
    )
    return state


# ══════════════════════════════════════════════════════════
# 内部实现：拆成小函数，便于 T2.4 扩展其他算法
# ══════════════════════════════════════════════════════════

def _fit_iforest(
    X_train: np.ndarray,
    X_score: np.ndarray,
    contamination: float,
    n_estimators: int,
    random_state: int,
) -> tuple:
    """训练 IForest 并返回 (scores, y_pred_raw, threshold, model)。

    scores     : 越大越可疑（= -decision_function）
    y_pred_raw : sklearn 原始输出，-1=异常/1=正常（未映射）
    threshold  : 对应 contamination 的分数切割点
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,          # 并行，加速大数据集
    )
    model.fit(X_train)

    # decision_function：正常样本得分高，异常样本得分低
    # 取负后：正常=低分，异常=高分，与直觉一致
    raw_df   = model.decision_function(X_score)
    scores   = -raw_df

    y_pred_raw = model.predict(X_score)   # -1 或 1，尚未映射

    # sklearn 内部阈值：-model.offset_ 对应 contamination 分位点
    # 取负后，threshold 是 scores 的切割点（scores > threshold → 异常）
    threshold = -model.offset_

    return scores, y_pred_raw, threshold, model


def _zscore_baseline(X: np.ndarray) -> tuple:
    """z-score 统计基线：作为 IForest 失败时的兜底。

    逻辑：每个样本各维度 z-score 的最大值作为异常分数
          超过 3-sigma 视为异常

    Returns:
        (scores, y_pred, threshold)
        scores : 每个样本的最大 |z-score|，越大越可疑
        y_pred : 0/1（1=异常）
        threshold : 固定 3.0
    """
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std  = np.where(std == 0, 1e-8, std)   # 防止除零

    z_scores = np.abs((X - mean) / std)
    scores   = z_scores.max(axis=1)         # 每行取最大 |z|

    threshold = 3.0
    y_pred    = (scores > threshold).astype(int)

    logger.info(
        f"[zscore_baseline] threshold=3.0，"
        f"检出异常: {y_pred.sum()}/{len(X)} ({y_pred.mean():.2%})"
    )
    return scores, y_pred, threshold

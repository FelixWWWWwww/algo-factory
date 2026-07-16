# factory/nodes/split.py
"""
T1.7: 数据切分节点 & 合成异常数据工具

两个功能：
  A. split_node      — 分层切分 X_processed，保持异常比例一致
  B. make_synthetic_dataset — 生成极不平衡合成数据集用于 Mock/自测

关键约束：
  - 必须用 stratify 保证测试集含异常（否则 Recall 无法计算）
  - 无标签时跳过切分，全量数据交给训练节点（无监督）
  - 测试集 scaler 复用 T1.6 已拟合的对象（.transform，不再 .fit）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 节点 A：分层切分
# ══════════════════════════════════════════════════════════

def split_node(
    state: "TaskState",
    test_size: float = 0.2,
    random_state: int = 42,
    min_test_anomalies: int = 5,
) -> "TaskState":
    """分层切分节点：将 X_processed / y_true 切成训练集和测试集。

    两条路径：
      有标签 → stratify 分层切分，保证测试集含足够异常
      无标签 → 跳过切分，全量数据作为训练集（无监督路径）

    Args:
        state:              全局状态
        test_size:          测试集比例（默认 0.2）
        random_state:       随机种子（保证可复现）
        min_test_anomalies: 测试集最少需含几个异常；不足时记警告

    写入状态：
        state.X_train, X_test, y_train, y_test, split_info
    """
    X = state.X_processed
    y = state.y_true

    if X is None:
        logger.error("[split] X_processed 为空，请先运行 preprocess_node")
        state.add_error("split", "RuntimeError", "X_processed 为空")
        return state

    # ── 路径 1：无标签，跳过切分 ──────────────────────────────────────
    if y is None:
        logger.info("[split] 无标签列，跳过切分 → 全量数据用于无监督训练")
        state.X_train = X
        state.X_test  = None
        state.y_train = None
        state.y_test  = None
        state.split_info = {
            "mode":          "unsupervised_no_split",
            "n_train":       len(X),
            "n_test":        0,
            "note":          "无标签，评估退化为 Top-K 人工审阅",
        }
        return state

    # ── 路径 2：有标签，stratify 分层切分 ───────────────────────────────
    n_anomaly_total = int((y == 1).sum())
    n_total = len(X)

    # 极端情况：异常样本太少，stratify 会报错（至少需要 2 个才能两边各分 1 个）
    if n_anomaly_total < 2:
        logger.warning(
            f"[split] 异常样本仅 {n_anomaly_total} 个，无法 stratify，"
            "退回随机切分"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,           # ← 核心：保证两侧异常比例一致
            random_state=random_state,
        )

    # ── 校验测试集含足够异常 ────────────────────────────────────────────
    n_test_anomaly = int((y_test == 1).sum())
    if n_test_anomaly == 0:
        logger.warning(
            "[split] ⚠️  测试集 0 个异常！Recall 分母为零，评估将降级为 Top-K。"
            "建议：增大数据量或调高 anomaly_ratio。"
        )
    elif n_test_anomaly < min_test_anomalies:
        logger.warning(
            f"[split] ⚠️  测试集仅 {n_test_anomaly} 个异常（建议≥{min_test_anomalies}），"
            "Recall 估计不稳定。"
        )

    # ── 写入状态 ────────────────────────────────────────────────────────
    state.X_train = X_train
    state.X_test  = X_test
    state.y_train = y_train
    state.y_test  = y_test

    train_ratio = round(float((y_train == 1).sum() / len(y_train)), 4)
    test_ratio  = round(float((y_test  == 1).sum() / len(y_test)),  4)

    state.split_info = {
        "mode":              "stratified",
        "test_size":         test_size,
        "random_state":      random_state,
        "n_total":           n_total,
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_train_anomaly":   int((y_train == 1).sum()),
        "n_test_anomaly":    n_test_anomaly,
        "train_anomaly_ratio": train_ratio,
        "test_anomaly_ratio":  test_ratio,
        "ratio_drift":       round(abs(train_ratio - test_ratio), 4),
    }

    logger.info(
        f"[split] 完成。训练集: {len(X_train)} 行（异常 {(y_train==1).sum()} 个，"
        f"{train_ratio:.2%}）| "
        f"测试集: {len(X_test)} 行（异常 {n_test_anomaly} 个，{test_ratio:.2%}）"
    )
    return state


# ══════════════════════════════════════════════════════════
# 工具 B：合成极不平衡数据集
# ══════════════════════════════════════════════════════════

def make_synthetic_dataset(
    n_samples:     int   = 1000,
    n_features:    int   = 8,
    anomaly_ratio: float = 0.02,
    random_state:  int   = 42,
    save_path:     Optional[str] = None,
) -> pd.DataFrame:
    """生成可复现的极不平衡合成数据集，用于 Mock/自测。

    正常样本：多元正态分布（均值=0，随机协方差）
    异常样本：均值偏移 + 方差放大（模拟真实离群点）

    Args:
        n_samples:     总样本数
        n_features:    特征维度
        anomaly_ratio: 异常占比（建议 0.01–0.05；>0.1 会退化成普通二分类）
        random_state:  随机种子（保证可复现）
        save_path:     若指定则存为 CSV

    Returns:
        DataFrame，含 n_features 个特征列 + 'label'（1=异常/0=正常）

    Raises:
        ValueError: anomaly_ratio 超出合理范围
    """
    # ── 参数校验 ─────────────────────────────────────────────────────────
    if not (0.005 <= anomaly_ratio <= 0.15):
        raise ValueError(
            f"anomaly_ratio={anomaly_ratio:.3f} 超出合理范围 [0.005, 0.15]。\n"
            "  过低（<0.5%）→ 测试集可能 0 异常；"
            "过高（>15%）→ 退化成普通二分类，失去异常检测意义。"
        )

    rng = np.random.default_rng(random_state)

    n_anomaly = max(2, int(n_samples * anomaly_ratio))  # 至少 2 个，保证 stratify 可用
    n_normal  = n_samples - n_anomaly

    # ── 正常样本：多元正态，协方差随机（模拟真实特征相关性） ───────────────
    # 用 Cholesky 分解生成正定协方差矩阵
    A = rng.standard_normal((n_features, n_features))
    cov = A @ A.T / n_features          # 正定矩阵
    mean_normal = np.zeros(n_features)
    X_normal = rng.multivariate_normal(mean_normal, cov, size=n_normal)

    # ── 异常样本：多个异常簇，各自偏移 ──────────────────────────────────
    # 分 2–3 个异常簇，模拟不同类型异常（点异常、上下文异常）
    n_clusters = min(3, max(1, n_anomaly // 5))
    X_anom_parts = []
    cluster_size = n_anomaly // n_clusters

    for i in range(n_clusters):
        # 每个簇偏移方向不同，方差更大
        shift = rng.uniform(3, 6, size=n_features) * rng.choice([-1, 1], n_features)
        size  = cluster_size if i < n_clusters - 1 else n_anomaly - cluster_size * i
        X_part = rng.multivariate_normal(shift, cov * 4, size=size)
        X_anom_parts.append(X_part)

    X_anomaly = np.vstack(X_anom_parts)

    # ── 拼合 + 打标签 ───────────────────────────────────────────────────
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)

    # 打乱顺序（真实数据不会正常/异常整齐排列）
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # ── 构造 DataFrame ───────────────────────────────────────────────────
    col_names = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=col_names)
    df["label"] = y

    actual_ratio = y.mean()
    logger.info(
        f"[synthetic] 生成完成: {n_samples} 行 × {n_features} 特征 | "
        f"异常 {n_anomaly} 个（{actual_ratio:.2%}）| seed={random_state}"
    )

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"[synthetic] 已保存至: {save_path}")

    return df

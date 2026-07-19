# factory/nodes/preprocess.py
"""
T1.6: 预处理节点

职责：
  1. 缺失值填充（数值→中位数，类别→众数）
  2. 类别特征 One-Hot 编码
  3. 按算法族决定是否标准化
     - 距离/密度类（LOF、OneClassSVM）：强制 StandardScaler
     - 树类（IsolationForest）         ：跳过标准化
  4. 分离标签列（仅用于评估，不参与训练）

严禁：SMOTE / 任何过采样
注意：Scaler 只能在训练集上 fit；此阶段存储 fitted scaler，
      T1.7 切分后 test 集只能调 transform（不能再 fit）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)

# 需要强制标准化的算法族（基于距离或密度，量纲敏感）
DISTANCE_BASED_ALGORITHMS = {
    "LocalOutlierFactor", "LOF",
    "OneClassSVM", "OCSVM",
    "ECOD", "COPOD",          # pyod 算法同样量纲敏感
}


def preprocess_node(
    state: "TaskState",
    algorithm: str = "IsolationForest",
    label_col: str = "label",
) -> "TaskState":
    """预处理节点：缺失填充 → 编码 → 按需标准化 → 分离标签

    Args:
        state:     全局状态（读 raw_df，写 X_processed / scaler / y_true）
        algorithm: 后续使用的算法名，决定是否标准化
        label_col: 标签列名（默认 "label"；1=异常 0=正常）

    Returns:
        更新后的 state
    """
    if state.raw_df is None:
        logger.error("[preprocess] raw_df 为空，请先运行 data_ingestion_node")
        state.add_error("preprocess", "RuntimeError", "raw_df 为空")
        return state

    df = state.raw_df.copy()   # 保护原始数据，不原地修改
    info: dict = {}            # 记录本次操作（写入 preprocessing_info）

    # ── 1. 分离标签（标签只用于评估，不参与任何预处理拟合） ───────────
    if label_col in df.columns:
        state.y_true = df[label_col].values.astype(int)
        df = df.drop(columns=[label_col])
        info["label_separated"] = True
        logger.info(f"[preprocess] 分离标签列 '{label_col}'，"
                    f"异常样本数: {(state.y_true == 1).sum()}")
    else:
        state.y_true = None
        info["label_separated"] = False
        logger.info("[preprocess] 无标签列，走无监督路径")

    # ── 2. 拆分数值列与类别列 ──────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    info["num_cols"] = num_cols
    info["cat_cols"] = cat_cols

    # ── 3. 缺失值填充 ─────────────────────────────────────────────────
    # 数值列：中位数填充（不用均值——离群点会拉偏均值）
    num_medians = {}
    for col in num_cols:
        median_val = df[col].median()
        n_filled = df[col].isna().sum()
        if n_filled > 0:
            df[col] = df[col].fillna(median_val)
            num_medians[col] = round(float(median_val), 4)
            logger.info(f"[preprocess] 填充 '{col}': {n_filled} 个缺失 → 中位数 {median_val:.4f}")
    info["num_medians_used"] = num_medians

    # 类别列：众数填充
    cat_modes = {}
    for col in cat_cols:
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
        n_filled = df[col].isna().sum()
        if n_filled > 0:
            df[col] = df[col].fillna(mode_val)
            cat_modes[col] = str(mode_val)
    info["cat_modes_used"] = cat_modes

    # ── 4. 类别特征 One-Hot 编码 ──────────────────────────────────────
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=float)
        logger.info(f"[preprocess] One-Hot 编码类别列: {cat_cols}")
    info["one_hot_encoded"] = cat_cols

    # 更新最终特征列名
    state.feature_names = df.columns.tolist()
    X = df.values.astype(float)

    # ── 5. 按算法族决定是否标准化 ─────────────────────────────────────
    needs_scaling = algorithm in DISTANCE_BASED_ALGORITHMS
    info["algorithm"] = algorithm
    info["scaling_applied"] = needs_scaling

    if needs_scaling:
        logger.info(f"[preprocess] {algorithm} 属距离/密度类，强制 StandardScaler")
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            info["scaler_type"] = "StandardScaler"
        except Exception as e:
            # 兜底：极端离群点导致 StandardScaler 数值溢出时，换 RobustScaler
            logger.warning(f"[preprocess] StandardScaler 失败，回退 RobustScaler: {e}")
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            info["scaler_type"] = "RobustScaler（兜底）"

        state.scaler = scaler
    else:
        logger.info(f"[preprocess] {algorithm} 属树类，跳过标准化")
        state.scaler = None
        info["scaler_type"] = None

    # ── 6. 写入状态 ───────────────────────────────────────────────────
    state.X_processed = X
    state.preprocessing_info = info

    logger.info(
        f"[preprocess] 完成。特征矩阵: {X.shape}  "
        f"标准化: {'是' if needs_scaling else '否'}  "
        f"标签: {'有' if state.y_true is not None else '无'}"
    )
    return state

# factory/nodes/profile.py
"""轻量数据画像节点：在 Planner 之前对数据快速体检，产出 data_profile，供 LLM 动态选型。

不做训练/切分，只算能指导选型的统计量：规模、异常占比、特征类型、量纲差异、缺失、是否含时间列。
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)
_TIME_HINTS = ("time", "date", "timestamp", "ts")


def profile_node(state: "TaskState", data_path: str, label_col: str = "label") -> "TaskState":
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.warning(f"[profile] 读取失败: {e}")
        state.data_profile = {}
        return state

    sc = getattr(getattr(state, "task_card", None), "suspected_columns", {}) or {}
    lc = sc.get("label") or label_col
    has_labels = lc in df.columns
    anomaly_ratio = round(float((df[lc] == 1).mean()), 4) if has_labels else None

    feat = df.drop(columns=[lc]) if has_labels else df
    numeric = feat.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feat.columns if c not in numeric]
    time_cols = [c for c in feat.columns if any(h in str(c).lower() for h in _TIME_HINTS)]

    scale_disparity = False
    if len(numeric) >= 2:
        stds = feat[numeric].std().replace(0, 1e-9)
        scale_disparity = bool((stds.max() / stds.min()) > 100)

    missing_rate = round(float(feat.isna().mean().max()), 4) if len(feat.columns) else 0.0

    # 异常紧致度：异常样本的特征离散度 / 全体离散度。
    #   << 1 → 异常聚成致密簇（LOF 靠局部密度会漏检）；>= 1 → 异常离散分布（LOF 擅长）。
    #   这是决定"同一算法在不同数据上成败"的关键判别特征。
    anomaly_compactness = None
    if has_labels and numeric:
        try:
            y = (df[lc].values == 1)
            fn = feat[numeric]
            if y.sum() > 1:
                a_std = float(fn[y].std().mean())
                o_std = float(fn.std().mean())
                anomaly_compactness = round(a_std / (o_std + 1e-9), 3)
        except Exception:
            anomaly_compactness = None

    state.data_profile = {
        "n_samples": int(len(df)),
        "n_features": int(feat.shape[1]),
        "feature_types": {
            "numeric": numeric[:20],
            "categorical": non_numeric[:20],
            "temporal": time_cols,
        },
        "has_labels": has_labels,
        "anomaly_ratio": anomaly_ratio,
        "missing_rate": missing_rate,
        "scale_disparity": scale_disparity,
        "anomaly_compactness": anomaly_compactness,
        "has_time_column": len(time_cols) > 0,
    }
    if anomaly_ratio is not None:
        state.anomaly_ratio = anomaly_ratio
    logger.info(f"[profile] {state.data_profile['n_samples']}x{state.data_profile['n_features']} "
                f"labels={has_labels} ratio={anomaly_ratio} scale_disparity={scale_disparity}")
    return state

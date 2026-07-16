# factory/nodes/ingestion.py
"""
T1.5: 数据接入节点 & EDA 节点

两个节点函数，均接收 TaskState，返回更新后的 TaskState。
后续 T2.1 会把它们串进 Graph；现阶段可直接调用测试。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 节点 1：数据接入
# ─────────────────────────────────────────────

def data_ingestion_node(state: "TaskState", csv_path: str) -> "TaskState":
    """读取 CSV，推断 schema，写入全局状态。

    写入字段：
        state.raw_df       — 原始 DataFrame
        state.schema_info  — 列名 → {dtype, n_missing, missing_rate} 映射
    """
    logger.info(f"[ingestion] 读取文件: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"[ingestion] 读取失败: {e}")
        state.add_error("ingestion", "RuntimeError", str(e))
        return state

    state.raw_df = df

    # 推断 schema：每列 dtype + 缺失情况
    schema = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        schema[col] = {
            "dtype": str(df[col].dtype),
            "n_missing": n_missing,
            "missing_rate": round(n_missing / len(df), 4),
        }
    state.schema_info = schema

    logger.info(f"[ingestion] 读取完成: {df.shape[0]} 行 × {df.shape[1]} 列")
    return state


def ingestion_node(
    state: "TaskState",
    csv_path: str = "",
    data_path: str = "",
) -> "TaskState":
    """兼容旧入口名。"""
    path = csv_path or data_path
    if not path:
        logger.error("[ingestion] 未提供 CSV 路径")
        state.add_error("ingestion", "RuntimeError", "未提供 CSV 路径")
        return state
    return data_ingestion_node(state, path)


# ─────────────────────────────────────────────
# 节点 2：EDA 分析 + LLM 摘要
# ─────────────────────────────────────────────

def eda_node(state: "TaskState", llm_client) -> "TaskState":
    """EDA 分析节点。

    步骤：
      1. 统计缺失率、数值分布（分位数）、异常占比
      2. 用 LLM 生成自然语言摘要，写入 state.eda_summary
         若 LLM 失败，回退模板字符串（兜底）

    写入字段：
        state.anomaly_ratio  — 实测异常占比（有 label 列时）
        state.schema_info    — 追加分位数统计
        state.eda_summary    — 自然语言摘要
    """
    df = state.raw_df
    if df is None:
        logger.error("[eda] raw_df 为空，请先运行 data_ingestion_node")
        state.add_error("eda", "RuntimeError", "raw_df 为空")
        return state

    n_rows, n_cols = df.shape

    # ── 1. 异常占比（有标签列 'label' 时计算） ──────────────
    anomaly_ratio: float | None = None
    if "label" in df.columns:
        # 约定：1 = 异常，0 = 正常
        anomaly_ratio = round(float((df["label"] == 1).sum() / n_rows), 4)
        state.anomaly_ratio = anomaly_ratio
        logger.info(f"[eda] 实测异常占比: {anomaly_ratio:.2%}")

    # ── 2. 数值列分位数统计（不用均值！极不平衡下均值被正常样本主导）──
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = "label" if "label" in df.columns else None
    feature_cols = [c for c in num_cols if c != label_col]

    for col in feature_cols:
        q = df[col].quantile([0.25, 0.5, 0.75, 0.99]).to_dict()
        state.schema_info.setdefault(col, {}).update({
            "q25":  round(q[0.25], 4),
            "q50":  round(q[0.50], 4),
            "q75":  round(q[0.75], 4),
            "q99":  round(q[0.99], 4),
            "iqr":  round(q[0.75] - q[0.25], 4),
        })

    # ── 3. 组装给 LLM 的上下文 ──────────────────────────────
    missing_cols = {
        col: info["missing_rate"]
        for col, info in state.schema_info.items()
        if info.get("missing_rate", 0) > 0
    }
    ratio_str = f"{anomaly_ratio:.2%}" if anomaly_ratio is not None else "未知（无标签）"

    prompt = (
        f"你是异常检测领域专家，请用 2–4 句中文描述以下数据集的分布特征，"
        f"重点说明不平衡程度和潜在风险，不要用 accuracy 作为指标。\n\n"
        f"数据集信息：\n"
        f"- 行数: {n_rows}，列数: {n_cols}\n"
        f"- 异常占比（label=1）: {ratio_str}\n"
        f"- 缺失列: {missing_cols if missing_cols else '无'}\n"
        f"- 数值特征列: {feature_cols[:8]}{'...' if len(feature_cols) > 8 else ''}\n"
        f"\nEDA 分析报告（关键词：分布、分位数、缺失、不平衡、推荐指标）："
    )

    # ── 4. LLM 生成摘要，失败时模板兜底 ────────────────────
    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.chat(messages)
        if response.get("error"):
            raise ValueError(response["error"])
        summary = (response.get("message") or "").strip()
        if not summary:
            raise ValueError("LLM 返回为空")
        state.eda_summary = summary
        logger.info("[eda] LLM 摘要生成成功")
    except Exception as e:
        logger.warning(f"[eda] LLM 失败，使用模板兜底: {e}")
        # 兜底：模板字符串
        state.eda_summary = (
            f"数据集共 {n_rows} 行、{n_cols} 列。"
            f"疑似异常占比 {ratio_str}，属极度不平衡场景。"
            + (f"缺失率最高列：{max(missing_cols, key=missing_cols.get)}（{max(missing_cols.values()):.1%}）。" if missing_cols else "")
            + "推荐主指标 PR-AUC，禁用 accuracy。"
        )

    logger.info(f"[eda] EDA 摘要: {state.eda_summary[:80]}...")
    return state

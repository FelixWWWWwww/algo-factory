# tests/test_boundary.py
import numpy as np
import pandas as pd
import pytest
from factory.state import TaskState
from factory.nodes import (data_ingestion_node, preprocess_node, split_node,
                           train_node, evaluate_node)


def _pipeline(csv):
    s = TaskState(user_query="x")
    s = data_ingestion_node(s, str(csv))
    s = preprocess_node(s, algorithm="IsolationForest")
    s = split_node(s)
    s = train_node(s, algorithm="IsolationForest")
    return evaluate_node(s)


def test_no_label_goes_unlabeled_path(tmp_path):
    """无 label 列 → 无监督 Top-K 路径，不崩溃。"""
    csv = tmp_path / "nolabel.csv"
    pd.DataFrame(np.random.randn(300, 5),
                 columns=[f"f{i}" for i in range(5)]).to_csv(csv, index=False)
    s = _pipeline(csv)
    assert s.topk_indices                      # 产出 Top-K
    assert s.final_metrics == {}               # 无标签无可靠指标


def test_high_dim(tmp_path):
    """>100 维不崩溃。"""
    from factory.nodes import make_synthetic_dataset
    csv = tmp_path / "hd.csv"
    make_synthetic_dataset(400, 120, 0.03, save_path=str(csv))
    s = _pipeline(csv)
    assert "pr_auc" in s.final_metrics


def test_missing_values(tmp_path):
    """含缺失值 → 填充后正常跑。"""
    from factory.nodes import make_synthetic_dataset
    df = make_synthetic_dataset(400, 6, 0.03)
    df.iloc[0:10, 0] = np.nan
    csv = tmp_path / "miss.csv"
    df.to_csv(csv, index=False)
    s = _pipeline(csv)
    assert "pr_auc" in s.final_metrics
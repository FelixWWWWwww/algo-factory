import numpy as np
import pandas as pd
from factory.state import TaskState
from factory.nodes import (data_ingestion_node, preprocess_node, split_node,
                           train_node, evaluate_node, make_synthetic_dataset)


def _pipeline(csv):
    s = TaskState(user_query="x")
    s = data_ingestion_node(s, str(csv))
    s = preprocess_node(s, algorithm="IsolationForest")
    s = split_node(s)
    s = train_node(s, algorithm="IsolationForest")
    return evaluate_node(s)


def test_no_label_goes_unlabeled_path(tmp_path):
    csv = tmp_path / "nolabel.csv"
    pd.DataFrame(np.random.randn(300, 5),
                 columns=[f"f{i}" for i in range(5)]).to_csv(csv, index=False)
    s = _pipeline(csv)
    assert s.topk_indices
    assert s.final_metrics == {}


def test_high_dim(tmp_path):
    csv = tmp_path / "hd.csv"
    make_synthetic_dataset(400, 120, 0.03, save_path=str(csv))
    s = _pipeline(csv)
    assert "pr_auc" in s.final_metrics


def test_missing_values(tmp_path):
    df = make_synthetic_dataset(400, 6, 0.03)
    df.iloc[0:10, 0] = np.nan
    csv = tmp_path / "miss.csv"
    df.to_csv(csv, index=False)
    s = _pipeline(csv)
    assert "pr_auc" in s.final_metrics

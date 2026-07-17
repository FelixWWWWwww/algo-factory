import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor


def run(data_path: str) -> dict:
    """生成代码：LocalOutlierFactor 异常检测。若含 label 列(1=异常/0=正常)则计算监督指标。"""
    df = pd.read_csv(data_path)
    y_true = df.pop("label").values if "label" in df.columns else None
    X = df.values.astype(float)
    X_score = StandardScaler().fit_transform(X)

    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02)
    model.fit(X_score)
    scores = -model.decision_function(X_score)          # 越大越可疑
    y_pred = (model.predict(X_score) == -1).astype(int)  # -1→1(异常), 1→0(正常)

    result = {"n_anomalies_detected": int(y_pred.sum())}
    if y_true is not None and len(np.unique(y_true)) > 1:
        result.update({
            "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        })
    else:
        result["note"] = "无标签，仅输出检出数供 Top-K 人工审阅"

    print("RESULT_JSON:" + json.dumps(result))
    return result


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "data.csv")

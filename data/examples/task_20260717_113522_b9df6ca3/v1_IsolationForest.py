
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import json

def run(data_path: str, contamination: float = 0.02) -> dict:
    """生成的代码：Isolation Forest 异常检测

    约定：若存在标签列 'label'，1=异常、0=正常（仅用于评估，不参与训练）。
    """
    df = pd.read_csv(data_path)

    # 分离标签（若有）——异常检测通常无监督训练，标签仅用于评估
    y_true = df.pop("label").values if "label" in df.columns else None
    X = df.values

    # 标准化（IForest 不强依赖，但保持流程一致；LOF/OCSVM 则必做）
    X_scaled = StandardScaler().fit_transform(X)

    # 训练（无监督）
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)

    # 异常分数：decision_function 越小越异常，取负号使"越大越异常"
    scores = -model.decision_function(X_scaled)
    # 硬标签：sklearn 中 -1=异常、1=正常，映射为 1=异常、0=正常
    y_pred = (model.predict(X_scaled) == -1).astype(int)

    result = {"n_anomalies_detected": int(y_pred.sum()), "contamination": contamination}

    # 有标签才能算监督指标；否则只输出分数分布，交人工审阅 Top-K
    if y_true is not None and len(np.unique(y_true)) > 1:
        result.update({
            "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        })
    else:
        result["note"] = "无有效标签，仅输出异常分数供 Top-K 人工审阅"

    print("RESULT_JSON:" + json.dumps(result))
    return result

if __name__ == "__main__":
    import sys
    result = run(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
    print(json.dumps(result))

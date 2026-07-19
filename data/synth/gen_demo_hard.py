"""生成"专治 LOF"的演示数据：异常聚成致密簇 → LOF 因局部密度误判而失败，
IForest 仍稳过。用于 Demo 稳定复现 FailureCase 沉淀。

用法: python data/synth/gen_demo_hard.py  → 产出 data/synth/demo_hard.csv
"""
import numpy as np, pandas as pd, os

def make_hard(n=1500, d=6, ratio=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n_anom = int(n * ratio); n_norm = n - n_anom
    X_norm = rng.normal(0, 1, (n_norm, d))
    # 异常：紧紧挤在远处一个致密小簇（方差极小）——LOF 会误判为局部正常
    X_anom = np.full(d, 5.0) + rng.normal(0, 0.15, (n_anom, d))
    X = np.vstack([X_norm, X_anom])
    y = np.r_[np.zeros(n_norm, int), np.ones(n_anom, int)]
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(d)])
    df["label"] = y
    return df

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "demo_hard.csv")
    make_hard().to_csv(out, index=False)
    print(f"已生成: {out}")

# factory/nodes/train.py
"""
T2.6: 单模型训练节点（支持多算法）

职责：
  1. 从 state.contamination（或 state.anomaly_ratio）读取异常比例
  2. 在 X_train 上 fit 指定算法（IsolationForest / LOF / OneClassSVM）
  3. 对 X_test（或无切分时对 X_train）打分 → anomaly_scores
  4. 产出映射后的硬标签 y_pred（1=异常/0=正常）
  5. 记录 threshold 与 n_anomalies_detected
  6. fit 失败时回退 z-score / IQR 统计基线

关键约定：
  - sklearn 三种模型 predict() 均返回 -1=异常 / 1=正常
    评估前必须映射：y_pred = (raw == -1).astype(int)
  - decision_function() 越小越异常，取负使"越大越可疑"
    anomaly_scores = -model.decision_function(X_score)
  - 三种模型判定边界均为 decision_function = 0
    故 scores = -decision_function 的切割点固定为 threshold = 0.0
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import IsolationForest

if TYPE_CHECKING:
    from factory.state import TaskState

logger = logging.getLogger(__name__)

# 算法名归一化：短名 → sklearn 规范名
_ALGO_ALIASES = {
    "iforest": "IsolationForest",
    "isolationforest": "IsolationForest",
    "lof": "LocalOutlierFactor",
    "localoutlierfactor": "LocalOutlierFactor",
    "ocsvm": "OneClassSVM",
    "oneclasssvm": "OneClassSVM",
}


def _canonical(algorithm: str) -> str:
    return _ALGO_ALIASES.get(str(algorithm).strip().lower(), algorithm)


def train_node(
    state: "TaskState",
    algorithm: str = "IsolationForest",
    n_estimators: int = 100,
    random_state: int = 42,
    import_path: str = "",
) -> "TaskState":
    """单模型训练节点：fit → score → 映射硬标签。

    Args:
        state:        全局状态（读 X_train/X_test，写 anomaly_scores/y_pred/...）
        algorithm:    "IsolationForest" / "LocalOutlierFactor" / "OneClassSVM"
                      （亦接受 IForest / LOF / OCSVM 短名）
        n_estimators: IForest 树数量（仅 IForest 生效）
        random_state: 随机种子

    写入状态：
        state.trained_model / anomaly_scores / y_pred / threshold /
        n_anomalies_detected / train_info
    """
    if state.X_train is None:
        logger.error("[train] X_train 为空，请先运行 split_node")
        state.add_error("train", "RuntimeError", "X_train 为空")
        return state

    algo = _canonical(algorithm)

    # ── contamination：优先 EDA 实测值，其次全局设定 ──────────────────
    contamination = (
        state.anomaly_ratio
        if state.anomaly_ratio is not None
        else state.contamination
    )
    contamination = float(np.clip(contamination, 1e-4, 0.5))
    logger.info(f"[train] 算法={algo}  contamination={contamination:.4f}")

    # ── 评分集（有切分用 X_test，无切分用全量 X_train）────────────────
    X_score = state.X_test if state.X_test is not None else state.X_train
    score_on = "X_test" if state.X_test is not None else "X_train（无标签路径）"

    # ── 主路径：指定算法 ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        scores, y_pred_raw, threshold, model = _fit_model(
            algo, state.X_train, X_score,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            import_path=import_path,
        )
        used_fallback, fallback_reason = False, None
        logger.info(f"[train] {algo} fit 完成（{time.perf_counter()-t0:.2f}s）")
    except Exception as e:
        logger.warning(f"[train] {algo} 失败，回退 z-score 基线: {e}")
        scores, y_pred_raw, threshold = _zscore_baseline(X_score)
        model = None
        used_fallback, fallback_reason = True, str(e)

    # ── 标签映射：-1=异常/1=正常 → 1=异常/0=正常 ─────────────────────
    if not used_fallback:
        y_pred = (y_pred_raw == -1).astype(int)
    else:
        y_pred = y_pred_raw.astype(int)

    n_detected = int(y_pred.sum())

    # ── 写入状态 ──────────────────────────────────────────────────────
    state.trained_model        = model
    state.anomaly_scores       = scores.tolist()
    state.y_pred               = y_pred.tolist()
    state.threshold            = float(threshold)
    state.n_anomalies_detected = n_detected
    state.train_info = {
        "algorithm":            algo if not used_fallback else "zscore_baseline（兜底）",
        "requested_algorithm":  algorithm,
        "contamination":        contamination,
        "n_estimators":         n_estimators if algo == "IsolationForest" and not used_fallback else None,
        "random_state":         random_state,
        "scored_on":            score_on,
        "n_score_samples":      len(X_score),
        "n_anomalies_detected": n_detected,
        "detection_rate":       round(n_detected / len(X_score), 4),
        "threshold":            float(threshold),
        "elapsed_sec":          round(time.perf_counter() - t0, 3),
        "used_fallback":        used_fallback,
        "fallback_reason":      fallback_reason,
    }
    logger.info(
        f"[train] 完成。检出 {n_detected}/{len(X_score)} "
        f"({n_detected/len(X_score):.2%})  threshold={threshold:.4f}  fallback={used_fallback}"
    )
    return state


# ══════════════════════════════════════════════════════════
# 内部实现：多算法工厂（统一 fit / score / 映射口径）
# ══════════════════════════════════════════════════════════

def _fit_model(
    algo: str,
    X_train: np.ndarray,
    X_score: np.ndarray,
    contamination: float,
    n_estimators: int,
    random_state: int,
    import_path: str = "",
) -> tuple:
    """训练指定算法，返回 (scores, y_pred_raw, threshold, model)。

    scores     : 越大越可疑（= -decision_function）
    y_pred_raw : sklearn 原始输出，-1=异常/1=正常（未映射）
    threshold  : 固定 0.0（decision_function 判定边界，取负后不变）
    """
    if algo == "IsolationForest":
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train)

    elif algo == "LocalOutlierFactor":
        from sklearn.neighbors import LocalOutlierFactor
        # novelty=True 才能对新样本 predict / decision_function
        n_neighbors = int(min(20, max(5, len(X_train) - 1)))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination=contamination,
        )
        model.fit(X_train)

    elif algo == "OneClassSVM":
        from sklearn.svm import OneClassSVM
        # nu ≈ 异常比例上界，映射 contamination
        nu = float(np.clip(contamination, 1e-3, 0.5))
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        model.fit(X_train)

    else:
        # 非预设算法：按 import_path 动态加载（支持 LLM 选出的任意算法，如 pyod/其它 sklearn）
        return _dynamic_fit_score(algo, import_path, X_train, X_score, contamination)

    # 三种模型统一：decision_function 正常样本高分、异常样本低分（<0 为异常）
    raw_df     = model.decision_function(X_score)
    scores     = -raw_df                       # 取负：越大越可疑
    y_pred_raw = model.predict(X_score)         # -1 / 1
    threshold  = 0.0                            # 判定边界（scores > 0 → 异常）
    return scores, y_pred_raw, threshold, model


def _zscore_baseline(X: np.ndarray) -> tuple:
    """z-score 统计基线：IForest/LOF/OCSVM 全失败时的兜底。

    每样本各维 z-score 的最大值作为异常分数，超 3-sigma 视为异常。
    Returns: (scores, y_pred[0/1], threshold=3.0)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1e-8, std)
    z = np.abs((X - mean) / std)
    scores = z.max(axis=1)
    threshold = 3.0
    y_pred = (scores > threshold).astype(int)
    logger.info(f"[zscore_baseline] 检出 {y_pred.sum()}/{len(X)} ({y_pred.mean():.2%})")
    return scores, y_pred, threshold


# 常见算法名 → import 路径（LLM 未给 import_path 时的兜底猜测）
_GUESS_PATH = {
    "EllipticEnvelope": "sklearn.covariance.EllipticEnvelope",
    "SGDOneClassSVM": "sklearn.linear_model.SGDOneClassSVM",
    "ECOD": "pyod.models.ecod.ECOD",
    "COPOD": "pyod.models.copod.COPOD",
    "KNN": "pyod.models.knn.KNN",
    "HBOS": "pyod.models.hbos.HBOS",
    "AutoEncoder": "pyod.models.auto_encoder.AutoEncoder",
}


def _dynamic_fit_score(algo, import_path, X_train, X_score, contamination):
    """动态 import 任意异常检测算法，统一打分口径（scores 越大越可疑，y_pred_raw 用 -1=异常）。

    兼容两套约定：
      - sklearn 家族：decision_function 越大越正常 → 取负；predict 返回 -1/1。
      - pyod 家族：decision_function 越大越异常 → 不取负；predict 返回 1=异常/0=正常 → 统一成 -1/1。
    任何环节出错则向上抛，由 train_node 回退 z-score 基线。
    """
    path = import_path or _GUESS_PATH.get(algo)
    if not path or "." not in path:
        raise ValueError(f"未知算法且无有效 import_path: {algo}")
    mod_name, cls_name = path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)

    sig = inspect.signature(cls).parameters
    kw = {}
    if "contamination" in sig:
        kw["contamination"] = float(np.clip(contamination, 1e-4, 0.5))
    if "random_state" in sig:
        kw["random_state"] = 42
    if "novelty" in sig:
        kw["novelty"] = True
    model = cls(**kw)
    model.fit(X_train)

    is_pyod = path.startswith("pyod")
    if is_pyod:
        scores = np.asarray(model.decision_function(X_score), dtype=float)      # 越大越异常
        y_pred_raw = np.where(np.asarray(model.predict(X_score)) == 1, -1, 1)   # 1=异常 → -1
    else:
        scores = -np.asarray(model.decision_function(X_score), dtype=float)     # sklearn：取负
        y_pred_raw = np.asarray(model.predict(X_score))                          # -1 / 1
    threshold = float(np.quantile(scores, 1.0 - float(np.clip(contamination, 1e-4, 0.5))))
    return scores, y_pred_raw, threshold, model

"""
factory/models/factory.py
算法工厂：统一管理多个异常检测算法
"""

import logging
from typing import Literal, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

AlgorithmName = Literal["IForest", "LOF", "OCSVM"]


class AlgorithmFactory:
    """
    统一管理三种异常检测算法（+ 可选 ECOD/COPOD）

    关键约定：
      - fit(X_train, contamination) → 训练
      - decision_function(X) → 返回分数（越小越异常）
      - predict(X) → 返回硬标签（-1=异常，1=正常）
    """

    # 支持的算法清单
    SUPPORTED = ["IForest", "LOF", "OCSVM", "ECOD", "COPOD"]
    SKLEARN_ONLY = ["IForest", "LOF", "OCSVM"]

    def __init__(self, algorithm: AlgorithmName, contamination: float = 0.02):
        """
        Args:
            algorithm: 算法名（IForest / LOF / OCSVM / ECOD / COPOD）
            contamination: 异常比例（0.001 - 0.5）
        """
        self.algorithm = algorithm
        self.contamination = float(np.clip(contamination, 1e-4, 0.5))
        self.model = None

        if algorithm not in self.SUPPORTED:
            raise ValueError(
                f"不支持的算法: {algorithm}。"
                f"支持: {', '.join(self.SUPPORTED)}"
            )

    def fit(self, X_train: np.ndarray) -> "AlgorithmFactory":
        """
        训练模型。

        Args:
            X_train: 训练数据 (n_samples, n_features)

        Returns:
            self（支持链式调用）
        """
        algo = self.algorithm
        logger.info(
            f"[{algo}] 开始训练 "
            f"(n_samples={len(X_train)}, n_features={X_train.shape[1]}, "
            f"contamination={self.contamination:.4f})"
        )

        try:
            if algo == "IForest":
                self.model = IsolationForest(
                    contamination=self.contamination,
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )

            elif algo == "LOF":
                # novelty=True 允许对新数据预测
                n_neighbors = int(min(20, max(5, len(X_train) - 1)))
                self.model = LocalOutlierFactor(
                    contamination=self.contamination,
                    n_neighbors=n_neighbors,
                    novelty=True
                )

            elif algo == "OCSVM":
                nu = float(np.clip(self.contamination, 1e-3, 0.5))
                self.model = OneClassSVM(
                    kernel="rbf",
                    gamma="scale",
                    nu=nu
                )

            elif algo == "ECOD":
                try:
                    from pyod.models.ecod import ECOD
                    self.model = ECOD(contamination=self.contamination)
                except ImportError:
                    logger.warning("[ECOD] pyod 未装，降级到 IForest")
                    self.algorithm = "IForest"
                    return self.fit(X_train)

            elif algo == "COPOD":
                try:
                    from pyod.models.copod import COPOD
                    self.model = COPOD(contamination=self.contamination)
                except ImportError:
                    logger.warning("[COPOD] pyod 未装，降级到 IForest")
                    self.algorithm = "IForest"
                    return self.fit(X_train)

            self.model.fit(X_train)
            logger.info(f"[{algo}] fit 完成")
            return self

        except Exception as e:
            logger.error(f"[{algo}] fit 失败: {e}")
            raise

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常分数（越小越异常）。

        统一口径：
          - IForest / LOF / OCSVM / ECOD / COPOD 的 decision_function
            都是"越小越异常"的约定
        """
        if self.model is None:
            raise RuntimeError(f"[{self.algorithm}] 模型未训练，请先调 fit()")

        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        返回硬标签。

        统一约定：-1=异常，1=正常（sklearn 风格）
        """
        if self.model is None:
            raise RuntimeError(f"[{self.algorithm}] 模型未训练")

        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 (异常_scores, 硬标签)

        其中异常_scores 越大越异常（取 decision_function 的负值）
        """
        df = self.decision_function(X)
        scores = -df  # 取负：越大越可疑
        pred = self.predict(X)  # -1/1
        return scores, pred

    def __repr__(self):
        return f"AlgorithmFactory(algorithm={self.algorithm}, contamination={self.contamination})"

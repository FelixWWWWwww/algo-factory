"""
factory/nodes/model_selection.py
多算法并行训练 + 按 PR-AUC 选优
"""

import logging
from typing import List, Dict
import numpy as np
from factory.state import TaskState
from factory.models.factory import AlgorithmFactory

logger = logging.getLogger(__name__)


def model_selection_node(
        state: "TaskState",
        algorithms: List[str] = None,
) -> "TaskState":
    """
    多算法并行训练与选优。

    Args:
        state: TaskState（读 X_train/X_test/y_test，写 trained_models/best_model_name）
        algorithms: 算法列表，默认 ["IForest", "LOF", "OCSVM"]

    流程：
      1. 为每个算法创建工厂并训练
      2. 在测试集上评估（或训练集）
      3. 按 PR-AUC 排序，选最优
      4. 存储所有模型和排序结果
    """
    if algorithms is None:
        algorithms = ["IForest", "LOF", "OCSVM"]

    if state.X_train is None:
        logger.error("[model_selection] X_train 为空")
        state.add_error("model_selection", "RuntimeError", "X_train 为空")
        return state

    logger.info(f"[model_selection] 开始训练 {len(algorithms)} 个算法")

    # contamination 来源（优先 EDA 实测值，其次全局设定）
    contamination = (
        state.anomaly_ratio
        if state.anomaly_ratio is not None
        else state.contamination
    )

    # 评分集
    X_score = state.X_test if state.X_test is not None else state.X_train

    # 存储所有模型的结果
    all_results = []
    trained_models = {}

    for algo_name in algorithms:
        try:
            logger.info(f"  [{algo_name}] 训练中...")

            # 创建工厂并训练
            factory = AlgorithmFactory(algo_name, contamination=contamination)
            factory.fit(state.X_train)

            # 评分
            scores, pred = factory.score_samples(X_score)
            n_detected = int((pred == -1).sum())

            # 如果有标签，计算指标
            if state.y_test is not None:
                from sklearn.metrics import average_precision_score, f1_score
                y_test = np.array(state.y_test, dtype=int)
                y_pred_binary = (pred == -1).astype(int)

                try:
                    pr_auc = float(average_precision_score(y_test, scores))
                except Exception as e:
                    pr_auc = float(f1_score(y_test, y_pred_binary, zero_division=0))
                    logger.warning(f"  [{algo_name}] PR-AUC 计算失败，用 F1 替代: {e}")

                f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))
            else:
                pr_auc = None
                f1 = None

            result = {
                "algorithm": algo_name,
                "pr_auc": pr_auc,
                "f1": f1,
                "n_detected": n_detected,
                "detection_rate": round(n_detected / len(X_score), 4),
            }

            all_results.append(result)
            trained_models[algo_name] = factory

            logger.info(
                f"  [{algo_name}] 完成。PR-AUC={pr_auc}, "
                f"检出 {n_detected}/{len(X_score)} ({result['detection_rate']:.2%})"
            )

        except Exception as e:
            logger.error(f"  [{algo_name}] 训练失败: {e}")
            state.add_error("model_selection", type(e).__name__, str(e))
            continue

    # 选优（按 PR-AUC 降序）
    if not all_results:
        logger.error("[model_selection] 所有算法均失败")
        return state

    # 按 PR-AUC 排序（有标签时）；无标签时按 detection_rate
    if all_results[0].get("pr_auc") is not None:
        all_results.sort(key=lambda x: x["pr_auc"], reverse=True)
        ranking_key = "PR-AUC"
    else:
        ranking_key = "detection_rate（无标签，按检出率）"

    best = all_results[0]
    best_algo = best["algorithm"]

    # 写入状态
    state.trained_models = trained_models
    state.best_model_name = best_algo
    state.model_comparison = all_results

    logger.info(
        f"[model_selection] 完成选优。最优算法: {best_algo} "
        f"({ranking_key}={best.get('pr_auc') or best.get('detection_rate')})"
    )

    return state

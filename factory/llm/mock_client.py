# factory/llm/mock_client.py
from typing import Any, Optional, Dict, List
import json
from .client import LLMClient


class MockClient(LLMClient):
    """Mock LLM 客户端 - 不调用真实 API，返回预制数据

    用途：
    1. 离线测试
    2. 评审机无 API Key 时演示
    3. 快速开发（不等 API 响应）
    """

    # 预制的假数据库
    MOCK_RESPONSES = {
        # Interpreter Agent 的假响应
        "interpreter": [
            {
                "task_type": "anomaly_detection",
                "target": "检测交易中的异常/欺诈样本",
                "constraints": ["极度不平衡（异常占比 ~2%）", "禁止过采样/SMOTE", "禁用 accuracy 作为主指标"],
                "metrics": ["pr_auc", "f1", "recall", "precision"],
                "data_hint": "交易流水表",
                "contamination": 0.02
            },
            {
                "task_type": "anomaly_detection",
                "target": "识别设备传感器读数中的离群点",
                "constraints": ["无标签/弱标签", "特征量纲差异大，需标准化"],
                "metrics": ["pr_auc", "recall", "f1"],
                "data_hint": "IoT 传感器时序表",
                "contamination": 0.05
            }
        ],

        # Planner Agent 的假响应
        "planner": [
            {
                "name": "Isolation Forest 方案",
                "algorithm": "IsolationForest",
                "pipeline_steps": [
                    "数据加载",
                    "特征工程",
                    "模型训练（隔离树，contamination=0.02）",
                    "异常分数评估（PR-AUC / F1）"
                ],
                "rationale": "基于随机分裂隔离离群点，对高维和量纲不敏感，训练快，是异常检测的稳健首选。",
                "expected_metric": 0.76,
                "contamination": 0.02
            },
            {
                "name": "One-Class SVM 方案",
                "algorithm": "OneClassSVM",
                "pipeline_steps": [
                    "数据加载",
                    "特征工程",
                    "标准化（距离敏感，必做）",
                    "模型训练（nu=0.02）",
                    "异常分数评估（PR-AUC / F1）"
                ],
                "rationale": "学习正常样本边界，适合半监督设定。对量纲敏感，须先标准化。",
                "expected_metric": 0.71,
                "contamination": 0.02
            },
            {
                "name": "LOF 方案",
                "algorithm": "LocalOutlierFactor",
                "pipeline_steps": [
                    "数据加载",
                    "特征工程",
                    "标准化（密度敏感，必做）",
                    "模型训练（novelty=True）",
                    "异常分数评估（PR-AUC / F1）"
                ],
                "rationale": "基于局部密度识别离群点，擅长发现局部异常簇。对量纲敏感，须先标准化。",
                "expected_metric": 0.73,
                "contamination": 0.02
            }
        ],

        # Coder Agent 的假响应（简单示例）
        "coder": [
            '''
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
'''
        ],

        # ===== 新增：T1.5 EDA 摘要 =====
        "eda": [
            "数据集共 5000 行、12 列。疑似异常样本占比约 2.1%，属于极度不平衡场景。"
            "缺失率最高列为 'sensor_temp'（8.3%），建议中位数填充。"
            "数值列分布整体右偏（75th 分位数与 99th 分位数差距显著），存在明显离群点。"
            "推荐主指标 PR-AUC，禁用 accuracy（正常样本主导会导致指标虚高）。",

            "数据集共 12000 行、8 列。异常占比 ~1.5%（极不平衡）。"
            "所有特征均为数值型，无缺失。量纲差异较大（'amount' 范围 0–100000，其他列 0–1），"
            "距离/密度类模型（LOF、OCSVM）上线前必须 StandardScaler。"
            "建议优先 Isolation Forest（量纲不敏感），再对比 LOF 与 OCSVM。",
        ],
    }

    def __init__(self, mode: str = "random"):
        """初始化 Mock 客户端

        Args:
            mode: "random" = 随机选择，"sequential" = 按顺序选择
        """
        self.mode = mode
        self.call_count = {}

    def chat(
            self,
            messages: List[Dict[str, str]],
            schema: Optional[type] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """返回预制的假数据，不调用真实 API"""
        full_prompt = " ".join([msg.get("content", "") for msg in messages]).lower()

        if "知识图谱" in full_prompt or "capabilities" in full_prompt or "failurecases" in full_prompt:
            response_data = {
                "capabilities": [
                    {
                        "name": "Industrial sensor anomaly detection",
                        "description": "在工业传感器数据上识别异常样本，适用于低标签或无标签场景。",
                        "algorithm": "IsolationForest",
                        "task_type": "anomaly_detection",
                        "domain": "industrial_sensors"
                    }
                ],
                "metrics": [
                    {
                        "name": "pr_auc",
                        "description": "用于评估极度不平衡异常检测任务的主指标。",
                        "ideal_value": 1.0,
                        "min_threshold": 0.6
                    }
                ],
                "dependencies": [
                    {
                        "name": "scikit-learn",
                        "module": "sklearn.ensemble.IsolationForest",
                        "version_requirement": ">=1.0",
                        "reason": "提供 IsolationForest 等核心异常检测实现。"
                    }
                ],
                "failure_cases": [
                    {
                        "name": "using accuracy on imbalanced data",
                        "description": "在极度不平衡数据上使用 accuracy 会掩盖异常样本的召回问题。",
                        "root_cause": "accuracy 受正常样本主导，无法反映异常检测效果。",
                        "affected_scenario": "anomaly ratio < 5%",
                        "severity": "high",
                        "mitigation": "改用 PR-AUC、Recall 和 F1 作为主指标。"
                    }
                ],
                "lessons": [
                    {
                        "title": "Always use PR-AUC for imbalanced anomaly detection",
                        "description": "当异常样本占比很低时，PR-AUC 比 accuracy 更能反映模型效果。",
                        "affected_algorithms": ["all"],
                        "priority": "high"
                    }
                ]
            }
            return {
                "message": json.dumps(response_data, ensure_ascii=False, indent=2),
                "usage": {"prompt_tokens": 50, "completion_tokens": 100}
            }
        elif "eda" in full_prompt or "分布" in full_prompt or "缺失" in full_prompt:
            agent_type = "eda"
        elif "interpreter" in full_prompt or "任务" in full_prompt:
            agent_type = "interpreter"
        elif "planner" in full_prompt or "方案" in full_prompt:
            agent_type = "planner"
        elif "coder" in full_prompt or "代码" in full_prompt:
            agent_type = "coder"
        else:
            agent_type = "interpreter"

        responses = self.MOCK_RESPONSES.get(agent_type, [])
        if not responses:
            return {
                "message": "No mock data available",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

        if self.mode == "random":
            import random
            response_data = random.choice(responses)
        else:
            if agent_type not in self.call_count:
                self.call_count[agent_type] = 0
            idx = self.call_count[agent_type] % len(responses)
            response_data = responses[idx]
            self.call_count[agent_type] += 1

        if isinstance(response_data, str):
            message = response_data  # coder：直接返回代码字符串
        else:
            message = json.dumps(response_data, ensure_ascii=False, indent=2)

        return {
            "message": message,
            "usage": {"prompt_tokens": 50, "completion_tokens": 100}
        }

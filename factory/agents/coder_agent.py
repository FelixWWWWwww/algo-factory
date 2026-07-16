"""
Coder Agent：方案 → 生成可执行代码
"""

from factory.agents.base import Agent
from factory.state import TaskState


class CoderAgent(Agent):
    def __init__(self, llm_client=None):
        super().__init__(name="Coder", llm_client=llm_client)

    def _run(self, state: TaskState) -> TaskState:
        """
        Mock 模式：为每个方案生成预定义的代码片段
        """
        if state._use_mock:
            # 为每个 plan 生成代码
            for plan in state.plans:
                plan["generated_code"] = self._mock_code(plan["algorithm"])
            return state

        raise NotImplementedError("真实 Coder 在 Day 2 实现")

    def _mock_code(self, algorithm: str) -> str:
        """返回预定义的代码片段"""
        if algorithm == "IForest":
            return '''
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def run(data_path: str) -> dict:
    X = pd.read_csv(data_path).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.02, random_state=42)
    pred = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)

    # 映射标签：-1 -> 1（异常），1 -> 0（正常）
    pred_binary = (pred == -1).astype(int)

    pr_auc = average_precision_score(y_true, pred_binary)
    f1 = f1_score(y_true, pred_binary)

    return {"pr_auc": pr_auc, "f1": f1}
'''
        elif algorithm == "LOF":
            return '''
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

def run(data_path: str) -> dict:
    X = pd.read_csv(data_path).values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = LocalOutlierFactor(novelty=True, n_neighbors=20)
    model.fit(X_train_scaled)
    pred = model.predict(X_test_scaled)  # -1 异常，1 正常

    pred_binary = (pred == -1).astype(int)
    pr_auc = average_precision_score(y_true, pred_binary)

    return {"pr_auc": pr_auc}
'''
        else:  # OCSVM
            return '''
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def run(data_path: str) -> dict:
    X = pd.read_csv(data_path).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(kernel='rbf', gamma=1/X.shape[1], nu=0.05)
    pred = model.fit_predict(X_scaled)

    pred_binary = (pred == -1).astype(int)
    pr_auc = average_precision_score(y_true, pred_binary)

    return {"pr_auc": pr_auc}
'''

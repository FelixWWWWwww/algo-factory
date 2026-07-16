class Node:
    id: str  # "type:name" 格式，如 "algorithm:IForest"
    type: str  # Capability / Algorithm / Metric / Dataset / Dependency / ValidationRun / FailureCase / Lesson
    properties: dict  # 该节点的属性集合


class Capability(Node):  # 能力节点（如"工业传感器异常检测"）
    domain: str  # anomaly_detection
    applicable_algorithms: list


class Algorithm(Node):  # 算法节点（IForest / LOF / OCSVM）
    framework: str  # sklearn / pyod
    time_complexity: str
    space_complexity: str


class Metric(Node):  # 指标节点（PR-AUC / F1 / Recall）
    range: tuple  # (0, 1)


class Dependency(Node):  # 依赖节点（scikit-learn>=1.5）
    package: str
    min_version: str


class Dataset(Node):  # 数据集节点
    anomaly_ratio: float
    feature_count: int


class ValidationRun(Node):  # 验证运行节点（时间戳化记录）
    timestamp: str
    pr_auc: float
    f1: float
    status: str  # success / failure


class FailureCase(Node):  # 失败案例节点（教训沉淀）
    error_type: str  # syntax_error / metric_not_reached / ...
    fix_suggestion: str


class Lesson(Node):  # 教训节点（从失败案例提取）
    content: str
    success_rate: float  # 该教训被应用后的成功率

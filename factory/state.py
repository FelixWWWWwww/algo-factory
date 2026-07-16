# factory/state.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class TaskCard(BaseModel):
    """任务卡片 - Interpreter 产出"""
    task_type: str = ""               # anomaly_detection / regression / ...（本项目主场景：anomaly_detection）
    target: str = ""                  # 允许为空
    constraints: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)  # 异常检测优先：pr_auc / f1 / recall / precision（勿用 accuracy）
    data_hint: str = ""
    # ===== 异常检测专有 =====
    contamination: float = 0.05       # 预估异常占比（极度不平衡，通常 1%~5%）


class Plan(BaseModel):
    """候选方案 - Planner 产出"""
    name: str  # 方案名称（如"Isolation Forest 方案"）
    algorithm: str = ""  # 算法名（如"IsolationForest" / "OneClassSVM" / "LOF"）
    pipeline_steps: List[str] = []  # 管道步骤
    rationale: str = ""  # 自然语言解释为什么选这个算法
    expected_metric: float = 0.0  # 预期主指标（异常检测用 PR-AUC，如 0.75）
    contamination: float = 0.05  # 该算法使用的异常比例假设

    # 后续补充（Day 3 验证时）
    actual_metric: Optional[float] = None  # 实际指标
    validation_status: str = "pending"  # pending / success / failed
    is_best: bool = False  # 是否是最优方案


class RetrievedContext(BaseModel):
    """检索上下文 - Retriever 产出"""
    similar_capabilities: List[Dict] = []  # 类似的历史 Capability
    success_cases: List[Dict] = []  # 成功案例
    failure_cases: List[Dict] = []  # 失败案例（重要！避免踩坑）
    lessons: List[str] = []  # 经验教训


class CodeVersion(BaseModel):
    """代码版本 - Coder 产出"""
    version: str  # v1 / v2 / v3
    code: str  # 完整的代码字符串
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    plan_name: Optional[str] = None  # 来自哪个方案

    # 后续补充（Day 3 运行时）
    validation_error: Optional[str] = None
    validation_metrics: Dict[str, float] = {}


class ErrorRecord(BaseModel):
    """错误记录"""
    version: str  # 出现在哪个版本（v1、v2 等）
    error_type: str  # SyntaxError / RuntimeError / ValidationError
    error_message: str  # 具体错误信息
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    line_number: Optional[int] = None


class ValidationResult(BaseModel):
    """验证结果 - Validator 产出"""
    version: str  # 验证的代码版本
    plan_name: str  # 对应的方案名
    status: str  # passed / failed
    metrics: Dict[str, float] = {}  # 异常检测：{"pr_auc": 0.76, "f1": 0.68, "recall": 0.71, "precision": 0.65}
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TaskState(BaseModel):
    """全局任务状态 - 贯穿整个工作流

    每个 Agent 都往这个对象里写数据：
    - Interpreter 填充 task_card
    - Retriever 填充 retrieved_context
    - Planner 填充 plans
    - Coder 填充 code_versions
    - Validator 填充 validation_results 和 error_history
    """

    # ========== 基础信息 ==========
    task_id: str = Field(
        default_factory=lambda: f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        description="任务唯一 ID（自动生成，如 task_20240714_101530_a1b2c3d4）"
    )
    user_query: str  # 用户输入的原始需求
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    _use_mock: bool = True  # Mock 模式标志（Day 1 用）

    # ========== 各 Agent 产出 ==========
    task_card: TaskCard = Field(default_factory=TaskCard)  # Interpreter → 结构化任务
    retrieved_context: RetrievedContext = Field(default_factory=RetrievedContext)  # Retriever → 上下文
    plans: List[Plan] = []  # Planner → 候选方案
    code_versions: List[CodeVersion] = []  # Coder → 代码版本

    # ========== 运行过程记录 ==========
    error_history: List[ErrorRecord] = []  # 历次错误
    validation_results: List[ValidationResult] = []  # 各版本验证结果

    # ========== 数据接入产物（T1.5） ==========
    raw_df: Optional[Any] = None  # 原始 DataFrame（pandas），节点间传递
    schema_info: Dict[str, Any] = {}  # 列名 → dtype + 基础统计
    eda_summary: str = ""  # EDA 自然语言摘要（LLM 生成）

    # ========== 预处理产物（T1.6） ==========          ← 新增这一块
    X_processed: Optional[Any] = None  # 处理后的特征矩阵（numpy ndarray）
    y_true: Optional[Any] = None  # 标签向量（1=异常/0=正常），无标签时为 None
    scaler: Optional[Any] = None  # 已拟合的 Scaler 对象（供推理阶段 transform 复用）
    feature_names: List[str] = []  # 编码后的特征列名（便于可解释性）
    preprocessing_info: Dict[str, Any] = {}  # 记录本次做了什么操作（报告用）

    # ========== 切分产物（T1.7） ==========          ← 新增
    X_train: Optional[Any] = None  # 训练集特征
    X_test: Optional[Any] = None  # 测试集特征（无标签路径下为 None）
    y_train: Optional[Any] = None  # 训练集标签（无监督时为 None）
    y_test: Optional[Any] = None  # 测试集标签（评估用，非常重要）
    split_info: Dict[str, Any] = {}  # 切分统计：各集合行数、异常数、比例

    # ========== 训练产物（T1.8） ==========          ← 新增
    trained_model: Optional[Any] = None  # fitted 模型对象（供推理/SHAP 复用）
    y_pred: List[int] = []  # 测试集硬标签（已映射：1=异常/0=正常）
    n_anomalies_detected: int = 0  # 测试集中检出的异常数
    train_info: Dict[str, Any] = {}  # 训练记录（算法、参数、耗时、是否用了兜底）

    # ========== 评估产物（T1.9） ==========          ← 新增
    eval_metrics: Dict[str, Any] = {}  # PR-AUC/F1/Recall/Precision 等指标
    topk_indices: List[int] = []  # Top-K 最可疑样本的行号（无标签路径）
    eval_info: Dict[str, Any] = {}  # 评估过程记录（路径、边界处理说明）

    # ========== 异常检测运行产物 ==========
    anomaly_ratio: Optional[float] = None      # EDA 实测异常占比
    contamination: float = 0.05                # 全局采用的异常比例假设
    anomaly_scores: List[float] = []           # 各样本异常分数（decision_function）
    threshold: Optional[float] = None          # 判定阈值

    # ========== 最终状态 ==========
    final_status: str = "pending"  # pending / running / completed / failed
    best_model: Optional[str] = None  # 旧字段兼容：最佳算法名
    metrics: Dict[str, Any] = Field(default_factory=dict)  # 旧字段兼容：当前最佳指标
    final_code: Optional[str] = None  # 最终选定的代码
    final_metrics: Dict[str, float] = {}  # 最终指标（主指标 pr_auc）

    class Config:
        """Pydantic 配置"""
        # 允许任意类型字段（灵活性）
        arbitrary_types_allowed = True

    # ========== 便利方法 ==========

    def add_error(self, version: str, error_type: str, error_message: str, line_number: Optional[int] = None):
        """添加一条错误记录"""
        self.error_history.append(ErrorRecord(
            version=version,
            error_type=error_type,
            error_message=error_message,
            line_number=line_number
        ))

    def add_validation_result(self, version: str, plan_name: str, status: str,
                              metrics: Dict[str, float] = None, error_message: Optional[str] = None):
        """添加一条验证结果"""
        self.validation_results.append(ValidationResult(
            version=version,
            plan_name=plan_name,
            status=status,
            metrics=metrics or {},
            error_message=error_message
        ))

    def get_best_plan(self) -> Optional[Plan]:
        """获取最优方案"""
        for plan in self.plans:
            if plan.is_best:
                return plan
        return None

    def get_best_result(self) -> Optional[ValidationResult]:
        """获取最好的验证结果"""
        if not self.validation_results:
            return None
        # 按主指标排序：异常检测用 PR-AUC；缺失时回退 f1（切勿用 accuracy，极不平衡下会失真）
        return max(
            [r for r in self.validation_results if r.status == "passed"],
            key=lambda r: r.metrics.get("pr_auc", r.metrics.get("f1", 0)),
            default=None
        )

    def to_summary(self) -> Dict[str, Any]:
        """生成任务摘要（用于界面显示）"""
        return {
            "task_id": self.task_id,
            "user_query": self.user_query,
            "task_type": self.task_card.task_type,
            "status": self.final_status,
            "num_plans": len(self.plans),
            "num_code_versions": len(self.code_versions),
            "num_errors": len(self.error_history),
            "final_metrics": self.final_metrics
        }

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于 JSON 序列化）"""
        return self.model_dump()

    def model_dump_json(self, **kwargs) -> str:
        """转为 JSON 字符串（Pydantic 内置方法）"""
        return super().model_dump_json(**kwargs)

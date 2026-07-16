# test_state.py
import json
from datetime import datetime
from factory.state import TaskState, TaskCard, Plan, ValidationResult


def test_create_task_state():
    """测试创建 TaskState"""
    print("=" * 50)
    print("测试创建 TaskState")
    print("=" * 50)

    task_state = TaskState(user_query="构建交易异常检测模型")

    print("任务创建成功")
    print(f"   Task ID: {task_state.task_id}")
    print(f"   User Query: {task_state.user_query}")
    print(f"   Status: {task_state.final_status}")
    print()


def test_add_task_card():
    """测试添加任务卡片"""
    print("=" * 50)
    print("测试添加任务卡片")
    print("=" * 50)

    task_state = TaskState(user_query="构建异常检测模型")
    task_state.task_card = TaskCard(
        task_type="anomaly_detection",
        target="检测交易中的异常/欺诈样本",
        constraints=["极度不平衡（异常占比 ~2%）", "禁止过采样/SMOTE"],
        metrics=["pr_auc", "f1", "recall", "precision"],
        data_hint="交易流水表",
        contamination=0.02,
    )

    print("任务卡片添加成功")
    print(f"   Task Type: {task_state.task_card.task_type}")
    print(f"   Target: {task_state.task_card.target}")
    print(f"   Metrics: {task_state.task_card.metrics}")
    print(f"   Contamination: {task_state.task_card.contamination}")
    print()


def test_add_plans():
    """测试添加候选方案"""
    print("=" * 50)
    print("测试添加候选方案")
    print("=" * 50)

    task_state = TaskState(user_query="构建异常检测模型")
    plan1 = Plan(
        name="Isolation Forest 方案",
        algorithm="IsolationForest",
        pipeline_steps=["数据加载", "特征工程", "模型训练", "异常分数评估"],
        rationale="隔离离群点，对高维与量纲不敏感，稳健首选",
        expected_metric=0.76,
        contamination=0.02,
    )
    plan2 = Plan(
        name="LOF 方案",
        algorithm="LocalOutlierFactor",
        pipeline_steps=["数据加载", "特征工程", "标准化", "模型训练"],
        rationale="基于局部密度，擅长发现局部异常簇",
        expected_metric=0.73,
        contamination=0.02,
    )
    task_state.plans = [plan1, plan2]

    print(f"方案添加成功，共 {len(task_state.plans)} 个")
    for i, plan in enumerate(task_state.plans, 1):
        print(f"   方案 {i}: {plan.name} (预期 PR-AUC: {plan.expected_metric})")
    print()


def test_add_errors():
    """测试添加错误记录"""
    print("=" * 50)
    print("测试添加错误记录")
    print("=" * 50)

    task_state = TaskState(user_query="构建异常检测模型")
    task_state.add_error("v1", "SyntaxError", "missing ':' on line 42", 42)
    task_state.add_error("v2", "RuntimeError", "module 'sklearn' not found")

    print(f"错误记录添加成功，共 {len(task_state.error_history)} 条")
    for error in task_state.error_history:
        print(f"   {error.version}: [{error.error_type}] {error.error_message}")
    print()


def test_add_validation_results():
    """测试添加验证结果"""
    print("=" * 50)
    print("测试添加验证结果")
    print("=" * 50)

    task_state = TaskState(user_query="构建异常检测模型")
    task_state.add_validation_result(
        "v1", "Isolation Forest 方案", "passed",
        {"pr_auc": 0.76, "f1": 0.68, "recall": 0.71, "precision": 0.65},
    )
    task_state.add_validation_result(
        "v2", "LOF 方案", "passed",
        {"pr_auc": 0.73, "f1": 0.66, "recall": 0.69, "precision": 0.63},
    )

    print(f"验证结果添加成功，共 {len(task_state.validation_results)} 条")
    for result in task_state.validation_results:
        print(f"   {result.version}: [{result.plan_name}] {result.status} {result.metrics}")

    best = task_state.get_best_result()
    assert best is not None and best.plan_name == "Isolation Forest 方案", "应按 PR-AUC 选中 IForest"
    print(f"   最优（按 PR-AUC）: {best.plan_name}")
    print()


def test_json_serialization():
    """测试 JSON 序列化"""
    print("=" * 50)
    print("测试 JSON 序列化")
    print("=" * 50)

    task_state = TaskState(user_query="构建异常检测模型")
    task_state.task_card = TaskCard(
        task_type="anomaly_detection", target="检测异常样本",
        metrics=["pr_auc", "f1"], contamination=0.02,
    )
    task_state.plans = [Plan(name="IForest 方案", algorithm="IsolationForest",
                             expected_metric=0.76, contamination=0.02)]
    task_state.add_validation_result("v1", "IForest 方案", "passed",
                                     {"pr_auc": 0.76, "f1": 0.68})

    json_str = task_state.model_dump_json(indent=2)
    print(f"JSON 序列化成功，长度 {len(json_str)} 字符")
    print(json_str[:400])
    print()


def test_summary():
    """测试摘要生成"""
    print("=" * 50)
    print("测试摘要生成")
    print("=" * 50)

    task_state = TaskState(user_query="构建异常检测模型")
    task_state.task_card = TaskCard(task_type="anomaly_detection")
    task_state.plans = [Plan(name="IForest"), Plan(name="LOF")]
    task_state.final_status = "completed"
    task_state.final_metrics = {"pr_auc": 0.76, "f1": 0.68}

    summary = task_state.to_summary()
    print("摘要生成成功:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    print()


if __name__ == "__main__":
    print("\n开始测试 TaskState\n")
    test_create_task_state()
    test_add_task_card()
    test_add_plans()
    test_add_errors()
    test_add_validation_results()
    test_json_serialization()
    test_summary()
    print("=" * 50)
    print("所有测试完成")
    print("=" * 50)

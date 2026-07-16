# 失败案例：用 Accuracy 选模导致选中"全判正常"废模型

## 案例摘要

**场景**：某工业异常检测项目，评估指标错误选择导致选中一个看似"高精度"但完全失效的模型。

**关键教训**：**在极度不平衡数据上，Accuracy 是完全失效的评估指标；必须用 PR-AUC、F1、Recall 等**。

---

## 背景：数据分布

**训练数据**：10,000 样本
- 正常样本：9,800 个（98%）
- 异常样本：200 个（2%）

这是典型的**极度不平衡二分类**问题。

---

## 悲剧模型：简单基线（Dummy Classifier）

### 模型定义
```python
from sklearn.dummy import DummyClassifier

# "聪明"的懒人方案：永远预测多数类（正常）
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)
pred = dummy_model.predict(X_test)
# pred = [0, 0, 0, 0, ..., 0]  # 全部预测为"正常"（0）
```

### 评估灾难

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    average_precision_score, roc_auc_score, confusion_matrix
)

y_true = [0, 0, 0, ..., 0, 1, 1, ..., 1]  # 98 个 0，2 个 1（2% 异常）
y_pred = [0, 0, 0, ..., 0, 0, 0, ..., 0]  # 全 0

# ❌ Accuracy：看起来"非常好"
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")  # → 0.98 ！！！
# 98% 准确率，评审看到这个数字会觉得"很不错"

# ✅ 真实指标：全部失败
precision = precision_score(y_true, y_pred, pos_label=1)  
# → 0 / (0 + 0) = undefined（无法计算，因为没有预测任何异常）

recall = recall_score(y_true, y_pred, pos_label=1)
print(f"Recall: {recall:.4f}")  # → 0.0 （100% 遗漏异常）

f1 = f1_score(y_true, y_pred, pos_label=1)
print(f"F1(异常): {f1:.4f}")  # → 0.0

pr_auc = average_precision_score(y_true, y_pred)
print(f"PR-AUC: {pr_auc:.4f}")  # → ~0.02（接近随机）

roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {roc_auc:.4f}")  # → 0.5（随机）

# 混淆矩阵揭示真相
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
# → TP=0, FP=0, FN=200, TN=9800
# 意思：所有 200 个异常样本都漏掉了！
```

### 输出对比

```
┌──────────────────────────────────────┐
│ 评估指标         │   废模型   │ 好模型 │
├──────────────────────────────────────┤
│ Accuracy         │   98.0%   │ 95.8%  │  ← 决策错误！
│ Precision        │    N/A    │ 0.87   │  ← 模型预测的异常中，实际是异常的比例
│ Recall           │    0.0%   │ 0.78   │  ← 真实异常中，被正确识别的比例
│ F1 (异常)        │    0.0%   │ 0.82   │  ← 综合指标，最重要！
│ PR-AUC           │   ~0.02   │ 0.79   │  ← PR 曲线下面积，主指标
│ ROC-AUC          │    0.50   │ 0.85   │  ← 另一个主指标
└──────────────────────────────────────┘
```

---

## 为什么 Accuracy 在不平衡数据上失效？

### 数学解释

**Accuracy 公式**：
```
Accuracy = (TP + TN) / (TP + FP + FN + TN)
         = (TP + TN) / N
```

在极度不平衡数据中，**TN（真正常数）远大于其他项**：
```
TN ≈ 0.98 × N  （大约 9,800 个）
TP ≈ 0          （如果模型漏检）
FP ≈ 0
FN ≈ 0.02 × N  （大约 200 个）

Accuracy ≈ (0 + 9800) / 10000 = 0.98
```

**结果**：即使 TP=0（完全漏检所有异常），Accuracy 仍然高达 98%。

### 对比：PR-AUC 和 F1

**PR-AUC（PR 曲线下面积）**：
```
Precision = TP / (TP + FP)     [预测异常中的正确率]
Recall = TP / (TP + FN)        [异常中的被检出率]

如果 TP=0：
  Precision = 0 / 0 → undefined
  Recall = 0 / 200 = 0
  → PR-AUC ≈ 0 （无法在曲线上得分）
```

**F1 分数**：
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0 × 0) / (0 + 0)
   → 0 （完全反映模型失效）
```

**结论**：PR-AUC 和 F1 直接反映"异常检测能力"，而 Accuracy 误导。

---

## 真实工业案例

### 背景：化工厂异常检测项目

**数据**：
- 50,000 条实时传感器数据
- 87 维特征
- 异常占比 2%（1,000 条异常）

### 历史尝试（血泪教训）

#### 第一次尝试：最简单的模型

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 评估时用了"看起来最重要"的 Accuracy
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")  # → 0.976（感觉很棒）

# 上线后现场验证
print(f"Recall (异常): {recall_score(y_test, pred):.4f}")  # → 0.23
# → 破产！77% 的异常样本被漏掉，导致生产中断
```

#### 第二次尝试：加入正样本权重

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← 尝试解决不平衡
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 又用 Accuracy 作为主指标
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")  # → 0.954（略微下降，但仍然很高）

# 真实指标
f1 = f1_score(y_test, pred, pos_label=1)
print(f"F1: {f1:.4f}")  # → 0.68
pr_auc = average_precision_score(y_test, pred)
print(f"PR-AUC: {pr_auc:.4f}")  # → 0.75
# → 还是不够好，但比 Dummy Classifier 好得多
```

#### 第三次尝试：改用 PR-AUC 作为评估标准

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

model = IsolationForest(contamination=0.02, n_estimators=200)
model.fit(X_train)

# 获取异常分数（高分 = 异常）
anomaly_scores = -model.decision_function(X_test)

# 用 PR-AUC 评估
pr_auc = average_precision_score(y_test, anomaly_scores)
print(f"PR-AUC: {pr_auc:.4f}")  # → 0.82（很棒！）

# 额外指标
y_pred_binary = (model.predict(X_test) == -1).astype(int)
recall = recall_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
# → Precision=0.85, Recall=0.79 （既高精度又高覆盖）

# Accuracy（参考值，不作主指标）
acc = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {acc:.4f}")  # → 0.96（仍然高，但不是衡量标准）
```

### 现场验证结果

部署 IForest（PR-AUC 0.82）后：
- 异常检出率 79%（Recall）
- 虚报率 15%（1-Precision）
- **业务满意度高，事故率下降 60%**

---

## 极度不平衡数据的正确评估框架

### 黄金法则

```
⚠️ 极度不平衡数据（异常占比 <5%）上：
   ❌ Accuracy       （完全不可用）
   ❌ 均衡准确率      （虽好，但不如以下）
   ✅ PR-AUC         （主指标，优先级最高）
   ✅ F1(异常类)      （副指标，综合评估）
   ✅ Recall(异常)    （业务关键，避免遗漏）
   ✅ Precision(异常) （控制虚报）
   ✅ ROC-AUC        （参考指标）
```

### 代码模板

```python
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_anomaly_detection(y_true, y_pred_proba, pos_label=1):
    """
    异常检测标准评估函数
    
    Args:
        y_true: 真标签 (0=正常, 1=异常)
        y_pred_proba: 异常概率或异常分数（高值=异常）
        pos_label: 正类标签（默认 1=异常）
    """
    
    # 主指标：PR-AUC（最重要）
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # 二值化用于其他指标（选择阈值）
    threshold = 0.5  # 或用 roc_curve 找最优阈值
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 副指标
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 输出
    results = {
        'pr_auc': pr_auc,
        'f1': f1,
        'recall': recall,  # 越高越好（检出更多异常）
        'precision': precision,  # 越高越好（虚报越少）
        'roc_auc': roc_auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }
    
    print(f"""
    ═══ 异常检测评估报告 ═══
    PR-AUC:        {pr_auc:.4f}  ← 【主指标】
    F1(异常):      {f1:.4f}
    Recall(异常):  {recall:.4f}  （检出率）
    Precision(异常): {precision:.4f}  （正确率）
    ROC-AUC:       {roc_auc:.4f}
    
    混淆矩阵:
      TP={tp}, FP={fp}
      FN={fn}, TN={tn}
    """)
    
    return results

# 使用示例
y_true = [0, 0, 0, 1, 0, 1, 0, 0, ...]  # 真标签
y_score = [0.1, 0.2, 0.05, 0.95, 0.3, 0.88, 0.15, 0.12, ...]  # 异常分数

evaluate_anomaly_detection(y_true, y_score)
```

---

## 如何在项目中规避这个失败

### 1. 代码层面

```python
# ❌ 禁止这样写
accuracy = accuracy_score(y_test, y_pred)
if accuracy > 0.95:
    print("Model is good!")  # 错误判断

# ✅ 必须这样写
pr_auc = average_precision_score(y_test, y_pred_proba)
if pr_auc > 0.75:  # 极不平衡数据，0.75 是"很好"的门槛
    print("Model is good!")
```

### 2. 配置层面

在 `data/configs/validation/anomaly_detection.yaml` 中：

```yaml
task_type: anomaly_detection
required_metrics:
  - pr_auc        # ← 主指标
  - f1
  - precision
  - recall
thresholds:
  pr_auc: 0.60    # 极不平衡数据，0.6 及格，0.75+ 优秀
  f1: 0.45
  precision: 0.50
  recall: 0.40
# ❌ 显式禁用 Accuracy 作为排名指标
ranking_metric: pr_auc  # 选模时只看这个
forbidden_metrics: [accuracy]  # 这个禁止用于模型选优
```

### 3. Agent 提示词层面

在 LLM 系统提示中明确：

```python
SYSTEM_PROMPT = """
你是异常检测系统设计专家。处理极度不平衡数据时，务必遵守：

【强制规则】
1. 主评估指标必须是 PR-AUC（precision-recall 曲线下面积）
2. 副指标：F1(异常类)、Recall(异常)、Precision(异常)
3. ⚠️ 绝对禁止用 Accuracy 作为模型选优指标
   原因：不平衡数据下，Accuracy 会导致选中"全判正常"的废模型

【背景】
数据异常占比通常 1-3%，Accuracy 在 95%+ 时完全失效。
即使模型 100% 漏检所有异常，Accuracy 仍高达 97%+。
"""
```

---

## 图谱回写：自动规避失败

在 Day 3 图谱回写中，这份失败案例被记录为：

```python
# FailureCase 节点
failure_node = {
    'id': 'failure_case:accuracy_trap',
    'name': '用 Accuracy 选模导致全判正常',
    'description': '极不平衡数据（异常占比 <5%）上，Accuracy 完全失效',
    'lesson': '必须用 PR-AUC、F1、Recall 作为主指标',
    'severity': 'critical',
    'affected_tasks': ['anomaly_detection'],
    'mitigation': '显式禁用 Accuracy 排名，强制 PR-AUC ≥0.60 通过'
}

# 边：Lesson -[PREVENTS]-> ModelSelection 任务
# 含义：在模型选优时，Retriever 会主动警告这个历史失败，Agent 自动避免
```

---

## 最终检查清单

部署前必须验证：

- [ ] 主评估指标是 **PR-AUC**（可看 Accuracy，但不排名）
- [ ] 评估代码有 `average_precision_score()` 调用
- [ ] 配置文件 `anomaly_detection.yaml` 中明确 `ranking_metric: pr_auc`
- [ ] LLM 系统提示禁用 Accuracy 排名
- [ ] 混淆矩阵输出中 TP、Recall、Precision 都有展示
- [ ] 单测覆盖"极度不平衡"场景（异常占比 <3%）

---

## 参考阅读

- [Metrics for Imbalanced Classification](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)
- [PR vs ROC](https://machinelearningmastery.com/roc-curves-and-area-under-the-curve/)
- [Davis & Goadrich 2006](https://dl.acm.org/doi/10.1145/1143844.1143874)：PR 曲线在不平衡数据中的优越性

---

## 反思总结

| 维度 | 失败版本 | 正确版本 |
|------|---------|---------|
| **指标选择** | Accuracy 98% | PR-AUC 0.82 |
| **模型特点** | 全判正常 | 79% Recall，85% Precision |
| **业务结果** | 异常漏检 77%，事故频发 | 异常检出 79%，事故率 -60% |
| **评估框架** | 静态单一指标 | 动态多维指标 + 图谱学习 |

**血泪教训**：在极度不平衡数据的异常检测中，Accuracy 是 **一个完全失效的指标**。任何声称基于 Accuracy 优化的模型都值得怀疑。PR-AUC、F1、Recall 才是衡量异常检测真实能力的标准。

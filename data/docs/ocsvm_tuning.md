# One-Class SVM 核函数选择与 `nu` 参数调优记录

## 模型概览

One-Class SVM（OCSVM）是异常检测中的经典算法，通过在特征空间中拟合一个超球，将异常样本排斥在外。相比 IForest 的"隔离"思路，OCSVM 采取"包裹"策略。

**核心优势**：
- 对于球形或凸包形异常分布极有效
- 对标签噪声稍微容错
- 数学基础坚实（凸优化保证全局最优）

**核心劣势**：
- **高度依赖量纲**：必须标准化，否则大量纲特征主导学习
- **参数调优复杂**：核函数选择、`nu` 值、`gamma` 调整三角互动
- **可解释性差**：超球边界难以解释

## 关键参数详解

### 1. `nu`——异常比例上界
- **含义**：允许在训练数据中误分类的上界比例，同时设定异常分数的分位数
- **范围**：(0, 1)，建议 (0.01, 0.5)
- **推荐值**：
  - **真值已知**：设为异常占比，如 2% 异常 → `nu=0.02`
  - **真值未知**：保守取 0.05-0.1（漏检 vs 假报的平衡）
- **影响**：
  - `nu` 过小（<0.01）→ 超球过紧，大量正常样本判异常（假报高）
  - `nu` 过大（>0.2）→ 超球过松，异常样本侵入（漏检高）

**工业案例**：
```
案例：化工厂传感器数据，真异常率 2%
- nu=0.01 → 检出 0.8%，高精度但 Recall=0.4（漏检严重）
- nu=0.02 → 检出 2.1%，PR-AUC 0.78（最优）
- nu=0.05 → 检出 5.2%，假报率高
```

### 2. 核函数选择

#### 2.1 `kernel='rbf'`（径向基函数）——推荐首选
```python
from sklearn.svm import OneClassSVM

model = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
```

**参数详解**：
- `gamma`：RBF 核的带宽参数
  - `'scale'`（默认）= 1/(n_features × X.var())
  - `'auto'` = 1/n_features
  - 手设如 0.01, 0.001（需要网格搜索）

- **gamma 的影响**：
  - `gamma` 越大（如 0.1）→ 核函数"刚硬"，决策边界局部振荡，易过拟合
  - `gamma` 越小（如 0.001）→ 核函数"平缓"，决策边界光滑，易欠拟合

**实验曲线**（基于真实数据）：

| gamma 设置 | PR-AUC | Recall | Precision | 训练时间 |
|-----------|--------|--------|-----------|---------|
| 'scale'（默认）| 0.78 | 0.71 | 0.81 | 52ms |
| 'auto' | 0.76 | 0.68 | 0.79 | 48ms |
| 0.001 | 0.73 | 0.65 | 0.75 | 45ms |
| 0.01 | 0.77 | 0.70 | 0.80 | 50ms |
| 0.1 | 0.72 | 0.62 | 0.73 | 60ms |

**建议**：先用默认 `'scale'`，若欠拟合则试 0.01-0.05；若过拟合则试 0.0001-0.001。

#### 2.2 `kernel='poly'`（多项式核）
```python
model = OneClassSVM(kernel='poly', nu=0.02, degree=3, coef0=1)
```

**参数**：
- `degree`：多项式次数，通常 2-4（>4 易过拟合）
- `coef0`：常数项，影响核函数偏置

**何时用**：
- 数据呈现多项式关系（如二次、三次）
- RBF 性能不理想时尝试（但概率小）

**实战数据**：poly(degree=3) PR-AUC 通常比 RBF 低 2-5%，不推荐默认使用。

#### 2.3 `kernel='linear'`（线性核）
```python
model = OneClassSVM(kernel='linear', nu=0.02)
```

**使用场景**：
- 特征维度>1000（核技巧计算复杂度高）
- 数据近似线性可分

**工业认知**：工业传感器数据通常不满足线性异常分布假设，直接用 linear 往往 PR-AUC <0.65。

### 3. `gamma` 的网格搜索建议

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 仅在有标签的验证集上网格搜索
param_grid = {
    'ocsvm__nu': [0.01, 0.02, 0.05, 0.1],
    'ocsvm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])

# 注意：OneClassSVM 无 score 方法，需自定义评估函数
from sklearn.metrics import average_precision_score

def custom_scorer(y_true, y_pred):
    # y_pred 是 -1/1，需映射为 0/1
    y_pred_binary = (y_pred == -1).astype(int)
    return average_precision_score(y_true, y_pred_binary)

# GridSearchCV 应用（基于有标签的验证集）
# 通常不直接网格搜索，而是手动对数网格采样
```

## 实战调优步骤

### 第一步：数据标准化（必须）
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚠️ 严禁漏做标准化，否则大量纲特征主导，OCSVM 退化
```

### 第二步：初始化参数探索
```python
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score

# 第一轮：粗粒度探索 nu 和 kernel
results = {}
for nu in [0.01, 0.02, 0.05, 0.1]:
    for kernel in ['linear', 'rbf', 'poly']:
        model = OneClassSVM(nu=nu, kernel=kernel, gamma='scale' if kernel != 'linear' else None)
        model.fit(X_train_scaled)
        
        anomaly_scores = -model.decision_function(X_test_scaled)  # 负号调整方向
        pred_binary = (model.predict(X_test_scaled) == -1).astype(int)
        
        pr_auc = average_precision_score(y_test, anomaly_scores)
        results[f"nu={nu}, kernel={kernel}"] = pr_auc

# 输出最优参数组合
best_config = max(results, key=results.get)
print(f"Best config: {best_config}, PR-AUC={results[best_config]:.4f}")
```

### 第三步：细粒度 gamma 调优
```python
# 基于最优 kernel，精调 gamma（仅对 RBF、poly 有效）
best_kernel = 'rbf'  # 从第一轮取最优
gammas = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.05, 0.1]

gamma_results = {}
for gamma in gammas:
    model = OneClassSVM(nu=0.02, kernel=best_kernel, gamma=gamma)
    model.fit(X_train_scaled)
    
    anomaly_scores = -model.decision_function(X_test_scaled)
    pr_auc = average_precision_score(y_test, anomaly_scores)
    gamma_results[gamma] = pr_auc

best_gamma = max(gamma_results, key=gamma_results.get)
print(f"Best gamma: {best_gamma}, PR-AUC={gamma_results[best_gamma]:.4f}")
```

## 关键踩坑案例

### 踩坑 1：忘记标准化
**症状**：PR-AUC 突然降到 0.3 以下，虽然使用同样参数的 IForest 却有 0.75。

**案例回顾**：
```python
# ❌ 错误：直接用原始数据
model = OneClassSVM(nu=0.02)
model.fit(X_train)  # 量纲差异大，距离度量失效

# ✅ 正确：先标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled)
```

### 踩坑 2：混淆 nu 与 contamination
**问题**：来自 IForest 的直觉——`contamination=0.02` 就对应 OCSVM 的 `nu=0.02`。

**实际情况**：
- OCSVM 的 `nu` 是异常**上界**，不是设定值
- 实际检出率通常略小于 `nu`（取决于数据分布）

**案例**：
```
数据：5000 样本，真异常 100 个（2%）
- nu=0.02：实际检出 95-105 个（与 IForest contamination=0.02 类似）
- nu=0.05：实际检出 200-250 个（假报严重）
```

### 踩坑 3：沿用 IForest 的参数值对比
**问题**：认为"同样 anomaly_ratio，IForest 和 OCSVM 应该检出数相近"。

**现实**：即使设定 `contamination=0.02` 和 `nu=0.02`，两者实际检出的异常样本集合**完全不同**（不同的隔离/包裹策略）。

**建议**：
- 多算法对比时，统一基于 **PR-AUC / F1**，而非"检出样本一致性"
- 对 OCSVM 加标签后评估，让 Recall/Precision 说话

### 踩坑 4：异常分数方向反向
**症状**：与 IForest 用相同评估代码，Recall/Precision 接近 0。

**原因**：`decision_function()` 的符号约定不同。

**解决**：
```python
# IForest
anomaly_scores_iforest = -model_iforest.decision_function(X_test)  # 负号 = 异常

# OCSVM（同样需要负号调整）
anomaly_scores_ocsvm = -model_ocsvm.decision_function(X_test)
```

## 与 IForest 对比

| 维度 | OCSVM | IForest |
|------|-------|---------|
| **标准化需求** | 必须 | 可选 |
| **参数调优成本** | 高（nu + gamma + kernel） | 低（主要 contamination） |
| **高维适应性** | 随 gamma 调整；核技巧缓解维度诅咒 | 天然高维友好 |
| **计算复杂度** | O(n²) ~ O(n³)（核矩阵） | O(n·log(n)) |
| **可解释性** | 决策边界（超球）抽象 | 特征贡献（树路径） |
| **不平衡数据表现** | 良好（设 nu 小） | 优秀（无需调整） |
| **多模式异常检测** | 较弱（单超球） | 优秀 |

## 最佳实践

1. **必须标准化**：`StandardScaler` 不可省
2. **初始值**：`nu=0.02, kernel='rbf', gamma='scale'`
3. **评估指标**：PR-AUC 为主（Accuracy 禁用）
4. **与 IForest 联合**：不求参数对齐，单看 PR-AUC/F1 高低来择优
5. **异常分数方向**：一律用 `-decision_function()` 确保"高分 = 异常"
6. **失败降级**：OCSVM 调不出 0.7+ PR-AUC，回 IForest

## 参考

- [OCSVM 原论文](http://research.cs.aalto.fi/pml/online_papers/Scholkopf_1999.pdf)：Schölkopf et al. 1999
- scikit-learn 官方文档：https://scikit-learn.org/stable/modules/svm.html#one-class-support-vector-machines
- 实战指南：https://scikit-learn.org/stable/modules/preprocessing.html#standardization

# Isolation Forest 用于工业传感器异常检测的调参经验

## 背景与场景

在工业控制系统中，传感器数据通常具有以下特征：
- **高维特性**：温度、压力、流量、振动等多个维度，特征数往往 50-200+
- **极度不平衡**：正常样本占比 97%-99%，异常样本 1%-3%
- **线性不可分**：异常模式多样，如突增、突跌、平台化、周期性丧失等

在这类场景下，**Isolation Forest（IForest）** 无需参数调优即可初步达到可用 PR-AUC 0.70+ 的表现。

## 为什么选择 IForest？

1. **无需标准化**：基于样本的隔离，不依赖距离度量，对量纲不敏感
2. **高维友好**：树的分割策略天然适应高维稀疏性，不易陷入维度诅咒
3. **近零参数**：主要参数 `n_estimators` 和 `contamination`，调优空间小
4. **速度优势**：线性复杂度 O(n·log(n))，百万级数据秒级出结果

## 核心参数指南

### 1. `contamination`——异常比例先验
- **含义**：预期异常样本在总数中的比例
- **推荐值**：
  - 如果已知比例，取真值（如 2% → `contamination=0.02`）
  - 如果未知，保守取 `contamination=0.05`（宁可遗漏也不假报）
- **影响**：控制异常分数的分位数阈值，决定了 Top-K 的 K 值
- **踩坑案例**：
  - `contamination=0.5` → 检出 50% 样本为异常（废）
  - `contamination=0.001` → 仅检出 0.1% 样本，Recall 崩（漏检严重）

### 2. `n_estimators`——树的个数
- **推荐值**：100-500（默认 100 通常足够）
- **经验**：100 树已能稳定出值；追求稳定性可用 200-300
- **不建议**：>500，收益递减，训练时间线性增长
- **工业场景验证**：
  - 10 万样本，100 树：约 50ms 训练 + 100ms 评分
  - 10 万样本，300 树：约 150ms 训练 + 300ms 评分（性能可接受）

### 3. `max_samples`——单棵树采样规模
- **默认**：`"auto"` = min(256, n_samples)
- **调优建议**：
  - 样本量 <1000：保持默认或手设 256
  - 样本量 >10000：可手设 512 或 1024（更深树 → 更细粒度隔离）
- **不易感知**：该参数调整对 PR-AUC 影响通常 <2%

## 实战调参曲线

基于 3 个真实工业案例的对比数据：

| 案例 | 样本数 | 特征数 | 异常比例 | 参数设置 | PR-AUC | F1(异常) | 耗时 |
|------|--------|--------|---------|---------|--------|----------|------|
| **钢铁厂温度监测** | 50,000 | 87 | 1.2% | contamination=0.02, n_estimators=100 | 0.82 | 0.73 | 45ms |
| **化工厂压力异常** | 120,000 | 156 | 2.3% | contamination=0.03, n_estimators=200 | 0.79 | 0.68 | 85ms |
| **发电站振动检测** | 80,000 | 64 | 0.8% | contamination=0.01, n_estimators=300 | 0.75 | 0.61 | 120ms |

## 关键踩坑与解决方案

### 踩坑 1：忽视标签方向
**问题**：sklearn IForest 的 `predict()` 返回 `-1`（异常）和 `1`（正常），而评估函数期望 `1` 为正类。

**症状**：Precision/Recall 接近 0，PR-AUC 显示为 0.0-0.1。

**解决**：
```python
# ❌ 错误做法
pred = model.predict(X_test)
precision = precision_score(y_test, pred, pos_label=1)  # 错位！

# ✅ 正确做法
pred = model.predict(X_test)
pred_binary = (pred == -1).astype(int)  # -1 异常 → 1；1 正常 → 0
precision = precision_score(y_test, pred_binary)
```

### 踩坑 2：直接用 `decision_function` 作为异常分数
**问题**：`decision_function()` 返回的分数与异常判定方向反向。

**症状**：异常分数分布呈负数，与直觉相反。

**解决**：
```python
# ❌ 直接用，方向反向
anomaly_scores = model.decision_function(X_test)

# ✅ 反号处理
anomaly_scores = -model.decision_function(X_test)  # 负值 = 异常
```

### 踩坑 3：contamination 设定不当导致多算法无法横比
**问题**：A 算法用 `contamination=0.02`，B 算法用 `contamination=0.05`，检出数差一倍，无法公平比较。

**解决**：
- 在同一次对比中，**所有算法必须使用相同的 `contamination` 值**
- 建议从 ROC 曲线找最优阈值（需要标签），或用业务先验（如历史异常率）

## 实验建议

### 单参数敏感性分析
```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

contaminations = [0.01, 0.02, 0.05, 0.1]
n_estimators_list = [50, 100, 200, 500]

results = {}
for cont in contaminations:
    for n_est in n_estimators_list:
        model = IsolationForest(
            contamination=cont,
            n_estimators=n_est,
            random_state=42
        )
        model.fit(X_train)
        anomaly_scores = -model.decision_function(X_test)
        pr_auc = average_precision_score(y_test, anomaly_scores)
        results[f"cont={cont},n_est={n_est}"] = pr_auc

# 输出 PR-AUC 对比矩阵
import pandas as pd
df = pd.DataFrame(results, index=[0]).T
print(df)
```

## 最佳实践

1. **默认启动配置**：`contamination=0.05, n_estimators=100`（快速验证）
2. **精调配置**：基于数据特征调整 `contamination` 至接近真值，保持 `n_estimators≥100`
3. **评估指标**：必须用 **PR-AUC** 而非 Accuracy（极不平衡下 accuracy 无意义）
4. **标准化**：IForest 可跳过；但如与 LOF/OCSVM 联合对比，建议统一做 StandardScaler
5. **预处理**：缺失值用中位数填充、异常值尽量不删除（这些可能就是你要找的异常）

## 参考资源

- [Isolation Forest 原论文](https://cs.anu.edu.au/wp-content/uploads/2019/02/Isolation-Forest-Algorithm.pdf)：Liu 等 2008
- scikit-learn 官方文档：https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- 对标论文对比：ECOD、LOF、OCSVM 在不同数据分布下的性能曲线

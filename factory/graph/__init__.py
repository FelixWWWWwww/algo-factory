from factory.graph.store import GraphStore


def init_graph() -> GraphStore:
    store = GraphStore()

    # 预置初始节点（Day 1 底座）

    # 1. 算法节点
    store.add_node("algorithm:IForest",
                   type="Algorithm",
                   framework="sklearn",
                   time_complexity="O(n log n)")
    store.add_node("algorithm:LOF",
                   type="Algorithm",
                   framework="sklearn",
                   novelty_required=True)
    store.add_node("algorithm:OCSVM",
                   type="Algorithm",
                   framework="sklearn",
                   scaler_required="StandardScaler")

    # 2. 指标节点
    store.add_node("metric:PR-AUC",
                   type="Metric",
                   range=[0, 1],
                   is_primary=True)  # 异常检测主指标
    store.add_node("metric:F1",
                   type="Metric",
                   range=[0, 1])

    # 3. 关系边
    store.add_edge("algorithm:IForest", "metric:PR-AUC",
                   edge_type="EVALUATED_BY")
    store.add_edge("algorithm:IForest", "dependency:scikit-learn>=1.5",
                   edge_type="REQUIRES")

    # 4. 能力节点
    store.add_node("capability:anomaly_detection_sensor",
                   type="Capability",
                   domain="anomaly_detection",
                   target_domain="industrial_sensors",
                   success_rate=0.0)  # 初值 0，Day 3 回写更新

    store.save("data/knowledge_graph.json")
    return store

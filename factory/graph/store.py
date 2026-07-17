import networkx as nx
import json
from pathlib import Path


class GraphStore:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # 允许多条边

    # 1. 写操作
    def add_node(self, node_id: str, **properties):
        """添加节点及属性"""
        self.graph.add_node(node_id, **properties)

    def add_edge(self, source: str, target: str, edge_type: str, **attrs):
        """添加边，edge_type 在属性中"""
        self.graph.add_edge(source, target, edge_type=edge_type, **attrs)

    # 2. 读操作
    def query_by_task_type(self, task_type: str) -> dict:
        """按任务类型检索相关节点"""
        # 找所有 task_type=anomaly_detection 的节点
        results = {}
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("task_type") == task_type:
                results[node_id] = attrs
        return results

    def neighborhood(self, node_id: str, hop: int = 2) -> dict:
        """获取节点的 k-hop 邻域"""
        ego = nx.ego_graph(self.graph, node_id, radius=hop)
        return {
            "nodes": dict(ego.nodes(data=True)),
            "edges": list(ego.edges(data=True))
        }

    # 3. 持久化
    def to_json(self):
        """导出为可序列化字典"""
        return {
            "nodes": dict(self.graph.nodes(data=True)),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **attr
                }
                for u, v, attr in self.graph.edges(data=True)
            ]
        }

    def save(self, path: str):
        """保存为 JSON（Day 3 图谱回写用）"""
        Path(path).write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str):
        """从 JSON 加载"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for node_id, attrs in data["nodes"].items():
            self.add_node(node_id, **attrs)
        for edge in data["edges"]:
            source, target = edge.pop("source"), edge.pop("target")
            self.add_edge(source, target, **edge)

    def export_graphml(self, path: str):
        """导出为 GraphML 格式（可用 PyVis 可视化）"""
        graphml_graph = nx.MultiDiGraph()
        for node_id, attrs in self.graph.nodes(data=True):
            normalized_attrs = {}
            for key, value in attrs.items():
                if isinstance(value, (list, dict, tuple, set)):
                    normalized_attrs[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized_attrs[key] = value
            graphml_graph.add_node(node_id, **normalized_attrs)

        for u, v, attrs in self.graph.edges(data=True):
            normalized_attrs = {}
            for key, value in attrs.items():
                if isinstance(value, (list, dict, tuple, set)):
                    normalized_attrs[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized_attrs[key] = value
            graphml_graph.add_edge(u, v, **normalized_attrs)

        nx.write_graphml(graphml_graph, path)

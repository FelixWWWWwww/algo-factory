"""
知识图谱自动抽取脚本

功能：读取 data/docs/ 下的 Markdown 文档 → LLM 抽取结构化知识 → 入图 → 持久化

执行方式：
  python -m factory.graph.extract
  或
  python factory/graph/extract.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field
from factory.llm.structured import call_structured
from factory.graph.store import GraphStore


# ============================================================================
# 抽取用的数据模型（用于 LLM 结构化输出）
# ============================================================================

class ExtractedCapability(BaseModel):
    """从文档抽取的单个能力"""
    name: str = Field(..., description="能力名称，如 'IForest on high-dim sensor data'")
    description: str = Field(..., description="能力描述，1-2句话")
    algorithm: str = Field(default="", description="关联的算法，如 'IsolationForest'")
    task_type: str = Field(default="anomaly_detection", description="任务类型")
    domain: str = Field(default="industrial_sensors", description="应用领域")


class ExtractedMetric(BaseModel):
    """从文档抽取的单个评估指标"""
    name: str = Field(..., description="指标名称，如 'pr_auc', 'f1', 'recall'")
    description: str = Field(..., description="指标定义和计算方法")
    ideal_value: float = Field(default=1.0, description="理想值（通常 0-1）")
    min_threshold: float = Field(default=0.5, description="可接受的最小值")


class ExtractedDependency(BaseModel):
    """从文档抽取的单个依赖"""
    name: str = Field(..., description="库或工具名称，如 'scikit-learn', 'pandas'")
    module: str = Field(..., description="具体模块，如 'sklearn.ensemble.IsolationForest'")
    version_requirement: str = Field(default=">=1.0", description="版本要求")
    reason: str = Field(..., description="为什么需要这个依赖")


class ExtractedFailureCase(BaseModel):
    """从文档抽取的单个失败案例"""
    name: str = Field(..., description="失败案例名称")
    description: str = Field(..., description="什么出错了")
    root_cause: str = Field(..., description="根本原因分析")
    affected_scenario: str = Field(..., description="影响的场景，如 'imbalanced data > 95%'")
    severity: str = Field(default="high", description="严重程度：low/medium/high/critical")
    mitigation: str = Field(..., description="如何规避或修复")


class ExtractedLesson(BaseModel):
    """从文档抽取的单个经验教训"""
    title: str = Field(..., description="经验标题，如 'Always use StandardScaler with OCSVM'")
    description: str = Field(..., description="具体经验说明，2-3句话")
    affected_algorithms: List[str] = Field(default=["all"], description="影响的算法列表")
    priority: str = Field(default="high", description="优先级：low/medium/high")


class DocumentExtractionResult(BaseModel):
    """单个文档的完整抽取结果"""
    document_path: str = Field(default="")
    document_name: str = Field(default="")
    timestamp: str = Field(default="")
    capabilities: List[ExtractedCapability] = Field(default_factory=list)
    metrics: List[ExtractedMetric] = Field(default_factory=list)
    dependencies: List[ExtractedDependency] = Field(default_factory=list)
    failure_cases: List[ExtractedFailureCase] = Field(default_factory=list)
    lessons: List[ExtractedLesson] = Field(default_factory=list)


# ============================================================================
# 核心抽取逻辑
# ============================================================================

class KnowledgeExtractor:
    """知识图谱抽取器"""

    def __init__(self, use_mock: bool = False):
        """
        Args:
            use_mock: 是否使用 Mock LLM（用于快速测试）
        """
        self.use_mock = use_mock
        self.graph_store = GraphStore()
        self.extraction_results: List[DocumentExtractionResult] = []

    def read_document(self, doc_path: str) -> str:
        """读取 Markdown 文档"""
        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_from_document(self, doc_path: str) -> DocumentExtractionResult:
        """
        从单个文档抽取知识

        流程：
          1. 读文档内容
          2. 用 LLM 抽取 capabilities/metrics/dependencies/lessons/failure_cases
          3. 返回结构化结果
        """
        print(f"\n{'='*60}")
        print(f"📄 正在抽取: {Path(doc_path).name}")
        print(f"{'='*60}")

        # 读文档
        doc_content = self.read_document(doc_path)
        doc_name = Path(doc_path).stem

        # LLM 抽取 Prompt
        extraction_prompt = f"""
请从以下 Markdown 文档中抽取结构化知识，用于构建异常检测知识图谱。

【文档内容】
```
{doc_content}
```

【抽取任务】

1. **Capabilities（能力）**：文档中提到的算法应用能力
   示例：
   - name: "IForest on high-dimensional sensor data"
   - description: "IForest 在高维工业传感器数据上的优势"
   - algorithm: "IsolationForest"

2. **Metrics（评估指标）**：推荐的评估指标及阈值
   示例：
   - name: "pr_auc"
   - description: "精确率-召回率曲线下面积，用于评估不平衡数据"
   - min_threshold: 0.60

3. **Dependencies（依赖）**：算法实现需要的库和模块
   示例：
   - name: "scikit-learn"
   - module: "sklearn.ensemble.IsolationForest"

4. **Lessons（经验）**：最佳实践和注意事项
   示例：
   - title: "Always standardize before OCSVM"
   - affected_algorithms: ["OneClassSVM"]

5. **FailureCases（失败案例）**：可能的陷阱和错误做法
   示例：
   - name: "using accuracy on imbalanced data"
   - root_cause: "accuracy 在不平衡数据上失效"
   - mitigation: "必须用 PR-AUC 作为主指标"

请返回 JSON 格式，包含以上所有字段（字段为空时返回空列表）。
"""

        # 调用 LLM 进行结构化抽取
        try:
            result = call_structured(
                prompt=extraction_prompt,
                pydantic_model=DocumentExtractionResult,
                use_mock=self.use_mock,
                # 添加必须的字段
                document_path=doc_path,
                document_name=doc_name,
                timestamp=datetime.now().isoformat()
            )
            print(f"✅ 抽取成功")
            print(f"   - Capabilities: {len(result.capabilities)}")
            print(f"   - Metrics: {len(result.metrics)}")
            print(f"   - Dependencies: {len(result.dependencies)}")
            print(f"   - Lessons: {len(result.lessons)}")
            print(f"   - FailureCases: {len(result.failure_cases)}")

            return result

        except Exception as e:
            print(f"❌ 抽取失败: {str(e)}")
            # 返回空结果，继续处理其他文档
            return DocumentExtractionResult(
                document_path=doc_path,
                document_name=doc_name,
                timestamp=datetime.now().isoformat()
            )

    def add_to_graph(self, extraction_result: DocumentExtractionResult) -> None:
        """
        将抽取结果加入知识图谱

        流程：
          1. 为每个 capability 创建节点
          2. 为每个 lesson 创建节点并连到 capability
          3. 为每个 failure_case 创建节点并连到相关 capability
          4. 为 metrics/dependencies 创建节点并连接
        """
        doc_name = extraction_result.document_name

        # 1. 添加 Capability 节点
        for cap in extraction_result.capabilities:
            cap_id = f"capability:{doc_name}:{cap.name.replace(' ', '_')}"
            self.graph_store.add_node(
                cap_id,
                type="Capability",
                name=cap.name,
                description=cap.description,
                algorithm=cap.algorithm,
                task_type=cap.task_type,
                domain=cap.domain,
                success_rate=0.0,  # 初始为 0，后续通过验证更新
            )
            print(f"  ✓ Added Capability: {cap.name}")

            # 连接到 Algorithm 节点
            if cap.algorithm:
                algo_id = f"algorithm:{cap.algorithm}"
                self.graph_store.add_node(
                    algo_id,
                    type="Algorithm",
                    name=cap.algorithm,
                    description=f"{cap.algorithm} algorithm",
                    task_types=["anomaly_detection"],
                )
                self.graph_store.add_edge(cap_id, algo_id, "USES_ALGORITHM")

        # 2. 添加 Metric 节点和关系
        for metric in extraction_result.metrics:
            metric_id = f"metric:{metric.name}"
            self.graph_store.add_node(
                metric_id,
                type="Metric",
                name=metric.name,
                description=metric.description,
                ideal_value=metric.ideal_value,
                min_threshold=metric.min_threshold,
            )

            # 关联到所有 capabilities
            for cap in extraction_result.capabilities:
                cap_id = f"capability:{doc_name}:{cap.name.replace(' ', '_')}"
                self.graph_store.add_edge(cap_id, metric_id, "EVALUATED_BY")
            print(f"  ✓ Added Metric: {metric.name}")

        # 3. 添加 Dependency 节点
        for dep in extraction_result.dependencies:
            dep_id = f"dependency:{dep.name}"
            self.graph_store.add_node(
                dep_id,
                type="Dependency",
                name=dep.name,
                module=dep.module,
                version_requirement=dep.version_requirement,
                reason=dep.reason,
            )

            # 关联到所有 capabilities
            for cap in extraction_result.capabilities:
                cap_id = f"capability:{doc_name}:{cap.name.replace(' ', '_')}"
                self.graph_store.add_edge(cap_id, dep_id, "REQUIRES")
            print(f"  ✓ Added Dependency: {dep.name}")

        # 4. 添加 Lesson 节点（经验教训）
        for lesson in extraction_result.lessons:
            lesson_id = f"lesson:{doc_name}:{lesson.title.replace(' ', '_')}"
            self.graph_store.add_node(
                lesson_id,
                type="Lesson",
                title=lesson.title,
                description=lesson.description,
                affected_algorithms=lesson.affected_algorithms,
                priority=lesson.priority,
                source_document=doc_name,
            )

            # 关联到相关的 capabilities
            for cap in extraction_result.capabilities:
                # 如果 lesson 影响该 algorithm 或 "all"
                if ("all" in lesson.affected_algorithms or
                    cap.algorithm in lesson.affected_algorithms):
                    cap_id = f"capability:{doc_name}:{cap.name.replace(' ', '_')}"
                    self.graph_store.add_edge(cap_id, lesson_id, "HAS_LESSON")
            print(f"  ✓ Added Lesson: {lesson.title}")

        # 5. 添加 FailureCase 节点（失败案例）——这是图谱自增强的关键
        for failure in extraction_result.failure_cases:
            failure_id = f"failure_case:{failure.name.replace(' ', '_')}"
            self.graph_store.add_node(
                failure_id,
                type="FailureCase",
                name=failure.name,
                description=failure.description,
                root_cause=failure.root_cause,
                affected_scenario=failure.affected_scenario,
                severity=failure.severity,
                mitigation=failure.mitigation,
                source_document=doc_name,
            )

            # 创建关联的 Lesson（自动生成规避策略）
            lesson_id = f"lesson:{failure.name.replace(' ', '_')}:mitigation"
            self.graph_store.add_node(
                lesson_id,
                type="Lesson",
                title=f"Avoid: {failure.name}",
                description=failure.mitigation,
                affected_algorithms=["all"],
                priority="critical" if failure.severity == "critical" else failure.severity,
                source_document=doc_name,
            )
            self.graph_store.add_edge(failure_id, lesson_id, "CAUSED_LESSON")
            print(f"  ✓ Added FailureCase: {failure.name}")

    def run(self, docs_dir: str = "data/docs") -> None:
        """
        执行完整的知识抽取流程

        Args:
            docs_dir: 文档目录路径
        """
        docs_dir = Path(docs_dir)

        if not docs_dir.exists():
            print(f"❌ 文档目录不存在: {docs_dir}")
            return

        # 找所有 .md 文件
        doc_files = sorted(docs_dir.glob("*.md"))
        if not doc_files:
            print(f"❌ 未找到 Markdown 文件在: {docs_dir}")
            return

        print(f"\n🔍 发现 {len(doc_files)} 个 Markdown 文档")

        # 逐个抽取
        for doc_path in doc_files:
            result = self.extract_from_document(str(doc_path))
            self.extraction_results.append(result)
            self.add_to_graph(result)

        # 保存图
        self.save_graph()

    def save_graph(self, output_dir: str = "data") -> None:
        """
        保存知识图谱到文件

        输出：
          - data/knowledge_graph.json: 完整图数据（JSON 序列化）
          - data/knowledge_graph.graphml: GraphML 格式（可用 Gephi/Cytoscape 可视化）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存为 JSON
        json_path = output_dir / "knowledge_graph.json"
        graph_json = self.graph_store.to_json()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 保存知识图谱（JSON）: {json_path}")

        # 保存为 GraphML
        graphml_path = output_dir / "knowledge_graph.graphml"
        self.graph_store.export_graphml(str(graphml_path))
        print(f"✅ 保存知识图谱（GraphML）: {graphml_path}")

        # 打印统计信息
        self._print_stats()

    def _print_stats(self) -> None:
        """打印图的统计信息"""
        graph = self.graph_store.graph
        print(f"\n📊 知识图谱统计:")
        print(f"   节点总数: {graph.number_of_nodes()}")
        print(f"   边总数: {graph.number_of_edges()}")

        # 按节点类型统计
        node_types = {}
        for node_id, attr in graph.nodes(data=True):
            node_type = attr.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"\n   节点分布:")
        for node_type, count in sorted(node_types.items()):
            print(f"     - {node_type}: {count}")

        # 按边类型统计
        edge_types = {}
        for u, v, attr in graph.edges(data=True):
            edge_type = attr.get("relation_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print(f"\n   边分布:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"     - {edge_type}: {count}")


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主程序入口"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # 检查环境变量来决定是否用 Mock
    use_mock = os.getenv("USE_MOCK", "1").lower() == "1"
    docs_dir = os.getenv("DOCS_DIR", "data/docs")

    print("=" * 70)
    print("🚀 知识图谱自动抽取工具")
    print("=" * 70)
    print(f"📂 文档目录: {docs_dir}")
    print(f"🤖 LLM 模式: {'Mock (测试)' if use_mock else 'Real (真实 API)'}")
    print("=" * 70)

    extractor = KnowledgeExtractor(use_mock=use_mock)
    extractor.run(docs_dir=docs_dir)

    print("\n" + "=" * 70)
    print("✅ 知识抽取完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

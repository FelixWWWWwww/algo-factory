"""证明「从失败中学习」：同一任务跑两次，第二次 Planner 主动规避上次失败的算法。

用法: python demo_second_run.py
预期: 第 1 次三方案平等；第 2 次 LOF / OCSVM 被标注"历史失败已规避"并降级排后。
"""
import os
from factory.pipeline import Pipeline

DATA = "data/synth/demo_hard.csv"
GRAPH = "data/knowledge_graph.json"


def show(tag, state):
    print(f"\n===== {tag} =====")
    print("  方案顺序 & 状态：")
    for i, p in enumerate(state.plans, 1):
        avoided = "⚠️ 历史失败已规避" if "历史失败" in p.rationale else "正常提出"
        print(f"    {i}. {p.algorithm:20s} [{avoided}]")
    for vr in state.validation_results:
        print(f"     验证 {vr.plan_name:22s} -> {vr.status}")


if __name__ == "__main__":
    # 清空旧图，保证从零开始，结论干净（删不掉就写空图兜底）
    try:
        if os.path.exists(GRAPH):
            os.remove(GRAPH)
    except OSError:
        open(GRAPH, "w", encoding="utf-8").write('{"nodes": {}, "edges": []}')

    # 第 1 次：图谱空白，Planner 没有任何先验
    run1 = Pipeline(use_mock=True).run("对工业传感器数据进行异常检测", data_path=DATA)
    show("第 1 次运行（图谱空白，无先验经验）", run1)

    # 第 2 次：新建 Pipeline → __init__ 加载上一次存下的图谱 →
    #          Retriever 读到 LOF/OCSVM 的失败 → Planner 规避
    run2 = Pipeline(use_mock=True).run("对工业传感器数据进行异常检测", data_path=DATA)
    show("第 2 次运行（已加载图谱，携带上次失败经验）", run2)

    learned = [p.algorithm for p in run2.plans if "历史失败" in p.rationale]
    print("\n────────────────────────────────────────")
    print("✅ 第 2 次主动规避/降级的算法：", learned or "（无）")
    print("   证明：系统把第 1 次的失败沉淀进知识图谱，第 2 次检索到后主动规避。")
    assert learned, "❌ 未体现从失败中学习"

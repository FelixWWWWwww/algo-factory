"""证明「画像感知的从失败中学习」：

  第 1 次（致密簇数据，图谱空白）：三方案平等。
  第 2 次（致密簇数据，已学习）  ：LOF / OCSVM 被规避（当前画像与失败时相似）。
  第 3 次（离散异常数据，画像不同）：LOF 不再被规避——换个数据集照样有机会当冠军。

用法: python demo_second_run.py
"""
import os
from factory.pipeline import Pipeline
from factory.nodes import make_synthetic_dataset

HARD = "data/synth/demo_hard.csv"          # 异常聚成致密簇（LOF 的克星）
SPREAD = "data/synth/demo_spread.csv"      # 异常离散分布（LOF 的主场）
GRAPH = "data/knowledge_graph.json"


def show(tag, state):
    print(f"\n===== {tag} =====")
    prof = state.data_profile or {}
    print(f"  数据画像：异常紧致度={prof.get('anomaly_compactness')}  异常占比={prof.get('anomaly_ratio')}")
    for i, p in enumerate(state.plans, 1):
        avoided = "⚠️ 历史失败已规避" if "历史失败" in p.rationale else "正常提出"
        print(f"    {i}. {p.algorithm:20s} [{avoided}]")


def avoided_algos(state):
    return [p.algorithm for p in state.plans if "历史失败" in p.rationale]


if __name__ == "__main__":
    # 准备两份画像迥异的数据
    if not os.path.exists(SPREAD):
        make_synthetic_dataset(1500, 6, 0.03, random_state=1, save_path=SPREAD)

    # 清空旧图，从零开始
    try:
        os.remove(GRAPH)
    except OSError:
        open(GRAPH, "w", encoding="utf-8").write('{"nodes": {}, "edges": []}')

    run1 = Pipeline(use_mock=True).run("对工业传感器数据进行异常检测", data_path=HARD)
    show("第 1 次（致密簇数据，图谱空白）", run1)

    run2 = Pipeline(use_mock=True).run("对工业传感器数据进行异常检测", data_path=HARD)
    show("第 2 次（致密簇数据，已加载失败经验）", run2)

    run3 = Pipeline(use_mock=True).run("对工业传感器数据进行异常检测", data_path=SPREAD)
    show("第 3 次（离散异常数据，画像不同）", run3)

    print("\n────────────────────────────────────────")
    print("第 2 次规避：", avoided_algos(run2) or "（无）", " ← 相似数据，复用失败经验")
    print("第 3 次规避：", avoided_algos(run3) or "（无）", " ← 画像不同，LOF 获得重新机会")
    assert "LocalOutlierFactor" in avoided_algos(run2), "❌ 相似数据上应规避 LOF"
    assert "LocalOutlierFactor" not in avoided_algos(run3), "❌ 不同数据上不应误伤 LOF"
    print("\n✅ 画像感知学习成立：LOF 只在“像致密簇”的数据上被降级，换数据集照样有机会当冠军。")

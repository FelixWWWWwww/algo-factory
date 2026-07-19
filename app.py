# app.py
import os, json, glob
import streamlit as st
from factory.pipeline import Pipeline

st.set_page_config(page_title="异常检测算法能力工厂", layout="wide")
st.title("🏭 异常检测算法能力工厂")

tab_run, tab_graph, tab_hist = st.tabs(["Run", "Graph", "History"])

with tab_run:
    query = st.text_input("任务描述", "对工业传感器数据进行异常检测")
    mock = st.checkbox("Mock 模式（无需 API Key）", True)
    up = st.file_uploader("上传 CSV（可选，含 label 列则算监督指标）", type="csv")
    if st.button("运行", type="primary"):
        data_path = None
        if up is not None:
            os.makedirs("data/synth", exist_ok=True)
            data_path = os.path.join("data/synth", "_upload.csv")
            with open(data_path, "wb") as f:
                f.write(up.getbuffer())
        with st.spinner("Agent 流水线运行中…"):
            pipe = Pipeline(use_mock=mock)
            state = pipe.run(query, data_path=data_path)
            pipe.dump_state(state)              # 落盘 → 进 History
        st.success(f"最佳模型：{state.best_model}")
        st.json(state.final_metrics)
        st.subheader("方案对比")
        st.table([{
            "方案": p.name, "算法": p.algorithm,
            "PR-AUC": (vr.metrics or {}).get("pr_auc"),
            "状态": vr.status,
        } for p, vr in zip(state.plans, state.validation_results)])
        img = os.path.join("reports", f"{state.task_id}_scores.png")
        if os.path.exists(img):
            st.image(img, caption="异常分数分布")
        rp = os.path.join("reports", f"{state.task_id}.md")
        if os.path.exists(rp):
            with st.expander("完整报告"):
                st.markdown(open(rp, encoding="utf-8").read())

with tab_graph:
    p = "data/knowledge_graph.json"
    if os.path.exists(p):
        g = json.load(open(p, encoding="utf-8"))
        st.write(f"节点数：{len(g.get('nodes', {}))}　边数：{len(g.get('edges', []))}")
        st.json(g)
    else:
        st.info("暂无图谱，先在 Run 页跑一次任务")

with tab_hist:
    files = sorted(glob.glob("logs/*.json"), reverse=True)
    if not files:
        st.info("暂无历史，先跑一次任务并 dump_state")
    for f in files[:20]:
        d = json.load(open(f, encoding="utf-8"))
        st.write(f"**{d['task_id']}** — {d.get('best_model')} — {d.get('final_metrics')}")
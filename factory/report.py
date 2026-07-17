# factory/report.py
"""T3.7 报告生成：方案对比表 + 最优指标 + 异常分数分布图。"""
import os, logging
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)

_TEMPLATE = """# 异常检测验证报告

- **任务 ID**：{{ task_id }}
- **需求**：{{ user_query }}
- **场景**：{{ task_type }}　**异常占比(EDA)**：{{ anomaly_ratio }}
- **最佳模型**：**{{ best_model }}**（主指标 PR-AUC）

## EDA 摘要
{{ eda_summary }}

## 方案对比（按 PR-AUC 排序，禁用 accuracy 选优）
| 方案 | 算法 | PR-AUC | F1 | Precision | Recall | 状态 |
|---|---|---|---|---|---|---|
{% for r in rows -%}
| {{ r.name }} | {{ r.algorithm }} | {{ r.pr_auc }} | {{ r.f1 }} | {{ r.precision }} | {{ r.recall }} | {{ r.status }} |
{% endfor %}

## 最优指标
{% for k, v in final_metrics.items() -%}
- **{{ k }}**：{{ v }}
{% endfor %}

## Top-K 最可疑样本行号
{{ topk }}

![异常分数分布](./{{ task_id }}_scores.png)

> ⚠️ 极不平衡下 accuracy 会因"全判正常"虚高，本报告全程以 PR-AUC / F1(anomaly) 为准。
"""


def generate_report(state, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p, vr in zip(state.plans, state.validation_results):
        m = vr.metrics or {}
        rows.append({
            "name": p.name, "algorithm": p.algorithm,
            "pr_auc": m.get("pr_auc", "-"), "f1": m.get("f1", "-"),
            "precision": m.get("precision", "-"), "recall": m.get("recall", "-"),
            "status": vr.status,
        })
    rows.sort(key=lambda r: (r["pr_auc"] if isinstance(r["pr_auc"], (int, float)) else -1), reverse=True)

    md = Template(_TEMPLATE).render(
        task_id=state.task_id, user_query=state.user_query,
        task_type=getattr(state.task_card, "task_type", ""),
        anomaly_ratio=state.anomaly_ratio, best_model=state.best_model,
        eda_summary=state.eda_summary or "（无）",
        rows=rows, final_metrics=state.final_metrics or {},
        topk=state.topk_indices[:20],
    )
    path = os.path.join(out_dir, f"{state.task_id}.md")
    Path(path).write_text(md, encoding="utf-8")
    _plot_scores(state, out_dir)
    logger.info(f"[report] 已生成: {path}")
    return path


def _plot_scores(state, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        scores = np.array(state.anomaly_scores, dtype=float)
        if scores.size == 0:
            return
        plt.figure(figsize=(7, 4))
        plt.hist(scores, bins=50)
        if state.threshold is not None:
            plt.axvline(state.threshold, color="r", linestyle="--", label="threshold")
            plt.legend()
        plt.title("Anomaly score distribution")
        plt.xlabel("score (越大越可疑)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{state.task_id}_scores.png"), dpi=100)
        plt.close()
    except Exception as e:
        logger.warning(f"[report] 画图失败: {e}")
"""T3.7 报告生成：完整验证报告（数据画像 + 选型依据 + 为什么选它 + 指标 + 分数图）。"""
import os, logging
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)

_TEMPLATE = """# 🏭 异常检测验证报告

## 概览
- **任务 ID**：{{ task_id }}
- **需求**：{{ user_query }}
- **场景**：{{ task_type }}{% if anomaly_subtype %} · 异常子类：{{ anomaly_subtype }}{% endif %}
- **最佳模型**：**{{ best_model }}**（主指标 PR-AUC = {{ best_pr_auc }}）
- **数据实测异常占比(EDA)**：{{ anomaly_ratio }}

## 📋 数据画像
| 维度 | 值 |
|---|---|
| 样本数 | {{ dp.n_samples }} |
| 特征数 | {{ dp.n_features }} |
| 数值 / 类别 / 时序特征数 | {{ dp.numeric }} / {{ dp.categorical }} / {{ dp.temporal }} |
| 是否含标签 | {{ dp.has_labels }} |
| 最高缺失率 | {{ dp.missing_rate }} |
| 量纲是否悬殊 | {{ dp.scale_disparity }} |

## 🧠 EDA 摘要
{{ eda_summary }}

## 🏆 为什么选择 {{ best_model }}？
在 **{{ n_plans }}** 个候选方案中，**{{ best_model }}** 实测 PR-AUC = **{{ best_pr_auc }}** 为最高，且通过五层验证（语法→安全→运行→指标→签名），据此选为最优方案。

- **入选依据**：{{ best_rationale }}
{% if failed %}- **其它方案为何落选**：
{% for f in failed %}  - **{{ f.algorithm }}**：{{ f.reason }}
{% endfor %}{% endif %}

## 🔬 候选方案与选型依据（按实测 PR-AUC 排序）
| 方案 | 算法 | 入选理由 | 预期 PR-AUC | 实测 PR-AUC | 状态 |
|---|---|---|---|---|---|
{% for r in rows -%}
| {{ r.name }} | {{ r.algorithm }} | {{ r.rationale }} | {{ r.expected }} | {{ r.pr_auc }} | {{ r.status }} |
{% endfor %}

## 📊 最优指标
{% for k, v in final_metrics.items() -%}
- **{{ k }}**：{{ v }}
{% endfor %}

## 🎯 Top-K 最可疑样本行号
{{ topk }}
{% if lessons %}
## 💡 检索到的经验教训（图谱沉淀，指导本次选型）
{% for l in lessons -%}
- {{ l }}
{% endfor %}{% endif %}
> 📊 异常分数分布图：见上方图表（或文件 `reports/{{ task_id }}_scores.png`）。
>
> ⚠️ 极度不平衡下 accuracy 会因“全判正常”而虚高，本报告全程以 **PR-AUC / F1(anomaly)** 为准。
"""


def _clean(t):
    return str(t or "").replace("|", "/").replace("\n", " ").strip()


def generate_report(state, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    plans = list(state.plans)
    plan_by_name = {p.name: p for p in plans}

    rows, failed = [], []
    for vr in state.validation_results:
        p = plan_by_name.get(vr.plan_name)
        m = vr.metrics or {}
        rows.append({
            "name": vr.plan_name,
            "algorithm": getattr(p, "algorithm", "") if p else "",
            "rationale": _clean(getattr(p, "rationale", "") if p else "") or "—",
            "expected": getattr(p, "expected_metric", "-") if p else "-",
            "pr_auc": m.get("pr_auc", "-"),
            "status": vr.status,
        })
        if vr.status != "passed":
            failed.append({
                "algorithm": (getattr(p, "algorithm", "") if p else "") or vr.plan_name,
                "reason": _clean(vr.error_message) or "未通过验证",
            })
    rows.sort(key=lambda r: (r["pr_auc"] if isinstance(r["pr_auc"], (int, float)) else -1),
              reverse=True)

    best_plan = next((p for p in plans if getattr(p, "is_best", False)), None) \
        or next((p for p in plans if p.algorithm == state.best_model), None)
    best_rationale = _clean(getattr(best_plan, "rationale", "")) if best_plan else "—"

    dp = state.data_profile or {}
    ft = dp.get("feature_types", {}) or {}
    ctx = dict(
        task_id=state.task_id, user_query=state.user_query,
        task_type=getattr(state.task_card, "task_type", ""),
        anomaly_subtype=getattr(state.task_card, "anomaly_subtype", ""),
        anomaly_ratio=state.anomaly_ratio, best_model=state.best_model,
        best_pr_auc=(state.final_metrics or {}).get("pr_auc", "-"),
        best_rationale=best_rationale or "—",
        n_plans=len(rows), failed=failed,
        eda_summary=state.eda_summary or "（无）",
        rows=rows, final_metrics=state.final_metrics or {},
        topk=state.topk_indices[:20],
        lessons=[_clean(l) for l in (getattr(state.retrieved_context, "lessons", []) or [])[:4]],
        dp=dict(
            n_samples=dp.get("n_samples", "-"), n_features=dp.get("n_features", "-"),
            numeric=len(ft.get("numeric", [])), categorical=len(ft.get("categorical", [])),
            temporal=len(ft.get("temporal", [])),
            has_labels=dp.get("has_labels", "-"), missing_rate=dp.get("missing_rate", "-"),
            scale_disparity=dp.get("scale_disparity", "-"),
        ),
    )
    md = Template(_TEMPLATE).render(**ctx)
    path = os.path.join(out_dir, f"{state.task_id}.md")
    Path(path).write_text(md, encoding="utf-8")
    _plot_scores(state, out_dir)
    return path


def _plot_scores(state, out_dir: str):
    """画异常分数分布图。坐标轴用英文，避免 matplotlib 缺中文字体乱码。"""
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
        plt.xlabel("anomaly score (higher = more suspicious)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{state.task_id}_scores.png"), dpi=100)
        plt.close()
    except Exception as e:
        logger.warning(f"[report] 画图失败: {e}")

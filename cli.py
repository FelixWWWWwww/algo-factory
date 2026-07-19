"""
cli.py
命令行入口，使用 Typer
"""

import typer
import sys
import os
import glob
from pathlib import Path
from factory.pipeline import Pipeline

app = typer.Typer(help="异常检测算法工厂 CLI")


def _ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


@app.command()
def run(
        query: str = typer.Argument(..., help="任务描述，如 '对工业传感器数据进行异常检测'"),
        mock: bool = typer.Option(True, "--mock/--real", help="是否使用 Mock 模式"),
        data: str = typer.Option(None, "--data", "-d", help="CSV 数据路径（可选；不传则自动合成）"),
        output_dir: str = typer.Option("logs", "--output-dir", "-o", help="日志输出目录"),
):
    """
    运行异常检测任务。

    示例：

    \b
    # Mock 模式（Day 1）
    python cli.py run "对工业传感器数据进行异常检测" --mock

    \b
    # 真实模式（Day 2+）
    python cli.py run "对工业传感器数据进行异常检测" --real
    """
    _ensure_utf8_stdout()

    print(f"\n🔍 开始执行异常检测任务")
    print(f"   Query: {query}")
    print(f"   Mock 模式: {mock}")

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 创建流水线并执行
    pipeline = Pipeline(use_mock=mock)
    state = pipeline.run(query, data_path=data)

    # 落盘日志
    output_file = pipeline.dump_state(state, output_dir=output_dir)

    print(f"\n✅ 任务完成！")
    print(f"   结果已保存到: {output_file}")
    print(f"   最佳模型: {state.best_model}")
    pr_auc = state.metrics.get("pr_auc")
    print(f"   PR-AUC: {pr_auc:.2f}" if isinstance(pr_auc, (int, float)) else "   PR-AUC: N/A")


@app.command()
def status(task_id: str = typer.Argument(..., help="Task ID")):
    """查看历史任务状态"""
    _ensure_utf8_stdout()
    logs_dir = Path("logs")
    task_file = logs_dir / f"{task_id}.json"

    if not task_file.exists():
        print(f"❌ 任务 {task_id} 未找到")
        return

    import json
    data = json.loads(task_file.read_text(encoding="utf-8"))
    print(f"\n📋 Task {task_id} 状态：")
    print(f"   Query: {data.get('user_query')}")
    print(f"   Best Model: {data.get('best_model')}")
    print(f"   Metrics: {data.get('metrics')}")


@app.command("run-dag")
def run_dag(
        query: str = typer.Argument("对工业传感器数据进行异常检测", help="任务描述"),
        mock: bool = typer.Option(True, "--mock/--real"),
        data: str = typer.Option(None, "--data", "-d", help="CSV 数据路径（可选）"),
):
    """以 DAG 编排方式运行（与 run 等价，保留命令名兼容）。"""
    _ensure_utf8_stdout()
    pipeline = Pipeline(use_mock=mock)
    state = pipeline.run(query, data_path=data)
    print(f"\n✅ 完成：best={state.best_model}  metrics={state.final_metrics}")
    print(f"   日志：{pipeline.dump_state(state)}")


@app.command()
def reset(
        history: bool = typer.Option(False, "--history", help="同时清空运行历史 logs/*.json"),
):
    """一键清空已学习的知识（知识图谱），让系统回到“空白大脑”，便于演示从零学习。"""
    _ensure_utf8_stdout()
    targets = ["data/knowledge_graph.json", "data/knowledge_graph.graphml"]
    if history:
        targets += glob.glob("logs/*.json")

    done = 0
    for p in targets:
        if not os.path.exists(p):
            continue
        try:
            os.remove(p)
            done += 1
        except OSError:
            # 删不掉就倒空（知识图谱写空图，其它跳过）
            if p.endswith("knowledge_graph.json"):
                Path(p).write_text('{"nodes": {}, "edges": []}', encoding="utf-8")
                done += 1

    print(f"🧹 已清空知识图谱{'（含运行历史）' if history else ''} —— 处理 {done} 个文件。")
    print("   系统已回到空白大脑，下次运行将从零重新学习。")


if __name__ == "__main__":
    app()

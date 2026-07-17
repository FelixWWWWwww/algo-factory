"""
cli.py
命令行入口，使用 Typer
"""

import typer
import sys
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
    state = pipeline.run(query)

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
):
    """以 DAG 编排方式运行（与 run 等价，保留命令名兼容）。"""
    _ensure_utf8_stdout()
    pipeline = Pipeline(use_mock=mock)
    state = pipeline.run(query)
    print(f"\n✅ 完成：best={state.best_model}  metrics={state.final_metrics}")
    print(f"   日志：{pipeline.dump_state(state)}")


if __name__ == "__main__":
    app()

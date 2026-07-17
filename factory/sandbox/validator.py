# factory/sandbox/validator.py
from __future__ import annotations
import ast, logging
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from factory.sandbox.runner import run_code
from factory.sandbox.security import check_code, UnsafeCodeError

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    status: str                       # passed / failed
    failed_layer: str | None = None   # L1_syntax / L2_security / L5_signature / L3_runtime / L4_metric
    metrics: dict = field(default_factory=dict)
    message: str = ""


def load_config(task_type: str = "anomaly_detection",
                config_dir: str = "data/configs/validation") -> dict:
    path = Path(config_dir) / f"{task_type}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate(code: str, data_path: str, config: dict) -> ValidationReport:
    # L1 语法
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ValidationReport("failed", "L1_syntax", message=f"语法错误: {e}")

    # L2 安全
    try:
        check_code(code)
    except UnsafeCodeError as e:
        return ValidationReport("failed", "L2_security", message=str(e))

    # L5 签名
    if "def run(" not in code:
        return ValidationReport("failed", "L5_signature",
                                message=f"缺少约定签名 {config.get('required_signature')}")

    # L3 沙箱运行
    res = run_code(code, data_path, timeout=int(config.get("timeout_sec", 60)))
    if res.status != "success":
        return ValidationReport("failed", "L3_runtime", metrics=res.metrics,
                                message=(res.stderr or res.status)[:500])

    metrics = res.metrics
    # L4 指标阈值
    for k, thr in (config.get("thresholds") or {}).items():
        v = metrics.get(k)
        if v is None:
            return ValidationReport("failed", "L4_metric", metrics=metrics,
                                    message=f"缺少指标 {k}（可能无标签，无法监督评估）")
        if v < thr:
            return ValidationReport("failed", "L4_metric", metrics=metrics,
                                    message=f"{k}={v} < 阈值 {thr}")
    return ValidationReport("passed", None, metrics=metrics, message="五层全部通过")
# factory/sandbox/runner.py
from __future__ import annotations
import subprocess, sys, re, json, tempfile, os, time, logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
_RESULT_RE = re.compile(r"RESULT_JSON:(\{.*\})")


@dataclass
class ExecutionResult:
    status: str                       # success / runtime_error / timeout
    metrics: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    elapsed_sec: float = 0.0


def run_code(code: str, data_path: str, timeout: int = 60) -> ExecutionResult:
    """在子进程中运行生成代码，解析 RESULT_JSON。主进程绝不受子进程崩溃影响。"""
    tmpdir = tempfile.mkdtemp(prefix="sandbox_")
    script = os.path.join(tmpdir, "gen.py")
    Path(script).write_text(code, encoding="utf-8")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, script, data_path],
            capture_output=True, text=True, encoding="utf-8",  # Windows 必须显式 utf-8
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult("timeout", stderr=str(e), elapsed_sec=time.perf_counter() - t0)

    elapsed = round(time.perf_counter() - t0, 3)
    if proc.returncode != 0:
        return ExecutionResult("runtime_error", stdout=proc.stdout, stderr=proc.stderr,
                               returncode=proc.returncode, elapsed_sec=elapsed)

    m = _RESULT_RE.search(proc.stdout or "")
    if not m:
        return ExecutionResult("runtime_error", stdout=proc.stdout,
                               stderr="未找到 RESULT_JSON 输出", returncode=0, elapsed_sec=elapsed)
    try:
        metrics = json.loads(m.group(1))
    except Exception as e:
        return ExecutionResult("runtime_error", stdout=proc.stdout,
                               stderr=f"RESULT_JSON 解析失败: {e}", elapsed_sec=elapsed)
    return ExecutionResult("success", metrics=metrics, stdout=proc.stdout,
                           stderr=proc.stderr, returncode=0, elapsed_sec=elapsed)
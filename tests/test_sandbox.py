import pytest
from factory.sandbox.security import check_code, UnsafeCodeError
from factory.sandbox.runner import run_code
from factory.agents import CoderAgent
from factory.llm import MockClient
from factory.nodes import make_synthetic_dataset


def test_security_blocks_os_system():
    with pytest.raises(UnsafeCodeError):
        check_code("import os\nos.system('echo hi')")


def test_security_allows_generated_code():
    code = CoderAgent(MockClient())._template("IsolationForest", 0.03)
    assert check_code(code) is True


def test_runner_parses_result_json(tmp_path):
    csv = tmp_path / "d.csv"
    make_synthetic_dataset(600, 6, 0.03, save_path=str(csv))
    code = CoderAgent(MockClient())._template("IsolationForest", 0.03)
    res = run_code(code, str(csv))
    assert res.status == "success"
    assert "pr_auc" in res.metrics


def test_runner_timeout():
    res = run_code("import time\ntime.sleep(5)\n", "x.csv", timeout=1)
    assert res.status == "timeout"

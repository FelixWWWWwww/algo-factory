from factory.sandbox.validator import validate, load_config
from factory.agents import CoderAgent
from factory.llm import MockClient
from factory.nodes import make_synthetic_dataset


def test_syntax_error_caught():
    r = validate("def run(\n", "x.csv", load_config())
    assert r.status == "failed" and r.failed_layer == "L1_syntax"


def test_missing_signature():
    r = validate("print('RESULT_JSON:{}')", "x.csv", load_config())
    assert r.status == "failed" and r.failed_layer == "L5_signature"


def test_iforest_passes(tmp_path):
    csv = tmp_path / "d.csv"
    make_synthetic_dataset(1200, 6, 0.03, save_path=str(csv))
    code = CoderAgent(MockClient())._template("IsolationForest", 0.03)
    r = validate(code, str(csv), load_config())
    assert r.status == "passed", r.message

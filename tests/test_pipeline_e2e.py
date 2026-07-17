import json
from factory.pipeline import Pipeline


def test_end_to_end_mock():
    s = Pipeline(use_mock=True, max_retries=0).run("对工业传感器数据进行异常检测")
    assert s.best_model in {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"}
    assert "pr_auc" in s.final_metrics
    assert len(s.validation_results) == 3
    assert any(vr.status == "passed" for vr in s.validation_results)


def test_dump_state_is_json_safe(tmp_path):
    pipe = Pipeline(use_mock=True, max_retries=0)
    s = pipe.run("异常检测")
    path = pipe.dump_state(s, output_dir=str(tmp_path))
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["best_model"] and data["validation_results"]

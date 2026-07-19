import os, json, tempfile
from fastapi import FastAPI, UploadFile, File, Form
from factory.pipeline import Pipeline

app = FastAPI(title="异常检测算法能力工厂 API")


@app.post("/run")
async def run_task(query: str = Form(...), mock: bool = Form(True),
                   file: UploadFile = File(None)):
    data_path = None
    if file is not None:
        fd, data_path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
    pipe = Pipeline(use_mock=mock)
    state = pipe.run(query, data_path=data_path)
    pipe.dump_state(state)                      # 落盘 → 进 History
    return {
        "task_id": state.task_id,
        "best_model": state.best_model,
        "final_metrics": state.final_metrics,
        "validation_results": [vr.model_dump() for vr in state.validation_results],
    }


@app.get("/graph")
def get_graph():
    p = "data/knowledge_graph.json"
    if not os.path.exists(p):
        return {"nodes": {}, "edges": []}
    raw = open(p, "rb").read()
    for enc in ("utf-8", "gbk"):          # 兼容历史上被 GBK 写坏的文件
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return {"nodes": {}, "edges": [], "error": "graph file decode failed"}

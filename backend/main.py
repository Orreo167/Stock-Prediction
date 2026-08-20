"""FastAPI 入口：托管前端静态页面，提供训练/预测异步接口。"""

import os
import sys
import threading
import time
import uuid

# 保证从任意目录启动都能导入 backend 包内的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from trainer import run_training

app = FastAPI(title="股票预测系统")

# 挂载前端静态资源
app.mount("/css", StaticFiles(directory=config.FRONTEND_DIR + "/css"), name="css")
app.mount("/js", StaticFiles(directory=config.FRONTEND_DIR + "/js"), name="js")

# 任务存储：job_id -> {status, progress, detail, result, error, created_at}
JOBS = {}
JOBS_LOCK = threading.Lock()


class PredictRequest(BaseModel):
    code: str
    horizon: int = config.FORECAST_HORIZON_DEFAULT
    force_retrain: bool = False
    options: dict = None


class _JobRunner(threading.Thread):
    def __init__(self, job_id, code, horizon, force_retrain, options=None):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.code = code
        self.horizon = horizon
        self.force_retrain = force_retrain
        self.options = options or {}

    def run(self):
        def progress_cb(stage, pct, detail):
            with JOBS_LOCK:
                job = JOBS[self.job_id]
                job["status"] = "running"
                job["stage"] = stage
                job["pct"] = round(pct * 100)
                job["detail"] = detail
        try:
            result = run_training(
                self.code, self.horizon,
                progress_cb=progress_cb,
                force_retrain=self.force_retrain,
                options=self.options,
            )
            with JOBS_LOCK:
                JOBS[self.job_id].update(status="done", pct=100, result=result)
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[self.job_id].update(
                    status="error", error=str(exc), detail=str(exc))


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR + "/index.html")


@app.post("/api/predict")
def start_predict(req: PredictRequest):
    code = req.code.strip()
    if not code.isdigit() or len(code) != 6:
        return JSONResponse(status_code=400, content={"error": "股票代码须为 6 位数字"})
    horizon = max(config.FORECAST_HORIZON_MIN,
                  min(config.FORECAST_HORIZON_MAX, req.horizon))
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued", "stage": "", "pct": 0,
            "detail": "任务已创建", "result": None, "error": None,
            "created_at": time.time(),
        }
    _JobRunner(job_id, code, horizon, req.force_retrain,
               options=req.options).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在"})
    return {
        "status": job["status"],
        "stage": job["stage"],
        "pct": job["pct"],
        "detail": job["detail"],
        "error": job["error"],
        "result": job["result"],
    }

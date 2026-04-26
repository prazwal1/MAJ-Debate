"""FastAPI web server for MAJ Debate Arena."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline_web import NVIDIA_MODELS, run_debate

app = FastAPI(title="MAJ Debate Arena")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# job_id → asyncio.Queue  (lives for the duration of one SSE stream)
_jobs: dict[str, asyncio.Queue] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
async def get_models():
    return {"models": [{"id": k, "label": v} for k, v in NVIDIA_MODELS.items()]}


@app.post("/api/debate")
async def start_debate(request: Request):
    config = await request.json()
    if not config.get("topic", "").strip():
        raise HTTPException(status_code=400, detail="topic is required")

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _jobs[job_id] = queue

    # Capture the running loop so the pipeline thread can post events safely.
    loop = asyncio.get_running_loop()

    def on_progress(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def _run() -> None:
        try:
            await asyncio.to_thread(run_debate, config["topic"], config, on_progress)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )

    asyncio.create_task(_run())
    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
async def sse_events(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job not found")

    async def generator():
        queue = _jobs[job_id]
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    yield "data: {\"type\":\"ping\"}\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    _jobs.pop(job_id, None)
                    break
        except asyncio.CancelledError:
            _jobs.pop(job_id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "active_jobs": len(_jobs)}

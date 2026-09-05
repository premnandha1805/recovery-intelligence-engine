"""
scripts/run_real_concurrency_test.py
====================================
Orchestrates a real two-process concurrency test:
- Spawns Python FastAPI with --workers 1 on port 8000
- Spawns NestJS backend on port 3000
- Chooses a NEW uncached payment_id
- Fires two concurrent POST /decisions requests to NestJS
- Validates both HTTP 200, second request < 6500ms, exactly 1 LLM evaluation
- Reports all 7 requested metrics
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
PYTHON_VENV = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
AUDIT_DB_PATH = ROOT_DIR / "decision_engine" / "audit.db"


def is_service_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "test"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_service(url: str, name: str, max_wait: float = 20.0):
    start = time.time()
    while time.time() - start < max_wait:
        if is_service_ready(url):
            print(f"[READY] {name} is listening at {url}")
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timeout waiting for {name} at {url}")


async def post_decision(session_url: str, payment_id: str, client_id: str) -> dict:
    import httpx
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(session_url, json={"payment_id": payment_id})
        t1 = time.monotonic()
        duration_ms = round((t1 - t0) * 1000, 2)
        return {
            "client_id": client_id,
            "status_code": resp.status_code,
            "duration_ms": duration_ms,
            "data": resp.json() if resp.status_code == 200 else resp.text,
            "start_time": t0,
            "end_time": t1,
        }


def clean_audit_record(payment_id: str):
    """Ensure payment_id is 100% uncached in audit.db."""
    from decision_engine.audit import init_audit_db
    init_audit_db(AUDIT_DB_PATH)
    try:
        with sqlite3.connect(str(AUDIT_DB_PATH)) as conn:
            conn.execute("DELETE FROM decision_audit WHERE payment_id = ?", (payment_id,))
            conn.execute("DELETE FROM decision_audit_events WHERE payment_id = ?", (payment_id,))
            conn.commit()
            print(f"[CLEANUP] Deleted any prior records for {payment_id} from {AUDIT_DB_PATH}")
    except Exception as e:
        print(f"[WARN] Error cleaning audit records: {e}")


def get_event_count_for_payment(payment_id: str) -> int:
    if not AUDIT_DB_PATH.exists():
        return 0
    with sqlite3.connect(str(AUDIT_DB_PATH)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM decision_audit_events WHERE payment_id = ?", (payment_id,))
        return cur.fetchone()[0]


async def run_test():
    # Pick a fresh payment ID from the dataset
    payment_id = "pay_000015_a4"
    clean_audit_record(payment_id)

    python_proc = None
    nest_proc = None

    try:
        # 1. Start Python FastAPI with --workers 1
        print("[START] Launching Python FastAPI service (--workers 1, port 8000)...")
        python_env = os.environ.copy()
        python_cmd = [
            str(PYTHON_VENV),
            "-m",
            "uvicorn",
            "decision_engine.service:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--workers",
            "1",
            "--log-level",
            "info",
        ]
        python_proc = subprocess.Popen(
            python_cmd,
            cwd=str(ROOT_DIR),
            env=python_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        wait_for_service("http://127.0.0.1:8000/health", "Python Decision Engine")

        # 2. Start NestJS Backend
        print("[START] Launching NestJS backend service (port 3000)...")
        backend_dir = ROOT_DIR / "backend"
        nest_env = os.environ.copy()
        nest_env["PORT"] = "3000"
        nest_env["DECISION_ENGINE_TIMEOUT_MS"] = "8000"
        nest_cmd = ["node", "dist/main.js"]
        nest_proc = subprocess.Popen(
            nest_cmd,
            cwd=str(backend_dir),
            env=nest_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        wait_for_service("http://127.0.0.1:3000/health", "NestJS Backend")

        # 3. Fire two concurrent POST /decisions calls for the SAME payment_id
        print(f"\n[EXECUTE] Firing two concurrent POST /decisions for payment_id: {payment_id}...")
        url = "http://127.0.0.1:3000/decisions"

        t_overall_start = time.monotonic()
        task1 = asyncio.create_task(post_decision(url, payment_id, "req1"))
        # Brief 5ms delay to guarantee req1 enters Python lock first
        await asyncio.sleep(0.005)
        task2 = asyncio.create_task(post_decision(url, payment_id, "req2"))

        res1, res2 = await asyncio.gather(task1, task2)
        total_time_ms = round((time.monotonic() - t_overall_start) * 1000, 2)

        # Identify which request finished first / evaluated fresh vs cached
        if res1["data"].get("decision_source") == "cache":
            fresh_req, cached_req = res2, res1
        else:
            fresh_req, cached_req = res1, res2

        # 4. Check events table in audit.db
        llm_count = get_event_count_for_payment(payment_id)

        # 5. Calculate metrics
        second_request_lock_wait = max(0.0, round((cached_req["duration_ms"] - (cached_req["end_time"] - fresh_req["end_time"]) * 1000), 2))
        if second_request_lock_wait <= 0:
            second_request_lock_wait = round(fresh_req["duration_ms"] - (cached_req["start_time"] - fresh_req["start_time"]) * 1000, 2)

        print("\n" + "=" * 60)
        print("REAL TWO-PROCESS CONCURRENCY TEST RESULTS")
        print("=" * 60)
        print(f"- NestJS timeout:           8000 ms")
        print(f"- Python deadline:          6.0 s (request_start + 6.0)")
        print(f"- first request duration:   {fresh_req['duration_ms']} ms")
        print(f"- second request duration:  {cached_req['duration_ms']} ms")
        print(f"- second request lock wait: {second_request_lock_wait} ms")
        print(f"- second request status:    HTTP {cached_req['status_code']}")
        print(f"- LLM invocation count:     {llm_count}")
        print(f"- Overall Wall-Clock Time:  {total_time_ms} ms")
        print("=" * 60)

        # Assertions
        assert res1["status_code"] == 200, f"Req 1 failed: {res1}"
        assert res2["status_code"] == 200, f"Req 2 failed: {res2}"
        assert cached_req["duration_ms"] < 6500, f"Second request exceeded 6500ms: {cached_req['duration_ms']}ms"
        assert llm_count == 1, f"Expected exactly 1 LLM evaluation, found {llm_count}"
        print("[SUCCESS] All concurrency and deadline assertions passed!")

    finally:
        # Clean shutdown of both processes
        print("\n[TEARDOWN] Stopping services...")
        if nest_proc is not None:
            nest_proc.terminate()
            try:
                nest_proc.wait(timeout=3)
            except Exception:
                nest_proc.kill()
        if python_proc is not None:
            python_proc.terminate()
            try:
                python_proc.wait(timeout=3)
            except Exception:
                python_proc.kill()
        print("[TEARDOWN] Complete.")


if __name__ == "__main__":
    asyncio.run(run_test())

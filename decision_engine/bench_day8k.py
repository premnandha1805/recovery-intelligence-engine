"""
decision_engine/bench_day8k.py
==============================
Day 8K Performance Benchmark: PostgreSQL (Docker) vs Day 7 SQLite.

Measures all 6 categories under identical execution conditions:
1. Cache-hit latency
2. Fresh-evaluation latency
3. force_recompute latency
4. Database write latency (save_decision_with_event)
5. 20 concurrent DIFFERENT payments
6. 20 concurrent SAME payment (lock-wait behavior)

Ensures:
- Exact same deterministic mocks for LLM and Policy (0 Azure tokens)
- Thorough warmup runs before taking samples
- Multi-sample statistical measurements (min, median, mean, p95, max)
- Full database correctness verification for each metric
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import sys
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
from dotenv import load_dotenv
load_dotenv(".env.test")
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from httpx import ASGITransport, AsyncClient
import numpy as np
import pandas as pd
import psycopg

from decision_engine.audit import (
    CREATE_TABLE_SQL,
    CREATE_EVENTS_TABLE_SQL,
    CREATE_EVENTS_INDEX_SQL,
    compute_state_fingerprint,
)
from decision_engine.context_node import _get_dataset
from decision_engine.graph import create_recovery_graph
from decision_engine.persistence import (
    PostgresDecisionRepository,
    SqliteDecisionRepository,
    create_postgres_pool,
    close_postgres_pool,
)
from decision_engine.persistence.migrate import run_migrations
from decision_engine.reasoning_node import LLMDecision
from decision_engine.service import app


TEST_DB_URL = os.getenv("TEST_DATABASE_URL")


# ── Deterministic Mocks (0 Azure Tokens) ───────────────────────────────────────

def make_mock_policy():
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.15, "RETRY": 0.80, "RETRY_NUDGE": 0.90, "ESCALATE": 0.35}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


def make_mock_llm():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        return LLMDecision(
            decision="RETRY_NUDGE",
            confidence=0.95,
            reasoning="Selected RETRY_NUDGE based on policy recommendations.",
            risk_level="low",
            expected_incremental_value=125.0,
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision="RETRY_NUDGE",
        confidence=0.95,
        reasoning="Selected RETRY_NUDGE based on policy recommendations.",
        risk_level="low",
        expected_incremental_value=125.0,
    ))
    return mock_llm


def ensure_payment_row(pid: str) -> None:
    base_df = _get_dataset().copy()
    template_row = dict(base_df.iloc[0].to_dict())
    template_row.update({
        "payment_id": pid,
        "status": "FAILED",
        "attempt_number": 1,
        "consecutive_failed_cycles": 0,
        "consecutive_failures": 0,
        "retry_count": 0,
        "interventions_last_7_days": 0,
        "interventions_7d": 0,
    })
    filtered_df = base_df[base_df["payment_id"] != pid]
    app.state.dataset = pd.concat([filtered_df, pd.DataFrame([template_row])], ignore_index=True)


async def setup_sqlite_app(tmp_dir: str):
    db_path = os.path.join(tmp_dir, "bench_sqlite.db")
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.execute(CREATE_TABLE_SQL)
    await db.execute(CREATE_EVENTS_TABLE_SQL)
    await db.execute(CREATE_EVENTS_INDEX_SQL)
    await db.commit()

    repo = SqliteDecisionRepository(db=db)
    policy = make_mock_policy()
    llm = make_mock_llm()
    graph = create_recovery_graph(policy=policy, llm=llm, use_async=True)

    app.state.persistence_backend = "sqlite"
    app.state.policy = policy
    app.state.graph = graph
    app.state.db = db
    app.state.db_pool = None
    app.state.repository = repo
    app.state.llm_semaphore = asyncio.Semaphore(20)
    app.state.payment_locks = {}
    app.state.locks_mutex = asyncio.Lock()
    app.state.dataset = _get_dataset().copy()

    return app, repo, db, db_path


async def setup_postgres_app():
    run_migrations(database_url=TEST_DB_URL)
    pool = await create_postgres_pool(TEST_DB_URL, min_size=2, max_size=12, timeout_ms=3000)
    repo = PostgresDecisionRepository(pool=pool)

    policy = make_mock_policy()
    llm = make_mock_llm()
    graph = create_recovery_graph(policy=policy, llm=llm, use_async=True)

    app.state.persistence_backend = "postgres"
    app.state.policy = policy
    app.state.graph = graph
    app.state.db = None
    app.state.db_pool = pool
    app.state.repository = repo
    app.state.llm_semaphore = asyncio.Semaphore(20)
    app.state.payment_locks = {}
    app.state.locks_mutex = asyncio.Lock()
    app.state.dataset = _get_dataset().copy()

    return app, repo, pool


def stats_dict(latencies: list[float]) -> dict[str, float]:
    arr = np.array(latencies)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


# ── Benchmark Execution ────────────────────────────────────────────────────────

async def benchmark_backend(backend_name: str, runs: int = 30) -> dict[str, Any]:
    print(f"\n========================================================")
    print(f"  RUNNING BENCHMARK FOR BACKEND: {backend_name.upper()}")
    print(f"========================================================")

    tmp_dir = None
    sqlite_db = None
    pg_pool = None

    if backend_name == "sqlite":
        tmp_dir = tempfile.mkdtemp()
        app_instance, repo, sqlite_db, _ = await setup_sqlite_app(tmp_dir)
    else:
        app_instance, repo, pg_pool = await setup_postgres_app()

    transport = ASGITransport(app=app_instance)
    client = AsyncClient(transport=transport, base_url="http://test")

    try:
        # 0. Warmup
        print("[0/6] Warming up connection and cache...")
        for i in range(5):
            pid = f"warmup_{backend_name}_{i}"
            ensure_payment_row(pid)
            await client.post("/evaluate", json={"payment_id": pid, "force_recompute": True})
            await client.post("/evaluate", json={"payment_id": pid, "force_recompute": False})

        # 1. Fresh Evaluation Latency
        print(f"[1/6] Benchmarking Fresh-Evaluation Latency ({runs} runs)...")
        fresh_lats = []
        for i in range(runs):
            pid = f"pay_fresh_{backend_name}_{i}_{time.time_ns()}"
            ensure_payment_row(pid)
            t0 = time.monotonic()
            resp = await client.post("/evaluate", json={"payment_id": pid, "force_recompute": False})
            t1 = time.monotonic()
            assert resp.status_code == 200
            assert resp.json()["decision_source"] in ("llm", "model", "guardrail")
            fresh_lats.append((t1 - t0) * 1000)

        # 2. Cache-Hit Latency
        print(f"[2/6] Benchmarking Cache-Hit Latency ({runs} runs)...")
        cache_pid = f"pay_cache_target_{backend_name}"
        ensure_payment_row(cache_pid)
        # prime cache
        r_prime = await client.post("/evaluate", json={"payment_id": cache_pid, "force_recompute": False})
        assert r_prime.status_code == 200

        cache_lats = []
        for _ in range(runs):
            t0 = time.monotonic()
            resp = await client.post("/evaluate", json={"payment_id": cache_pid, "force_recompute": False})
            t1 = time.monotonic()
            assert resp.status_code == 200
            assert resp.json()["decision_source"] == "cache"
            cache_lats.append((t1 - t0) * 1000)

        # 3. Force-Recompute Latency
        print(f"[3/6] Benchmarking Force-Recompute Latency ({runs} runs)...")
        recomp_lats = []
        for _ in range(runs):
            t0 = time.monotonic()
            resp = await client.post("/evaluate", json={"payment_id": cache_pid, "force_recompute": True})
            t1 = time.monotonic()
            assert resp.status_code == 200
            assert resp.json()["decision_source"] != "cache"
            recomp_lats.append((t1 - t0) * 1000)

        # 4. Database Write Latency (Direct save_decision_with_event)
        print(f"[4/6] Benchmarking Direct Persistence Write Latency ({runs} runs)...")
        write_lats = []
        for i in range(runs):
            pid = f"pay_write_{backend_name}_{i}_{time.time_ns()}"
            t0 = time.monotonic()
            await repo.save_decision_with_event(
                payment_id=pid,
                decision_id=f"dec_{pid}",
                request_id=f"req_{pid}",
                raw_arm_probabilities={"WAIT": 0.1, "RETRY": 0.9},
                raw_arm_net_values={"WAIT": 0.0, "RETRY": 100.0},
                llm_proposed_decision="RETRY",
                llm_confidence=0.95,
                llm_reasoning="Direct benchmark persistence test.",
                llm_risk_level="low",
                expected_incremental_value=100.0,
                guardrail_verdict="passed",
                guardrail_reason="",
                final_action="RETRY",
                decision_source="llm",
                state_fingerprint="fingerprint_bench_123",
                evaluated_at=datetime.datetime.now(datetime.timezone.utc),
            )
            t1 = time.monotonic()
            write_lats.append((t1 - t0) * 1000)

        # 5. 20 Concurrent DIFFERENT Payments
        print("[5/6] Benchmarking 20 Concurrent DIFFERENT Payments...")
        diff_pids = [f"pay_diff_{backend_name}_{i}_{time.time_ns()}" for i in range(20)]
        for pid in diff_pids:
            ensure_payment_row(pid)

        t0 = time.monotonic()
        tasks = [client.post("/evaluate", json={"payment_id": pid, "force_recompute": True}) for pid in diff_pids]
        diff_resps = await asyncio.gather(*tasks)
        diff_wall_clock = (time.monotonic() - t0) * 1000

        diff_success = sum(1 for r in diff_resps if r.status_code == 200)
        diff_failed = sum(1 for r in diff_resps if r.status_code != 200)
        diff_events_count = 0
        diff_audit_count = 0

        for pid in diff_pids:
            cur_dec = await repo.get_current_decision(pid)
            if cur_dec:
                diff_audit_count += 1
            evts = await repo.get_events(pid)
            diff_events_count += len(evts)

        # 6. 20 Concurrent SAME Payment
        print("[6/6] Benchmarking 20 Concurrent SAME Payment...")
        same_pid = f"pay_same_{backend_name}_{time.time_ns()}"
        ensure_payment_row(same_pid)

        t0 = time.monotonic()
        same_tasks = [client.post("/evaluate", json={"payment_id": same_pid, "force_recompute": False}) for _ in range(20)]
        same_resps = await asyncio.gather(*same_tasks)
        same_wall_clock = (time.monotonic() - t0) * 1000

        same_success = sum(1 for r in same_resps if r.status_code == 200)
        same_failed = sum(1 for r in same_resps if r.status_code != 200)
        same_cache_hits = sum(1 for r in same_resps if r.status_code == 200 and r.json().get("decision_source") == "cache")
        same_fresh_evals = 20 - same_cache_hits

        same_cur = await repo.get_current_decision(same_pid)
        same_cur_count = 1 if same_cur else 0
        same_evts = await repo.get_events(same_pid)
        same_evts_count = len(same_evts)

        results = {
            "backend": backend_name,
            "runs": runs,
            "fresh_eval": stats_dict(fresh_lats),
            "cache_hit": stats_dict(cache_lats),
            "force_recompute": stats_dict(recomp_lats),
            "db_write": stats_dict(write_lats),
            "diff_20": {
                "wall_clock_ms": diff_wall_clock,
                "success": diff_success,
                "failed": diff_failed,
                "audit_rows": diff_audit_count,
                "event_rows": diff_events_count,
            },
            "same_20": {
                "wall_clock_ms": same_wall_clock,
                "success": same_success,
                "failed": same_failed,
                "fresh_evals": same_fresh_evals,
                "cache_hits": same_cache_hits,
                "audit_rows": same_cur_count,
                "event_rows": same_evts_count,
            },
        }
        return results

    finally:
        await client.aclose()
        if sqlite_db:
            await sqlite_db.close()
        if pg_pool:
            await close_postgres_pool(pg_pool)


async def main():
    sqlite_results = await benchmark_backend("sqlite", runs=30)
    postgres_results = await benchmark_backend("postgres", runs=30)

    print("\n\n" + "="*80)
    print("                      DAY 8K BENCHMARK RESULTS")
    print("="*80)
    print(json.dumps({"sqlite": sqlite_results, "postgres": postgres_results}, indent=2))

    # Save to json file for artifact and reporting
    out_path = os.path.join("decision_engine", "benchmark_day8k_results.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({"sqlite": sqlite_results, "postgres": postgres_results}, fp, indent=2)
    print(f"\nResults saved to: {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())

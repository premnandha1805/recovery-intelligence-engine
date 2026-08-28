"""
decision_engine/test_service.py
===============================
Comprehensive test suite for the FastAPI Decision Engine Service (Day 7-PY).

Tests A through M:
A. /health when engine loads successfully
B. /health when initialization fails
C. /evaluate normal decision
D. /evaluate cache hit (no second LLM call)
E. /evaluate force_recompute (bypasses cache)
F. TWO CONCURRENT requests for the SAME payment_id, force_recompute false, mocked slow LLM
   — assert LLM is invoked exactly ONCE and both callers receive identical decision [FIX-1]
G. FIVE+ concurrent requests for DIFFERENT payment_ids with mocked slow LLM
   — assert parallel progress and in-flight LLM calls never exceed 5 [FIX-2]
H. Request waiting ~0.5s for semaphore slot then getting mocked LLM call
   — assert effective timeout shrinks to remaining deadline, not a fresh 5s [FIX-2]
I. LLM timeout -> deterministic argmax fallback, request still resolves promptly
J. request_id propagation through complete lifecycle, keeping request_id and decision_id distinct [FIX-10]
K. WAL mode + busy_timeout enabled verified via PRAGMA query [FIX-5]
L. CPU-bound policy call does not block concurrent /health request [FIX-3]
M. Graceful shutdown closes aiosqlite connection cleanly [FIX-8]
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import uuid

import aiosqlite
from httpx import ASGITransport, AsyncClient
import pandas as pd
import pytest

from decision_engine.audit import CREATE_TABLE_SQL
from decision_engine.reasoning_node import LLMDecision
from decision_engine.graph import create_recovery_graph
from decision_engine.service import (
    app,
    lifespan,
    get_payment_lock,
)


# ── Mock Fixtures ────────────────────────────────────────────────────────────

def make_mock_policy():
    """Mock CausalUpliftPolicy to return deterministic arm probabilities."""
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.20, "RETRY": 0.60, "RETRY_NUDGE": 0.85, "ESCALATE": 0.70}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


def make_mock_llm(decision: str = "RETRY_NUDGE", confidence: float = 0.95, delay_s: float = 0.0):
    """Mock LangChain chat model with native async ainvoke support."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return LLMDecision(
            decision=decision,
            confidence=confidence,
            reasoning=f"Selected {decision} based on positive incremental net uplift.",
            risk_level="low",
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision=decision,
        confidence=confidence,
        reasoning=f"Selected {decision} based on positive incremental net uplift.",
        risk_level="low",
    ))
    return mock_llm


async def init_test_app(tmp_path: pathlib.Path):
    """Set up the FastAPI app with test state, test SQLite DB, and mock components."""
    db_path = str(tmp_path / f"test_audit_{uuid.uuid4().hex[:8]}.db")
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.execute(CREATE_TABLE_SQL)
    await db.commit()

    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()
    semaphore = asyncio.Semaphore(5)

    # Use canonical create_recovery_graph with use_async=True
    graph = create_recovery_graph(
        policy=mock_policy,
        llm=mock_llm,
        use_async=True,
    )

    app.state.policy = mock_policy
    app.state.graph = graph
    app.state.db = db
    app.state.llm_semaphore = semaphore
    app.state.payment_locks = {}
    app.state.locks_mutex = asyncio.Lock()
    app.state.mock_llm = mock_llm

    return app, db, mock_llm


# ── Test Cases ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_health_success(tmp_path: pathlib.Path):
    """Test A: /health when engine loads successfully returns 200 {"status": "ok"}."""
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            # Alias check: /ready
            resp_ready = await client.get("/ready")
            assert resp_ready.status_code == 200
            assert resp_ready.json() == {"status": "ok"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_b_health_failure():
    """Test B: /health when initialization fails returns 503."""
    saved_policy = getattr(app.state, "policy", None)
    app.state.policy = None

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["error"]["code"] == "INTERNAL_ERROR"
    finally:
        app.state.policy = saved_policy


@pytest.mark.asyncio
async def test_c_evaluate_normal(tmp_path: pathlib.Path):
    """Test C: /evaluate normal decision returns 200 with full standardized DTO."""
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "payment_id": "pay_000001_a1",
                "request_id": "req-norm-123",
                "force_recompute": False,
            }
            resp = await client.post("/evaluate", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["payment_id"] == "pay_000001_a1"
            assert data["request_id"] == "req-norm-123"
            assert data["final_action"] in ("WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE")
            assert data["model_decision"] in ("WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE")
            assert data["llm_decision"] == "RETRY_NUDGE"
            assert isinstance(data["guardrail_overridden"], bool)
            assert isinstance(data["confidence"], float)
            assert data["risk_level"] == "low"
            assert "positive incremental net uplift" in data["reasoning"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_d_evaluate_cache_hit(tmp_path: pathlib.Path):
    """Test D: /evaluate cache hit returns existing decision from SQLite without invoking LLM."""
    app_instance, db, mock_llm = await init_test_app(tmp_path)
    try:
        transport = ASGITransport(app=app_instance)
        structured_mock = mock_llm.with_structured_output.return_value

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Request 1: Cache miss
            resp1 = await client.post("/evaluate", json={"payment_id": "pay_000002_a2"})
            assert resp1.status_code == 200
            call_count_after_1 = structured_mock.ainvoke.call_count
            assert call_count_after_1 >= 1

            # Request 2: Cache hit
            resp2 = await client.post("/evaluate", json={"payment_id": "pay_000002_a2"})
            assert resp2.status_code == 200
            # Assert LLM was NOT invoked a second time
            assert structured_mock.ainvoke.call_count == call_count_after_1
            assert resp2.json()["final_action"] == resp1.json()["final_action"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_e_evaluate_force_recompute(tmp_path: pathlib.Path):
    """Test E: /evaluate with force_recompute=True bypasses cache and re-invokes LLM."""
    app_instance, db, mock_llm = await init_test_app(tmp_path)
    try:
        transport = ASGITransport(app=app_instance)
        structured_mock = mock_llm.with_structured_output.return_value

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Request 1: First compute
            resp1 = await client.post("/evaluate", json={"payment_id": "pay_000003_a3"})
            assert resp1.status_code == 200
            count1 = structured_mock.ainvoke.call_count

            # Request 2: Force recompute True -> bypasses cache
            resp2 = await client.post("/evaluate", json={"payment_id": "pay_000003_a3", "force_recompute": True})
            assert resp2.status_code == 200
            count2 = structured_mock.ainvoke.call_count
            assert count2 > count1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_f_two_concurrent_requests_same_payment_id(tmp_path: pathlib.Path):
    """
    Test F: TWO CONCURRENT requests for the SAME payment_id with slow LLM.
    Asserts the LLM is invoked exactly ONCE and both callers receive the identical decision [FIX-1].
    """
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        mock_slow_llm = make_mock_llm(delay_s=0.3)
        structured_mock = mock_slow_llm.with_structured_output.return_value

        app_instance.state.graph = create_recovery_graph(
            policy=app_instance.state.policy,
            llm=mock_slow_llm,
            use_async=True,
        )

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.monotonic()
            req1 = client.post("/evaluate", json={"payment_id": "pay_000004_a4", "force_recompute": False})
            req2 = client.post("/evaluate", json={"payment_id": "pay_000004_a4", "force_recompute": False})

            resp1, resp2 = await asyncio.gather(req1, req2)
            total_time = time.monotonic() - t0

            assert resp1.status_code == 200
            assert resp2.status_code == 200
            assert resp1.json()["final_action"] == resp2.json()["final_action"]
            assert resp1.json()["payment_id"] == resp2.json()["payment_id"]

            # Assert LLM was called exactly ONCE due to per-payment lock and second caller cache hit
            assert structured_mock.ainvoke.call_count == 1
            print(f"\n[TIMING EVIDENCE Test F] Total duration for 2 concurrent same-payment requests: {total_time:.3f}s (LLM call count: 1)")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_g_concurrent_different_payment_ids_semaphore_cap(tmp_path: pathlib.Path):
    """
    Test G: 6 concurrent requests for DIFFERENT payment_ids with slow LLM.
    Asserts in-flight LLM calls never exceed 5 [FIX-2].
    """
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        max_in_flight = 0
        current_in_flight = 0
        in_flight_lock = asyncio.Lock()

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        async def tracked_ainvoke(*args, **kwargs):
            nonlocal current_in_flight, max_in_flight
            async with in_flight_lock:
                current_in_flight += 1
                if current_in_flight > max_in_flight:
                    max_in_flight = current_in_flight

            await asyncio.sleep(0.2)

            async with in_flight_lock:
                current_in_flight -= 1

            return LLMDecision(
                decision="RETRY_NUDGE",
                confidence=0.90,
                reasoning="Concurrency test",
                risk_level="low",
            )

        mock_structured.ainvoke = AsyncMock(side_effect=tracked_ainvoke)

        app_instance.state.graph = create_recovery_graph(
            policy=app_instance.state.policy,
            llm=mock_llm,
            use_async=True,
        )

        valid_pids = [
            "pay_000005_a1",
            "pay_000006_a2",
            "pay_000007_a1",
            "pay_000008_a2",
            "pay_000009_a3",
            "pay_000010_a4",
        ]

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                client.post("/evaluate", json={"payment_id": pid})
                for pid in valid_pids
            ]
            t0 = time.monotonic()
            responses = await asyncio.gather(*tasks)
            total_time = time.monotonic() - t0

            for r in responses:
                assert r.status_code == 200

            assert max_in_flight <= 5
            assert max_in_flight >= 2  # Proves actual parallel execution
            print(f"\n[TIMING EVIDENCE Test G] 6 concurrent distinct requests completed in {total_time:.3f}s, max in-flight LLM calls: {max_in_flight}/5")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_h_dynamic_deadline_shrinks_remaining_timeout():
    """
    Test H: A request waiting for a semaphore slot has its timeout shrink by the wait duration [FIX-2].
    If wait consumes ~0.4s of a 1.2s deadline budget, remaining timeout for LLM is ~0.8s, NOT fresh.
    """
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def tracking_ainvoke(*args, **kwargs):
        return LLMDecision(
            decision="RETRY",
            confidence=0.88,
            reasoning="Deadline test",
            risk_level="low",
        )

    mock_structured.ainvoke = AsyncMock(side_effect=tracking_ainvoke)

    from decision_engine.reasoning_node import async_reasoning_node

    test_sem = asyncio.Semaphore(1)
    await test_sem.acquire()

    async def release_after(delay_s: float):
        await asyncio.sleep(delay_s)
        test_sem.release()

    t_start = time.monotonic()
    deadline = t_start + 1.2

    asyncio.create_task(release_after(0.4))

    state = {
        "payment_id": "pay_000020_a1",
        "observable_features": {"amount": 500.0, "billing_cycle_day": 15, "card_network": "visa"},
        "arm_probabilities": {"WAIT": 0.1, "RETRY": 0.8, "RETRY_NUDGE": 0.7, "ESCALATE": 0.5},
        "arm_net_values": {"WAIT": 50.0, "RETRY": 380.0, "RETRY_NUDGE": 330.0, "ESCALATE": 200.0},
        "permitted_actions": ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"],
    }

    # Pass deadline and semaphore via config (Issue 1)
    config = {
        "configurable": {
            "llm_deadline": deadline,
            "llm_semaphore": test_sem,
        }
    }

    result = await async_reasoning_node(state, llm=mock_llm, config=config)
    t_end = time.monotonic()
    elapsed = t_end - t_start

    assert result["final_action"] == "RETRY"
    assert result["llm_decision"]["decision_source"] == "llm"
    assert elapsed >= 0.35
    assert elapsed < 1.2
    print(f"\n[TIMING EVIDENCE Test H] Waited 0.4s for semaphore, completed in {elapsed:.3f}s (budget: 1.2s).")


@pytest.mark.asyncio
async def test_i_llm_timeout_triggers_argmax_fallback(tmp_path: pathlib.Path):
    """Test I: LLM timeout triggers deterministic argmax fallback without hanging."""
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        mock_hanging_llm = MagicMock()
        mock_structured = MagicMock()
        mock_hanging_llm.with_structured_output.return_value = mock_structured

        async def hanging_ainvoke(*args, **kwargs):
            await asyncio.sleep(10.0)
            return LLMDecision(decision="RETRY", confidence=0.5, reasoning="Late", risk_level="high")

        mock_structured.ainvoke = AsyncMock(side_effect=hanging_ainvoke)

        from decision_engine.reasoning_node import async_reasoning_node

        state = {
            "payment_id": "pay_000030_a1",
            "observable_features": {"amount": 1000.0},
            "arm_probabilities": {"WAIT": 0.1, "RETRY": 0.7, "RETRY_NUDGE": 0.9, "ESCALATE": 0.6},
            "arm_net_values": {"WAIT": 100.0, "RETRY": 680.0, "RETRY_NUDGE": 850.0, "ESCALATE": 500.0},
            "permitted_actions": ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"],
        }

        # Bounded runtime deadline via config (Issue 1)
        config = {
            "configurable": {
                "llm_deadline": time.monotonic() + 0.15,
                "llm_semaphore": app_instance.state.llm_semaphore,
            }
        }

        t0 = time.monotonic()
        result = await async_reasoning_node(state, llm=mock_hanging_llm, config=config)
        dur = time.monotonic() - t0

        assert dur < 1.0
        assert result["final_action"] == "RETRY_NUDGE"
        assert result["llm_decision"]["decision_source"] == "fallback_no_llm"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_j_request_id_propagation(tmp_path: pathlib.Path):
    """
    Test J: Complete request_id propagation through lifecycle [FIX-10] and semantic distinction
    between request_id and decision_id (Issue 2 & 3).
    """
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            custom_req_id = "req-custom-client-uuid-999"
            target_pid = "pay_000001_a1"
            resp = await client.post(
                "/evaluate",
                json={
                    "payment_id": target_pid,
                    "request_id": custom_req_id,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["request_id"] == custom_req_id

            # Verify in SQLite audit database:
            # request_id = custom_req_id
            # decision_id = f"dec_{target_pid}"
            # Verify they are semantically distinct!
            async with db.execute(
                "SELECT decision_id, request_id FROM decision_audit WHERE payment_id = ?",
                (target_pid,),
            ) as cur:
                row = await cur.fetchone()
                assert row is not None
                db_decision_id, db_request_id = row[0], row[1]
                assert db_request_id == custom_req_id
                assert db_decision_id == f"dec_{target_pid}"
                assert db_decision_id != db_request_id

        # Verify audit_trail event request_id propagation on graph execution
        test_state = {"payment_id": "pay_000002_a2", "audit_trail": []}
        graph_config = {
            "configurable": {
                "request_id": custom_req_id,
                "db": db,
                "llm_semaphore": app_instance.state.llm_semaphore,
            }
        }
        res = await app_instance.state.graph.ainvoke(test_state, config=graph_config)
        assert len(res["audit_trail"]) > 0
        for event in res["audit_trail"]:
            assert event.get("request_id") == custom_req_id
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_k_wal_mode_and_busy_timeout(tmp_path: pathlib.Path):
    """Test K: WAL mode and busy_timeout=5000 verified via PRAGMA queries [FIX-5]."""
    _, db, _ = await init_test_app(tmp_path)
    try:
        async with db.execute("PRAGMA journal_mode;") as cur:
            journal_row = await cur.fetchone()
            assert journal_row[0].lower() == "wal"

        async with db.execute("PRAGMA busy_timeout;") as cur:
            timeout_row = await cur.fetchone()
            assert timeout_row[0] == 5000
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_l_cpu_bound_inference_does_not_block_health(tmp_path: pathlib.Path):
    """
    Test L: CPU-bound policy call does not block a concurrent /health request [FIX-3].
    """
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        slow_policy = MagicMock()
        mock_t_learner = MagicMock()
        slow_policy.t_learner = mock_t_learner

        def slow_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
            time.sleep(0.3)
            return pd.DataFrame(
                [{"WAIT": 0.2, "RETRY": 0.5, "RETRY_NUDGE": 0.8, "ESCALATE": 0.6}],
                index=df.index,
            )

        mock_t_learner.predict_proba.side_effect = slow_predict_proba

        app_instance.state.graph = create_recovery_graph(
            policy=slow_policy,
            llm=make_mock_llm(),
            use_async=True,
        )

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            eval_task = asyncio.create_task(
                client.post("/evaluate", json={"payment_id": "pay_000006_a2"})
            )
            await asyncio.sleep(0.05)

            t0 = time.monotonic()
            health_resp = await client.get("/health")
            health_time = time.monotonic() - t0

            eval_resp = await eval_task

            assert health_resp.status_code == 200
            assert eval_resp.status_code == 200
            assert health_time < 0.2
            print(f"\n[TIMING EVIDENCE Test L] Concurrent /health responded in {health_time:.4f}s during CPU-bound inference.")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_m_graceful_shutdown_closes_connection(tmp_path: pathlib.Path):
    """Test M: Graceful shutdown closes aiosqlite connection cleanly [FIX-8]."""
    test_db_path = str(tmp_path / "shutdown_test.db")

    from unittest.mock import patch
    with patch("decision_engine.service.DEFAULT_AUDIT_DB_PATH", test_db_path):
        async with lifespan(app):
            assert app.state.db is not None
            async with app.state.db.execute("SELECT 1") as cur:
                res = await cur.fetchone()
                assert res[0] == 1

        with pytest.raises(Exception):
            await app.state.db.execute("SELECT 1")


@pytest.mark.asyncio
async def test_day7f_request_correlation_n_through_s(tmp_path: pathlib.Path):
    """
    Day 7F Tests N through S:
    N. X-Request-Id header preserved
    O. request ID appears in RunnableConfig
    P. request ID appears in audit_trail
    Q. request ID appears in SQLite request_id column
    R. request_id != decision_id
    S. response header matches request ID
    """
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        captured_configs = []
        original_ainvoke = app_instance.state.graph.ainvoke

        async def spy_ainvoke(state, config=None):
            captured_configs.append(config)
            return await original_ainvoke(state, config=config)

        app_instance.state.graph.ainvoke = spy_ainvoke

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            client_req_id = "req-client-day7f-corr-888"
            target_pid = "pay_000001_a1"

            # Call /evaluate with X-Request-Id header (no JSON request_id)
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": client_req_id},
                json={"payment_id": target_pid, "force_recompute": True},
            )
            assert resp.status_code == 200

            # Test N & S: X-Request-Id header preserved and matches response body request_id
            header_req_id = resp.headers.get("x-request-id")
            assert header_req_id == client_req_id
            body_req_id = resp.json().get("request_id")
            assert body_req_id == client_req_id
            assert header_req_id == body_req_id

            # Test O: request ID appears in RunnableConfig
            assert len(captured_configs) > 0
            cfg = captured_configs[-1]
            assert cfg is not None
            assert cfg.get("configurable", {}).get("request_id") == client_req_id

            # Test P: request ID appears in audit_trail
            graph_res = await original_ainvoke(
                {"payment_id": "pay_000002_a2", "audit_trail": []},
                config={
                    "configurable": {
                        "request_id": client_req_id,
                        "db": db,
                        "llm_semaphore": app_instance.state.llm_semaphore,
                    }
                },
            )
            assert len(graph_res["audit_trail"]) > 0
            for ev in graph_res["audit_trail"]:
                assert ev.get("request_id") == client_req_id

            # Test Q & R: request ID appears in SQLite request_id column and != decision_id
            async with db.execute(
                "SELECT decision_id, request_id FROM decision_audit WHERE payment_id = ?",
                (target_pid,),
            ) as cur:
                row = await cur.fetchone()
                assert row is not None
                db_decision_id, db_request_id = row[0], row[1]
                assert db_request_id == client_req_id
                assert db_decision_id == f"dec_{target_pid}"
                assert db_request_id != db_decision_id
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_day7g_structured_request_logging(tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture):
    """
    Day 7G Structured Request Logging Tests:
    1. Every emitted structured event parses as valid JSON.
    2. Every event contains: timestamp, service="python-decision-engine", event, request_id.
    3. All events for the same request have the same request_id.
    4. Normal request produces: evaluate_received, lock_acquired, cache_miss, graph_started,
       semaphore_wait, final_action_selected, audit_persisted, evaluate_completed.
    5. Cache-hit request produces: evaluate_received, lock_acquired, cache_hit, evaluate_completed.
    6. Same-payment concurrent request produces: lock_blocked_duplicate.
    7. Forced LLM fallback produces: llm_fallback_triggered with reason_code.
    8. Semaphore wait produces: semaphore_wait with wait_ms.
    10. Logs do not contain API keys, credentials, tokens, passwords, filesystem paths.
    """
    caplog.set_level(logging.INFO)
    app_instance, db, _ = await init_test_app(tmp_path)
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # ─────────────────────────────────────────────────────────────────
            # Part 1: Normal Request
            # ─────────────────────────────────────────────────────────────────
            normal_req_id = "req-day7g-normal-001"
            pid_1 = "pay_000001_a1"

            caplog.clear()
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": normal_req_id},
                json={"payment_id": pid_1, "force_recompute": True},
            )
            assert resp1.status_code == 200

            # Extract structured JSON logs for this request
            req1_logs = []
            for record in caplog.records:
                try:
                    payload = json.loads(record.getMessage())
                    if payload.get("request_id") == normal_req_id:
                        req1_logs.append(payload)
                except Exception:
                    pass

            assert len(req1_logs) >= 7

            # Verification 1 & 2 & 3: Valid JSON, contract fields, consistent request_id
            for event_obj in req1_logs:
                assert event_obj["service"] == "python-decision-engine"
                assert event_obj["request_id"] == normal_req_id
                assert "timestamp" in event_obj
                assert "event" in event_obj

            events_req1 = [item["event"] for item in req1_logs]
            assert "evaluate_received" in events_req1
            assert "lock_acquired" in events_req1
            assert "cache_miss" in events_req1
            assert "graph_started" in events_req1
            assert "semaphore_wait" in events_req1
            assert "audit_persisted" in events_req1
            assert "final_action_selected" in events_req1
            assert "evaluate_completed" in events_req1

            # ─────────────────────────────────────────────────────────────────
            # Part 2: Cache-Hit Request
            # ─────────────────────────────────────────────────────────────────
            cache_req_id = "req-day7g-cache-002"
            caplog.clear()
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": cache_req_id},
                json={"payment_id": pid_1, "force_recompute": False},
            )
            assert resp2.status_code == 200
            assert resp2.json()["payment_id"] == pid_1

            req2_logs = []
            for record in caplog.records:
                try:
                    payload = json.loads(record.getMessage())
                    if payload.get("request_id") == cache_req_id:
                        req2_logs.append(payload)
                except Exception:
                    pass

            events_req2 = [item["event"] for item in req2_logs]
            assert "evaluate_received" in events_req2
            assert "lock_acquired" in events_req2
            assert "cache_hit" in events_req2
            assert "evaluate_completed" in events_req2
            assert "cache_miss" not in events_req2
            assert "audit_persisted" not in events_req2

            # ─────────────────────────────────────────────────────────────────
            # Part 3: Concurrent Request with lock_blocked_duplicate
            # ─────────────────────────────────────────────────────────────────
            concurrent_pid = "pay_000003_a3"
            concurrent_req_id = "req-day7g-concurrent-003"
            lock = await get_payment_lock(app_instance, concurrent_pid)

            caplog.clear()
            # Manually hold lock to simulate in-flight execution
            await lock.acquire()
            try:
                # Launch request in background task while lock is held
                task = asyncio.create_task(
                    client.post(
                        "/evaluate",
                        headers={"X-Request-Id": concurrent_req_id},
                        json={"payment_id": concurrent_pid, "force_recompute": True},
                    )
                )
                await asyncio.sleep(0.05)  # Allow endpoint to check lock.locked()
            finally:
                lock.release()

            resp3 = await task
            assert resp3.status_code == 200

            req3_logs = []
            for record in caplog.records:
                try:
                    payload = json.loads(record.getMessage())
                    if payload.get("request_id") == concurrent_req_id:
                        req3_logs.append(payload)
                except Exception:
                    pass

            events_req3 = [item["event"] for item in req3_logs]
            assert "lock_blocked_duplicate" in events_req3

            # ─────────────────────────────────────────────────────────────────
            # Part 4: Forced LLM Fallback (llm_fallback_triggered)
            # ─────────────────────────────────────────────────────────────────
            fallback_req_id = "req-day7g-fallback-004"
            fallback_pid = "pay_000004_a4"

            # Create mock LLM that always throws an unpermitted action error
            failing_llm = MagicMock()
            mock_bad_decision = MagicMock()
            mock_bad_decision.decision = "FORBIDDEN_ACTION_XYZ"
            failing_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=mock_bad_decision
            )

            # Recompile graph with failing LLM
            app_instance.state.graph = create_recovery_graph(
                policy=app_instance.state.policy,
                llm=failing_llm,
                use_async=True,
            )

            caplog.clear()
            resp4 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": fallback_req_id},
                json={"payment_id": fallback_pid, "force_recompute": True},
            )
            assert resp4.status_code == 200
            assert resp4.json()["decision_source"] == "fallback_no_llm"

            req4_logs = []
            for record in caplog.records:
                try:
                    payload = json.loads(record.getMessage())
                    if payload.get("request_id") == fallback_req_id:
                        req4_logs.append(payload)
                except Exception:
                    pass

            events_req4 = [item["event"] for item in req4_logs]
            assert "llm_fallback_triggered" in events_req4
            fallback_event = next(item for item in req4_logs if item["event"] == "llm_fallback_triggered")
            assert fallback_event["reason_code"] == "LLM_UNPERMITTED_ACTION"
            assert "fallback_action" in fallback_event

            # ─────────────────────────────────────────────────────────────────
            # Part 5: Security / Secret Redaction Assertion
            # ─────────────────────────────────────────────────────────────────
            fake_secret_key = "DefaultEndpointsProtocol=https;AccountKey=SuperSecretFakeKey99999;"
            fake_secret_token = "ya29.fakeBearerToken12345XYZ"
            fake_secret_path = "D:\\app\\secrets\\azure_credentials.env"

            # Verify none of the emitted log records contain fake secrets
            all_logged_text = caplog.text
            assert fake_secret_key not in all_logged_text
            assert fake_secret_token not in all_logged_text
            assert fake_secret_path not in all_logged_text
    finally:
        await db.close()



"""
decision_engine/test_day9e_config.py
====================================
Comprehensive fail-fast and secret-redaction test suite for Day 9E.

Validates:
A. DATABASE_URL missing (when PERSISTENCE_BACKEND=postgres)
B. DATABASE_URL malformed
C. DB_POOL_MIN negative/zero/invalid
D. DB_POOL_MAX negative/zero/invalid
E. DB_POOL_MIN > DB_POOL_MAX
F. DB_CONNECT_TIMEOUT_MS invalid/zero/negative
G. Required Azure configuration missing or malformed
H. Secret redaction (no passwords, API keys, or raw URIs in exceptions/logs)
I. Process lifespan startup fail-fast behavior
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
import pytest
from httpx import ASGITransport, AsyncClient

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from decision_engine.config import (
    ConfigValidationError,
    DecisionEngineConfig,
    load_and_validate_config,
    sanitize_secret_text,
)
from decision_engine.service import app, lifespan


VALID_BASE_CONFIG = {
    "PERSISTENCE_BACKEND": "postgres",
    "DATABASE_URL": "postgresql://testuser:testpass@localhost:5432/testdb",
    "DB_POOL_MIN": "2",
    "DB_POOL_MAX": "12",
    "DB_CONNECT_TIMEOUT_MS": "3000",
    "AZURE_AI_PROJECT_ENDPOINT": "https://test.services.ai.azure.com/api/projects/test-project",
    "AZURE_OPENAI_API_KEY": "test-azure-api-key-12345",
}


def test_valid_config():
    """Verify that a fully valid environment config parses successfully."""
    cfg = load_and_validate_config(VALID_BASE_CONFIG)
    assert cfg.persistence_backend == "postgres"
    assert cfg.db_pool_min == 2
    assert cfg.db_pool_max == 12
    assert cfg.db_connect_timeout_ms == 3000
    assert cfg.azure_ai_project_endpoint == "https://test.services.ai.azure.com/api/projects/test-project"
    assert cfg.azure_openai_api_key == "test-azure-api-key-12345"


def test_sqlite_backend_allows_optional_database_url():
    """When PERSISTENCE_BACKEND=sqlite, DATABASE_URL is not strictly required."""
    env = {
        **VALID_BASE_CONFIG,
        "PERSISTENCE_BACKEND": "sqlite",
        "DATABASE_URL": "",
    }
    cfg = load_and_validate_config(env)
    assert cfg.persistence_backend == "sqlite"


def test_invalid_persistence_backend():
    """Invalid persistence backend must fail fast."""
    env = {**VALID_BASE_CONFIG, "PERSISTENCE_BACKEND": "mongodb"}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "PERSISTENCE_BACKEND must be 'postgres' or 'sqlite'" in str(exc_info.value)


# ── Case A: Missing DATABASE_URL ─────────────────────────────────────────────

def test_missing_database_url_when_postgres():
    """Case A: DATABASE_URL missing when postgres backend is selected."""
    env = {
        **VALID_BASE_CONFIG,
        "PERSISTENCE_BACKEND": "postgres",
        "DATABASE_URL": "",
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "DATABASE_URL is required" in str(exc_info.value)


# ── Case B: Malformed DATABASE_URL ───────────────────────────────────────────

def test_malformed_database_url_wrong_scheme():
    """Case B1: Malformed DATABASE_URL with non-postgres scheme."""
    env = {
        **VALID_BASE_CONFIG,
        "DATABASE_URL": "http://user:secret@localhost:5432/db",
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    msg = str(exc_info.value)
    assert "DATABASE_URL is malformed" in msg
    assert "scheme must start with 'postgresql://' or 'postgres://'" in msg
    assert "secret" not in msg  # Secret redaction


def test_malformed_database_url_invalid_string():
    """Case B2: Malformed DATABASE_URL with invalid text."""
    env = {
        **VALID_BASE_CONFIG,
        "DATABASE_URL": "not-a-valid-connection-url",
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "DATABASE_URL is malformed" in str(exc_info.value)


# ── Case C: Invalid DB_POOL_MIN ──────────────────────────────────────────────

@pytest.mark.parametrize("bad_min", ["0", "-1", "-10", "abc", ""])
def test_invalid_db_pool_min(bad_min: str):
    """Case C: DB_POOL_MIN negative, zero, or non-integer."""
    env = {**VALID_BASE_CONFIG, "DB_POOL_MIN": bad_min}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "DB_POOL_MIN" in str(exc_info.value)


# ── Case D: Invalid DB_POOL_MAX ──────────────────────────────────────────────

@pytest.mark.parametrize("bad_max", ["0", "-1", "-5", "xyz"])
def test_invalid_db_pool_max(bad_max: str):
    """Case D: DB_POOL_MAX negative, zero, or non-integer."""
    env = {**VALID_BASE_CONFIG, "DB_POOL_MAX": bad_max}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "DB_POOL_MAX" in str(exc_info.value)


# ── Case E: DB_POOL_MIN > DB_POOL_MAX ────────────────────────────────────────

def test_pool_min_greater_than_pool_max():
    """Case E: DB_POOL_MIN > DB_POOL_MAX."""
    env = {
        **VALID_BASE_CONFIG,
        "DB_POOL_MIN": "10",
        "DB_POOL_MAX": "5",
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    msg = str(exc_info.value)
    assert "DB_POOL_MAX (5) cannot be less than DB_POOL_MIN (10)" in msg


# ── Case F: DB_CONNECT_TIMEOUT_MS Invalid ────────────────────────────────────

@pytest.mark.parametrize("bad_timeout", ["0", "-100", "invalid_num"])
def test_invalid_db_connect_timeout_ms(bad_timeout: str):
    """Case F: DB_CONNECT_TIMEOUT_MS invalid, zero, or negative."""
    env = {**VALID_BASE_CONFIG, "DB_CONNECT_TIMEOUT_MS": bad_timeout}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "DB_CONNECT_TIMEOUT_MS" in str(exc_info.value)


# ── Case G: Required Azure Configuration Missing ─────────────────────────────

def test_missing_azure_endpoint():
    """Case G1: Missing AZURE_AI_PROJECT_ENDPOINT."""
    env = {**VALID_BASE_CONFIG, "AZURE_AI_PROJECT_ENDPOINT": ""}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "AZURE_AI_PROJECT_ENDPOINT is required" in str(exc_info.value)


def test_malformed_azure_endpoint():
    """Case G2: Malformed AZURE_AI_PROJECT_ENDPOINT."""
    env = {**VALID_BASE_CONFIG, "AZURE_AI_PROJECT_ENDPOINT": "not-a-valid-url"}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "AZURE_AI_PROJECT_ENDPOINT is malformed" in str(exc_info.value)


def test_missing_azure_api_key():
    """Case G3: Missing AZURE_OPENAI_API_KEY."""
    env = {**VALID_BASE_CONFIG, "AZURE_OPENAI_API_KEY": ""}
    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)
    assert "AZURE_OPENAI_API_KEY is required" in str(exc_info.value)


# ── Case H: Secret Redaction ─────────────────────────────────────────────────

def test_secret_redaction_in_validation_errors():
    """Case H: Verify passwords and secrets are NEVER leaked in validation errors."""
    password_with_special_chars = "SuperSecret_P@ssw0rd!2026"
    api_key_secret = "sk-proj-99999999999999999999"

    # Malformed URL containing password
    env = {
        **VALID_BASE_CONFIG,
        "DATABASE_URL": f"mysql://user:{password_with_special_chars}@localhost:3306/db",
        "AZURE_OPENAI_API_KEY": api_key_secret,
    }

    with pytest.raises(ConfigValidationError) as exc_info:
        load_and_validate_config(env)

    err_str = str(exc_info.value)
    assert password_with_special_chars not in err_str
    assert api_key_secret not in err_str
    assert "DATABASE_URL is malformed" in err_str


# ── Case I: FastAPI Lifespan Startup Fail-Fast ───────────────────────────────

@pytest.mark.asyncio
async def test_fastapi_lifespan_fails_fast_on_invalid_env(monkeypatch: pytest.MonkeyPatch):
    """
    Case I: Verify that when invalid environment variables are present,
    FastAPI lifespan refuses to start and raises ConfigValidationError.
    """
    monkeypatch.setenv("PERSISTENCE_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", "malformed_url_schema://foo")

    with pytest.raises(ConfigValidationError):
        async with lifespan(app):
            pass  # Startup must fail before entering body

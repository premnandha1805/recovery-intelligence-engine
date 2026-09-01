"""
decision_engine/config.py
=========================
Centralized startup configuration validation using Pydantic.
Validates environment settings at process startup before any traffic is served.
Strictly redacts passwords, tokens, API keys, and connection strings from all
validation errors and logs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigValidationError(RuntimeError):
    """Raised when environment configuration fails startup validation."""
    pass


def sanitize_secret_text(text: str) -> str:
    """
    Remove any connection URI credentials, query tokens, or API keys from error messages.
    """
    if not text:
        return ""
    # Mask postgresql://user:password@host
    masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", text)
    # Mask api keys / bearer tokens
    masked = re.sub(r"(key|token|secret|password)=([^\s&]+)", r"\1=***", masked, flags=re.IGNORECASE)
    return masked


class DecisionEngineConfig(BaseModel):
    """
    Centralized configuration schema for the Python Decision Engine.
    """

    persistence_backend: Literal["postgres", "sqlite"] = "postgres"
    database_url: Optional[str] = None
    db_pool_min: int = Field(default=2, ge=1)
    db_pool_max: int = Field(default=12, ge=1)
    db_connect_timeout_ms: int = Field(default=3000, gt=0)
    azure_ai_project_endpoint: str
    azure_openai_api_key: str

    @field_validator("persistence_backend", mode="before")
    @classmethod
    def validate_backend(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in ("postgres", "sqlite"):
                return val
        raise ConfigValidationError("PERSISTENCE_BACKEND must be 'postgres' or 'sqlite'")

    @field_validator("db_pool_min", mode="before")
    @classmethod
    def validate_pool_min(cls, v: Any) -> int:
        try:
            val = int(v)
        except (TypeError, ValueError):
            raise ConfigValidationError("DB_POOL_MIN must be a valid positive integer")
        if val < 1:
            raise ConfigValidationError(f"DB_POOL_MIN must be at least 1 (got {val})")
        return val

    @field_validator("db_pool_max", mode="before")
    @classmethod
    def validate_pool_max(cls, v: Any) -> int:
        try:
            val = int(v)
        except (TypeError, ValueError):
            raise ConfigValidationError("DB_POOL_MAX must be a valid positive integer")
        if val < 1:
            raise ConfigValidationError(f"DB_POOL_MAX must be at least 1 (got {val})")
        return val

    @field_validator("db_connect_timeout_ms", mode="before")
    @classmethod
    def validate_timeout(cls, v: Any) -> int:
        try:
            val = int(v)
        except (TypeError, ValueError):
            raise ConfigValidationError("DB_CONNECT_TIMEOUT_MS must be a valid positive integer")
        if val <= 0:
            raise ConfigValidationError(f"DB_CONNECT_TIMEOUT_MS must be a positive integer > 0 (got {val})")
        return val

    @field_validator("azure_ai_project_endpoint", mode="before")
    @classmethod
    def validate_azure_endpoint(cls, v: Any) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ConfigValidationError("AZURE_AI_PROJECT_ENDPOINT is required and cannot be empty")
        clean = v.strip()
        parsed = urlparse(clean)
        if not parsed.scheme or parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigValidationError(
                "AZURE_AI_PROJECT_ENDPOINT is malformed: must be a valid HTTP or HTTPS URI"
            )
        return clean

    @field_validator("azure_openai_api_key", mode="before")
    @classmethod
    def validate_azure_key(cls, v: Any) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ConfigValidationError("AZURE_OPENAI_API_KEY is required and cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_cross_field(self) -> "DecisionEngineConfig":
        if self.db_pool_max < self.db_pool_min:
            raise ConfigValidationError(
                f"DB_POOL_MAX ({self.db_pool_max}) cannot be less than DB_POOL_MIN ({self.db_pool_min})"
            )

        if self.persistence_backend == "postgres":
            if not self.database_url or not isinstance(self.database_url, str) or not self.database_url.strip():
                raise ConfigValidationError("DATABASE_URL is required when PERSISTENCE_BACKEND is 'postgres'")
            url_str = self.database_url.strip()
            if not (url_str.startswith("postgresql://") or url_str.startswith("postgres://")):
                raise ConfigValidationError(
                    "DATABASE_URL is malformed: scheme must start with 'postgresql://' or 'postgres://'"
                )
            try:
                parsed = urlparse(url_str)
                if not parsed.netloc and not parsed.hostname:
                    raise ConfigValidationError(
                        "DATABASE_URL is malformed: missing host or network location"
                    )
            except Exception as exc:
                if isinstance(exc, ConfigValidationError):
                    raise
                raise ConfigValidationError("DATABASE_URL is malformed: invalid PostgreSQL URI format")

        return self


def load_and_validate_config(env_dict: Optional[dict[str, Any]] = None) -> DecisionEngineConfig:
    """
    Load configuration from environment variables or custom dict, validate through Pydantic,
    and return a validated DecisionEngineConfig instance.

    Guarantees that no passwords, API keys, or raw connection strings are ever leaked in
    error messages.
    """
    if env_dict is None:
        raw_env = os.environ
    else:
        raw_env = env_dict

    backend = raw_env.get("PERSISTENCE_BACKEND", "postgres")
    db_url = raw_env.get("DATABASE_URL") or raw_env.get("POSTGRES_URL") or raw_env.get("TEST_DATABASE_URL")
    pool_min = raw_env.get("DB_POOL_MIN", "2")
    pool_max = raw_env.get("DB_POOL_MAX", "12")
    timeout_ms = raw_env.get("DB_CONNECT_TIMEOUT_MS", "3000")
    azure_endpoint = raw_env.get("AZURE_AI_PROJECT_ENDPOINT", "")
    azure_key = raw_env.get("AZURE_OPENAI_API_KEY", "")

    try:
        return DecisionEngineConfig(
            persistence_backend=backend,
            database_url=db_url,
            db_pool_min=pool_min,
            db_pool_max=pool_max,
            db_connect_timeout_ms=timeout_ms,
            azure_ai_project_endpoint=azure_endpoint,
            azure_openai_api_key=azure_key,
        )
    except ConfigValidationError:
        raise
    except Exception as exc:
        sanitized_msg = sanitize_secret_text(str(exc))
        raise ConfigValidationError(f"Configuration validation failed: {sanitized_msg}") from None

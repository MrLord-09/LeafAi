"""Signed, expiring admin-cookie authentication for the dashboard."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from fastapi import HTTPException, Request

SESSION_COOKIE = "leafai_admin"
SESSION_LIFETIME_SECONDS = 12 * 60 * 60
DEFAULT_ADMIN_USERNAME = "Alam"
DEFAULT_ADMIN_PASSWORD = "Alam@2304"
DEFAULT_SESSION_SECRET = secrets.token_urlsafe(48)


def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").casefold() == "production"


def _username() -> str:
    configured_username = os.environ.get("ADMIN_USERNAME")
    if _is_production():
        return configured_username or ""
    return configured_username or DEFAULT_ADMIN_USERNAME


def _password() -> str:
    configured_password = os.environ.get("ADMIN_PASSWORD")
    if _is_production():
        return configured_password or ""
    return configured_password or DEFAULT_ADMIN_PASSWORD


def _secret() -> bytes:
    configured_secret = os.environ.get("ADMIN_SESSION_SECRET")
    if _is_production() and not configured_secret:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_SESSION_SECRET must be configured for production.",
        )
    secret = configured_secret or DEFAULT_SESSION_SECRET
    if len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_SESSION_SECRET must contain at least 32 characters.",
        )
    return secret.encode("utf-8")


def admin_is_configured() -> bool:
    configured_secret = os.environ.get("ADMIN_SESSION_SECRET")
    secret_is_valid = bool(
        configured_secret and len(configured_secret) >= 32
        if _is_production()
        else len(configured_secret or DEFAULT_SESSION_SECRET) >= 32
    )
    return secret_is_valid and bool(_username()) and bool(_password())


def validate_admin_configuration() -> None:
    if _is_production() and not admin_is_configured():
        raise RuntimeError(
            "Set ADMIN_USERNAME, ADMIN_PASSWORD, and an ADMIN_SESSION_SECRET "
            "of at least 32 characters before running in production."
        )


def _require_configuration() -> None:
    _secret()
    if _is_production() and (not _username() or not _password()):
        raise HTTPException(
            status_code=503,
            detail="ADMIN_USERNAME and ADMIN_PASSWORD must be configured for production.",
        )


def authenticate(username: str, password: str) -> bool:
    _require_configuration()
    expected_username = _username()
    expected_password = _password()
    if not expected_username or not expected_password:
        return False
    username_matches = hmac.compare_digest(username.encode(), expected_username.encode())
    password_matches = hmac.compare_digest(password.encode(), expected_password.encode())
    return username_matches and password_matches


def create_session(username: str) -> tuple[str, str]:
    csrf_token = secrets.token_urlsafe(32)
    payload = json.dumps(
        {
            "username": username,
            "expires": int(time.time()) + SESSION_LIFETIME_SECONDS,
            "csrf": csrf_token,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_payload = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    signature = hmac.new(_secret(), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded_payload}.{encoded_signature}", csrf_token


def read_session(request: Request) -> dict[str, Any]:
    token = request.cookies.get(SESSION_COOKIE, "")
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected_signature = hmac.new(
            _secret(),
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        supplied_signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise ValueError("invalid signature")
        payload = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        session = json.loads(payload)
        configured_username = _username()
        if (
            session.get("username") != configured_username
            or int(session.get("expires", 0)) < int(time.time())
            or not isinstance(session.get("csrf"), str)
        ):
            raise ValueError("invalid or expired session")
        return session
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Admin session is missing or expired.")


def require_admin(request: Request) -> dict[str, Any]:
    _require_configuration()
    return read_session(request)


def require_csrf(request: Request) -> dict[str, Any]:
    session = require_admin(request)
    supplied_token = request.headers.get("X-CSRF-Token", "")
    if not hmac.compare_digest(supplied_token, session["csrf"]):
        raise HTTPException(status_code=403, detail="CSRF validation failed.")
    return session

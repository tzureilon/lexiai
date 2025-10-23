from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from fastapi import Depends, Header, HTTPException, status

from .db import db

DEFAULT_TOKEN_TTL_SECONDS = int(os.getenv("LEXIAI_TOKEN_TTL", "3600"))


def _load_secret() -> str:
    file_path = os.getenv("LEXIAI_SECRET_FILE")
    secret = os.getenv("LEXIAI_SECRET")
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                candidate = handle.read().strip()
                if candidate:
                    secret = candidate
        except OSError as exc:  # pragma: no cover - depends on filesystem state
            raise RuntimeError(
                "LEXIAI_SECRET_FILE could not be read; ensure the API service has access to the managed secret file."
            ) from exc
    if not secret:
        raise RuntimeError(
            "LEXIAI_SECRET is not configured. Provide a strong secret via environment variable or managed secret file."
        )
    if secret == "change-me" or len(secret) < 32:
        raise RuntimeError(
            "LEXIAI_SECRET must be rotated to a value of at least 32 characters before starting the API service."
        )
    return secret


def _load_additional_secrets() -> Sequence[str]:
    secrets: list[str] = []
    for raw in os.getenv("LEXIAI_ADDITIONAL_SECRETS", "").split(","):
        raw = raw.strip()
        if raw:
            secrets.append(raw)
    for file_path in os.getenv("LEXIAI_ADDITIONAL_SECRET_FILES", "").split(","):
        candidate = file_path.strip()
        if not candidate:
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
                if value:
                    secrets.append(value)
        except OSError:
            continue
    return tuple(secrets)


DEFAULT_SECRET = _load_secret()
ADDITIONAL_SECRETS: Sequence[str] = _load_additional_secrets()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    roles: tuple[str, ...]
    token_id: str | None = None
    token_expires_at: datetime | None = None

    def has_role(self, *candidates: str) -> bool:
        candidate_set = set(candidates)
        return any(role in candidate_set for role in self.roles)


class TokenSigner:
    """Minimal JWT-like signer that uses HMAC-SHA256."""

    def __init__(self, secret: str = DEFAULT_SECRET, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> None:
        if not secret:
            raise RuntimeError("LEXIAI_SECRET must be configured for secure operation")
        self._secret = secret.encode("utf-8")
        self._ttl = ttl_seconds

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def encode(
        self,
        *,
        tenant_id: str,
        user_id: str,
        roles: Iterable[str],
        token_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        issued_at = int(_utcnow().timestamp())
        ttl = ttl_seconds or self._ttl
        payload = {
            "sub": user_id,
            "tenant": tenant_id,
            "roles": list(roles),
            "iat": issued_at,
            "exp": issued_at + ttl,
        }
        if token_id:
            payload["token_id"] = token_id
        header = {"alg": "HS256", "typ": "JWT"}
        signing_input = f"{_b64encode(json.dumps(header, separators=(',', ':')).encode())}.{_b64encode(json.dumps(payload, separators=(',', ':')).encode())}"
        signature = self._sign(signing_input.encode())
        return f"{signing_input}.{signature}"

    def decode(self, token: str) -> dict:
        try:
            header_b64, payload_b64, signature = token.split(".")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format") from exc
        signing_input = f"{header_b64}.{payload_b64}".encode()
        if not self._verify(signing_input, signature):
            verified = False
            for secret in ADDITIONAL_SECRETS:
                if not secret:
                    continue
                if self._verify(signing_input, signature, override_secret=secret):
                    verified = True
                    break
            if not verified:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")
        try:
            payload = json.loads(_b64decode(payload_b64))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token payload malformed") from exc
        exp = payload.get("exp")
        if exp is not None:
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
            if expires_at < _utcnow():
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        return payload

    def _sign(self, data: bytes, override_secret: str | None = None) -> str:
        secret = override_secret.encode("utf-8") if override_secret else self._secret
        signature = hmac.new(secret, data, hashlib.sha256).digest()
        return _b64encode(signature)

    def _verify(self, data: bytes, signature: str, override_secret: str | None = None) -> bool:
        expected = self._sign(data, override_secret=override_secret)
        try:
            return hmac.compare_digest(expected, signature)
        except Exception:  # pragma: no cover - defensive
            return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_token_against_store(token: str, payload: dict) -> tuple[str, str, tuple[str, ...], str | None, datetime | None]:
    token_hash = hash_token(token)
    rows = db.query(
        "SELECT tenant_id, user_id, scopes, expires_at, id FROM api_tokens WHERE token_hash = ?",
        (token_hash,),
        read_only=True,
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API token revoked or unknown")
    record = rows[0]
    tenant_id = record["tenant_id"]
    user_id = record["user_id"]
    scopes = tuple(json.loads(record["scopes"])) if isinstance(record["scopes"], str) else tuple(record["scopes"])
    expires_at = None
    if record["expires_at"]:
        expires_at = datetime.fromisoformat(record["expires_at"].replace("Z", ""))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < _utcnow():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API token expired")
    payload_tenant = payload.get("tenant")
    payload_user = payload.get("sub")
    if payload_tenant != tenant_id or payload_user != user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token subject mismatch")
    db.execute(
        "UPDATE api_tokens SET last_used_at = ? WHERE tenant_id = ? AND id = ?",
        (_utcnow().isoformat(), tenant_id, record["id"]),
    )
    return tenant_id, user_id, scopes, record["id"], expires_at


def build_principal_from_token(token: str, payload: dict) -> Principal:
    tenant_id, user_id, scopes, token_id, expires_at = _validate_token_against_store(token, payload)
    roles = tuple(payload.get("roles", []))
    if not roles:
        roles = scopes
    return Principal(
        tenant_id=tenant_id,
        user_id=user_id,
        roles=tuple(roles),
        token_id=token_id,
        token_expires_at=expires_at,
    )


def get_current_principal(authorization: str = Header(..., alias="Authorization")) -> Principal:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header invalid") from exc
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unsupported authorization scheme")
    payload = token_signer.decode(token)
    principal = build_principal_from_token(token, payload)
    return principal


def require_roles(*roles: str):
    async def _dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        if roles and not principal.has_role(*roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return principal

    return _dependency


token_signer = TokenSigner()


__all__ = [
    "Principal",
    "TokenSigner",
    "token_signer",
    "get_current_principal",
    "require_roles",
    "build_principal_from_token",
    "hash_token",
]

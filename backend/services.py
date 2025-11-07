from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from .ai import (
    CaseOutcomeClassifier,
    CasePrediction,
    DocumentMatch,
    DocumentSegment,
    DocumentVectorStore,
    LegalInsightClassifier,
    LegalKnowledgeBase,
    LegalRagEngine,
    WitnessStrategyGenerator,
)
from .db import db
from .llm import ClaudeClient, LLMGenerationError
from .schemas import (
    AuditLogEntry,
    BackupRecord,
    ChatMessage,
    ContextualReference,
    DatasetRecord,
    DocumentDetails,
    DocumentInsights,
    DocumentMetadataUpdate,
    DocumentSearchResult,
    DocumentSummary,
    DocumentVersion,
    DocumentVersionCreate,
    LoginRequest,
    LoginResponse,
    LlmFailureRecord,
    ModelMetric,
    ModelRunRecord,
    OpsEventRecord,
    PredictionResponse,
    PredictionSignal,
    PrivacyRequest,
    PrivacyRequestCreate,
    PrivacyRequestUpdate,
    StoredPrediction,
    StoredWitnessPlan,
    Tenant,
    TenantCreate,
    TokenCreate,
    TokenResponse,
    UserCreate,
    UserSummary,
    UserUpdate,
    WitnessQuestionSet,
    WorkflowTask,
    WorkflowTaskCreate,
    WorkflowTaskUpdate,
)
from .security import Principal, hash_token, token_signer

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class AuthorizationError(RuntimeError):
    """Raised when a user attempts to access a resource without permission."""


class IdentityService:
    def __init__(self) -> None:
        self._password_iterations = int(os.getenv("LEXIAI_PASSWORD_ITERATIONS", "390000"))

    # Tenant management -------------------------------------------------
    def create_tenant(self, payload: TenantCreate) -> Tenant:
        now = _now_iso()
        metadata = json.dumps(payload.metadata or {})
        try:
            db.execute(
                """
                INSERT INTO tenants (id, name, status, created_at, updated_at, contact_email, metadata)
                VALUES (?, ?, 'active', ?, ?, ?, ?)
                """,
                (payload.id, payload.name, now, now, payload.contact_email, metadata),
            )
        except Exception as exc:  # pragma: no cover - database constraint path
            raise ValueError("Tenant already exists or could not be created") from exc
        return self.get_tenant(payload.id)

    def get_tenant(self, tenant_id: str) -> Tenant:
        rows = db.query("SELECT * FROM tenants WHERE id = ?", (tenant_id,), read_only=True)
        if not rows:
            raise ValueError("Tenant not found")
        return self._row_to_tenant(rows[0])

    def list_tenants(self) -> List[Tenant]:
        rows = db.query("SELECT * FROM tenants ORDER BY created_at", read_only=True)
        return [self._row_to_tenant(row) for row in rows]

    def _row_to_tenant(self, row: Mapping[str, Any]) -> Tenant:
        metadata = row.get("metadata") or "{}"
        metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
        return Tenant(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "")),
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "")),
            contact_email=row.get("contact_email"),
            metadata=metadata_dict or {},
        )

    # User management ---------------------------------------------------
    def create_user(self, payload: UserCreate) -> UserSummary:
        self.get_tenant(payload.tenant_id)
        password_hash = self._hash_password(payload.password)
        now = _now_iso()
        try:
            db.execute(
                """
                INSERT INTO users (id, tenant_id, email, password_hash, roles, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.user_id,
                    payload.tenant_id,
                    payload.email.lower(),
                    password_hash,
                    json.dumps(payload.roles),
                    now,
                    now,
                ),
            )
        except Exception as exc:  # pragma: no cover - database constraint path
            raise ValueError("User already exists or could not be created") from exc
        return self.get_user(payload.tenant_id, payload.user_id)

    def update_user(self, payload: UserUpdate) -> UserSummary:
        updates: dict[str, Any] = {}
        if payload.email is not None:
            updates["email"] = payload.email.lower()
        if payload.roles is not None:
            updates["roles"] = json.dumps(payload.roles)
        if payload.status is not None:
            updates["status"] = payload.status
        if not updates:
            return self.get_user(payload.tenant_id, payload.user_id)
        updates["updated_at"] = _now_iso()
        set_clause = ", ".join(f"{column} = ?" for column in updates)
        params = list(updates.values()) + [payload.tenant_id, payload.user_id]
        db.execute(
            f"UPDATE users SET {set_clause} WHERE tenant_id = ? AND id = ?",
            params,
        )
        return self.get_user(payload.tenant_id, payload.user_id)

    def get_user(self, tenant_id: str, user_id: str) -> UserSummary:
        rows = db.query(
            "SELECT * FROM users WHERE tenant_id = ? AND id = ?",
            (tenant_id, user_id),
            read_only=True,
        )
        if not rows:
            raise ValueError("User not found")
        return self._row_to_user(rows[0])

    def get_user_by_email(self, tenant_id: str, email: str) -> UserSummary:
        rows = db.query(
            "SELECT * FROM users WHERE tenant_id = ? AND email = ?",
            (tenant_id, email.lower()),
            read_only=True,
        )
        if not rows:
            raise ValueError("User not found")
        return self._row_to_user(rows[0])

    def list_users(self, tenant_id: str) -> List[UserSummary]:
        rows = db.query(
            "SELECT * FROM users WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
            read_only=True,
        )
        return [self._row_to_user(row) for row in rows]

    def authenticate(self, payload: LoginRequest) -> LoginResponse:
        user = self.get_user_by_email(payload.tenant_id, payload.email)
        if not self._verify_password(payload.password, user):
            raise AuthorizationError("Invalid credentials")
        token_payload = TokenCreate(tenant_id=user.tenant_id, user_id=user.user_id, scopes=user.roles)
        token_response = self.issue_token(token_payload)
        return LoginResponse(token=token_response.token, token_id=token_response.token_id, expires_at=token_response.expires_at, roles=user.roles)

    def issue_token(self, payload: TokenCreate) -> TokenResponse:
        user = self.get_user(payload.tenant_id, payload.user_id)
        ttl = payload.expires_in or token_signer.ttl_seconds
        expires_at_dt = datetime.utcnow() + timedelta(seconds=ttl)
        token_id = secrets.token_hex(16)
        token = token_signer.encode(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            roles=payload.scopes or user.roles,
            token_id=token_id,
            ttl_seconds=ttl,
        )
        db.execute(
            """
            INSERT OR REPLACE INTO api_tokens (id, tenant_id, user_id, token_hash, scopes, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_id,
                user.tenant_id,
                user.user_id,
                hash_token(token),
                json.dumps(payload.scopes or user.roles),
                _now_iso(),
                expires_at_dt.replace(microsecond=0).isoformat() + "Z",
            ),
        )
        return TokenResponse(token=token, token_id=token_id, expires_at=expires_at_dt)

    def _row_to_user(self, row: Mapping[str, Any]) -> UserSummary:
        roles = row["roles"]
        if isinstance(roles, str):
            roles_list = json.loads(roles)
        else:
            roles_list = list(roles)
        return UserSummary(
            tenant_id=row["tenant_id"],
            user_id=row["id"],
            email=row["email"],
            roles=roles_list,
            status=row.get("status", "active"),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "")),
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "")),
        )

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self._password_iterations,
        )
        return f"{base64.b64encode(salt).decode()}${base64.b64encode(derived).decode()}"

    def _verify_password(self, password: str, user: UserSummary) -> bool:
        rows = db.query(
            "SELECT password_hash FROM users WHERE tenant_id = ? AND id = ?",
            (user.tenant_id, user.user_id),
            read_only=True,
        )
        if not rows:
            return False
        stored = rows[0]["password_hash"]
        try:
            salt_b64, derived_b64 = stored.split("$")
        except ValueError:
            return False
        salt = base64.b64decode(salt_b64)
        derived = base64.b64decode(derived_b64)
        attempt = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self._password_iterations,
        )
        return hmac.compare_digest(derived, attempt)


class AuthorizationService:
    def __init__(self, identity_service: IdentityService) -> None:
        self._identity = identity_service

    def require_tenant(self, principal: Principal, tenant_id: str) -> None:
        if principal.tenant_id != tenant_id and not principal.has_role("platform:admin"):
            raise AuthorizationError("Tenant mismatch")

    def require_user(self, principal: Principal, user_id: str) -> None:
        if principal.user_id == user_id:
            return
        if principal.has_role("tenant:admin", "platform:admin"):
            return
        raise AuthorizationError("Forbidden for requested user")

    def require_roles(self, principal: Principal, *roles: str) -> None:
        if not principal.has_role(*roles):
            raise AuthorizationError("Required role missing")

    def ensure_document_access(self, principal: Principal, tenant_id: str, document_id: int) -> None:
        self.require_tenant(principal, tenant_id)
        rows = db.query(
            "SELECT user_id, sensitivity FROM documents WHERE tenant_id = ? AND id = ?",
            (tenant_id, document_id),
            read_only=True,
        )
        if not rows:
            raise AuthorizationError("Document not found")
        owner = rows[0]["user_id"]
        sensitivity = rows[0]["sensitivity"]
        if sensitivity == "restricted" and not principal.has_role("compliance", "tenant:admin", "platform:admin"):
            raise AuthorizationError("Insufficient clearance for document")
        if owner != principal.user_id and not principal.has_role("tenant:admin", "platform:admin"):
            raise AuthorizationError("Document owned by another user")


class OperationsService:
    def record_backup(self, tenant_id: str, backup_type: str, location: str, triggered_by: str, checksum: str | None = None) -> BackupRecord:
        now = _now_iso()
        result = db.execute(
            """
            INSERT INTO data_backups (tenant_id, backup_type, location, checksum, triggered_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, backup_type, location, checksum, triggered_by, now),
        )
        backup_id = result.lastrowid
        return BackupRecord(
            id=backup_id,
            tenant_id=tenant_id,
            backup_type=backup_type,
            location=location,
            checksum=checksum,
            triggered_by=triggered_by,
            created_at=datetime.fromisoformat(now.replace("Z", "")),
        )

    def perform_backup(
        self,
        tenant_id: str,
        *,
        backup_type: str,
        triggered_by: str,
        target_directory: str | None = None,
    ) -> BackupRecord:
        location, checksum = db.create_backup(
            tenant_id,
            backup_type=backup_type,
            target_directory=target_directory,
        )
        record = self.record_backup(
            tenant_id,
            backup_type,
            location,
            triggered_by,
            checksum=checksum,
        )
        self.record_event(
            tenant_id=tenant_id,
            event_type="backup.completed",
            severity="info",
            message=f"Backup completed to {location}",
            metadata={
                "backup_type": backup_type,
                "checksum": checksum,
                "triggered_by": triggered_by,
            },
        )
        return record

    def record_event(
        self,
        *,
        tenant_id: str,
        event_type: str,
        severity: str,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> OpsEventRecord:
        now = _now_iso()
        result = db.execute(
            """
            INSERT INTO ops_events (tenant_id, event_type, severity, message, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, event_type, severity, message, json.dumps(metadata or {}), now),
        )
        event_id = result.lastrowid
        return OpsEventRecord(
            id=event_id,
            tenant_id=tenant_id,
            event_type=event_type,
            severity=severity,
            message=message,
            metadata=dict(metadata or {}),
            created_at=datetime.fromisoformat(now.replace("Z", "")),
        )

    def list_events(self, tenant_id: str, limit: int = 50) -> List[OpsEventRecord]:
        rows = db.query(
            "SELECT * FROM ops_events WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
            read_only=True,
        )
        events: List[OpsEventRecord] = []
        for row in rows:
            metadata = row.get("metadata")
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            events.append(
                OpsEventRecord(
                    id=row["id"],
                    tenant_id=row["tenant_id"],
                    event_type=row["event_type"],
                    severity=row["severity"],
                    message=row["message"],
                    metadata=metadata_dict,
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "")),
                )
            )
        return events

    def list_backups(self, tenant_id: str, limit: int = 20) -> List[BackupRecord]:
        rows = db.query(
            "SELECT * FROM data_backups WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
            read_only=True,
        )
        backups: List[BackupRecord] = []
        for row in rows:
            backups.append(
                BackupRecord(
                    id=row["id"],
                    tenant_id=row["tenant_id"],
                    backup_type=row["backup_type"],
                    location=row["location"],
                    checksum=row.get("checksum"),
                    triggered_by=row["triggered_by"],
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "")),
                )
            )
        return backups


class LLMReliabilityService:
    def record_failure(
        self,
        *,
        tenant_id: str,
        user_id: str | None,
        operation: str,
        payload: Mapping[str, Any] | None,
        error: str,
    ) -> None:
        db.execute(
            """
            INSERT INTO llm_failures (tenant_id, user_id, operation, payload, error, occurred_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                user_id,
                operation,
                json.dumps(payload or {}),
                error,
                _now_iso(),
            ),
        )

    def list_failures(self, tenant_id: str, limit: int = 20) -> List[LlmFailureRecord]:
        rows = db.query(
            """
            SELECT * FROM llm_failures WHERE tenant_id = ? ORDER BY occurred_at DESC LIMIT ?
            """,
            (tenant_id, limit),
            read_only=True,
        )
        records: List[LlmFailureRecord] = []
        for row in rows:
            payload = row.get("payload")
            payload_dict = json.loads(payload) if isinstance(payload, str) else payload
            records.append(
                LlmFailureRecord(
                    id=row["id"],
                    tenant_id=row["tenant_id"],
                    user_id=row.get("user_id"),
                    operation=row["operation"],
                    error=row["error"],
                    payload=payload_dict,
                    occurred_at=datetime.fromisoformat(row["occurred_at"].replace("Z", "")),
                    retried=bool(row.get("retried", 0)),
                )
            )
        return records


class MLPipelineService:
    def create_dataset_snapshot(self, tenant_id: str, name: str, source: str, snapshot: Mapping[str, Any], created_by: str) -> DatasetRecord:
        now = _now_iso()
        result = db.execute(
            """
            INSERT INTO model_datasets (tenant_id, name, source, snapshot, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, name, source, json.dumps(snapshot), now, created_by),
        )
        dataset_id = result.lastrowid
        return DatasetRecord(
            id=dataset_id,
            tenant_id=tenant_id,
            name=name,
            source=source,
            snapshot=dict(snapshot),
            created_at=datetime.fromisoformat(now.replace("Z", "")),
            created_by=created_by,
        )

    def start_training_run(self, tenant_id: str, dataset_id: int, run_type: str, notes: str | None = None) -> ModelRunRecord:
        now = _now_iso()
        result = db.execute(
            """
            INSERT INTO model_runs (tenant_id, dataset_id, run_type, status, started_at, notes)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (tenant_id, dataset_id, run_type, now, notes),
        )
        run_id = result.lastrowid
        return ModelRunRecord(
            id=run_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            run_type=run_type,
            status="running",
            started_at=datetime.fromisoformat(now.replace("Z", "")),
            completed_at=None,
            notes=notes,
            metrics=[],
        )

    def complete_run(self, tenant_id: str, run_id: int, status: str, metrics: Mapping[str, float]) -> ModelRunRecord:
        finished_at = _now_iso()
        db.execute(
            """
            UPDATE model_runs SET status = ?, completed_at = ? WHERE tenant_id = ? AND id = ?
            """,
            (status, finished_at, tenant_id, run_id),
        )
        metric_records: List[ModelMetric] = []
        for metric, value in metrics.items():
            db.execute(
                """
                INSERT INTO model_run_metrics (tenant_id, run_id, metric, value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tenant_id, run_id, metric, value, finished_at),
            )
            metric_records.append(
                ModelMetric(
                    run_id=run_id,
                    metric=metric,
                    value=value,
                    recorded_at=datetime.fromisoformat(finished_at.replace("Z", "")),
                )
            )
        return ModelRunRecord(
            id=run_id,
            tenant_id=tenant_id,
            dataset_id=self._get_dataset_id(run_id),
            run_type=self._get_run_type(run_id),
            status=status,
            started_at=self._get_run_start(run_id),
            completed_at=datetime.fromisoformat(finished_at.replace("Z", "")),
            notes=self._get_run_notes(run_id),
            metrics=metric_records,
        )

    def _get_dataset_id(self, run_id: int) -> int:
        rows = db.query("SELECT dataset_id FROM model_runs WHERE id = ?", (run_id,), read_only=True)
        return rows[0]["dataset_id"]

    def _get_run_type(self, run_id: int) -> str:
        rows = db.query("SELECT run_type FROM model_runs WHERE id = ?", (run_id,), read_only=True)
        return rows[0]["run_type"]

    def _get_run_notes(self, run_id: int) -> str | None:
        rows = db.query("SELECT notes FROM model_runs WHERE id = ?", (run_id,), read_only=True)
        return rows[0].get("notes")

    def _get_run_start(self, run_id: int) -> datetime:
        rows = db.query("SELECT started_at FROM model_runs WHERE id = ?", (run_id,), read_only=True)
        return datetime.fromisoformat(rows[0]["started_at"].replace("Z", ""))

    def list_datasets(self, tenant_id: str, limit: int = 20) -> List[DatasetRecord]:
        rows = db.query(
            "SELECT * FROM model_datasets WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
            read_only=True,
        )
        datasets: List[DatasetRecord] = []
        for row in rows:
            snapshot = row.get("snapshot")
            snapshot_dict = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
            datasets.append(
                DatasetRecord(
                    id=row["id"],
                    tenant_id=row["tenant_id"],
                    name=row["name"],
                    source=row["source"],
                    snapshot=snapshot_dict,
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "")),
                    created_by=row["created_by"],
                )
            )
        return datasets

    def list_runs(self, tenant_id: str, limit: int = 20) -> List[ModelRunRecord]:
        rows = db.query(
            "SELECT * FROM model_runs WHERE tenant_id = ? ORDER BY started_at DESC LIMIT ?",
            (tenant_id, limit),
            read_only=True,
        )
        runs: List[ModelRunRecord] = []
        for row in rows:
            metrics = db.query(
                "SELECT * FROM model_run_metrics WHERE run_id = ? ORDER BY recorded_at",
                (row["id"],),
                read_only=True,
            )
            metric_models = [
                ModelMetric(
                    run_id=metric_row["run_id"],
                    metric=metric_row["metric"],
                    value=metric_row["value"],
                    recorded_at=datetime.fromisoformat(metric_row["recorded_at"].replace("Z", "")),
                )
                for metric_row in metrics
            ]
            runs.append(
                ModelRunRecord(
                    id=row["id"],
                    tenant_id=row["tenant_id"],
                    dataset_id=row["dataset_id"],
                    run_type=row["run_type"],
                    status=row["status"],
                    started_at=datetime.fromisoformat(row["started_at"].replace("Z", "")),
                    completed_at=(
                        datetime.fromisoformat(row["completed_at"].replace("Z", ""))
                        if row.get("completed_at")
                        else None
                    ),
                    notes=row.get("notes"),
                    metrics=metric_models,
                )
            )
        return runs

DEADLINE_PATTERN = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d+ ימים|\d+ שבועות|עד ל?יום [^,.\n]+)")
LABEL_TITLES = {"obligation": "התחייבות", "risk": "סיכון", "deadline": "מועד"}


@dataclass
class SegmentScores:
    sentence: str
    scores: dict[str, float]

    def overall(self) -> float:
        return max(self.scores.values()) if self.scores else 0.0


class AuditService:
    def log_action(
        self,
        action: str,
        *,
        tenant_id: str,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
        db.execute(
            """
            INSERT INTO audit_logs (tenant_id, user_id, action, resource_type, resource_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, user_id, action, resource_type, resource_id, payload, _now_iso()),
        )

    def list_entries(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        user_id: str | None = None,
        action: str | None = None,
    ) -> List[AuditLogEntry]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if action:
            clauses.append("action = ?")
            params.append(action)
        query = (
            "SELECT id, user_id, action, resource_type, resource_id, metadata, created_at "
            "FROM audit_logs WHERE " + " AND ".join(clauses) + " ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        rows = db.query(query, params)
        entries: list[AuditLogEntry] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str) and metadata:
                try:
                    metadata_dict = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata_dict = {"raw": metadata}
            else:
                metadata_dict = None
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", ""))
            entries.append(
                AuditLogEntry(
                    id=row["id"],
                    user_id=row["user_id"],
                    action=row["action"],
                    resource_type=row["resource_type"],
                    resource_id=row["resource_id"],
                    metadata=metadata_dict,
                    created_at=created_at,
                )
            )
        return entries


class DocumentService:
    PREVIEW_LENGTH = 160

    def __init__(
        self,
        classifier: LegalInsightClassifier,
        _: Path | None = None,
        *,
        llm_client: ClaudeClient | None = None,
        audit_service: "AuditService" | None = None,
        ml_pipeline_service: MLPipelineService | None = None,
        reliability_service: LLMReliabilityService | None = None,
        operations_service: OperationsService | None = None,
    ) -> None:
        self._classifier = classifier
        self._llm = llm_client
        self._audit = audit_service or AuditService()
        self._ml_pipeline = ml_pipeline_service or MLPipelineService()
        self._reliability = reliability_service or LLMReliabilityService()
        self._operations = operations_service or OperationsService()
        self._tenant_indexes: dict[str, DocumentVectorStore] = {}

    def save_document(
        self,
        filename: str,
        raw_content: bytes,
        *,
        tenant_id: str,
        user_id: str,
        retention_policy: str | None = None,
        sensitivity: str | None = None,
        change_note: str | None = None,
    ) -> DocumentSummary:
        text = self._decode_content(raw_content)
        uploaded_at = _now_iso()
        cursor = db.execute(
            """
            INSERT INTO documents (tenant_id, filename, content, uploaded_at, user_id, latest_version, retention_policy, sensitivity)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                tenant_id,
                filename,
                text,
                uploaded_at,
                user_id,
                retention_policy or "standard",
                sensitivity or "internal",
            ),
        )
        doc_id = cursor.lastrowid
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        db.execute(
            """
            INSERT INTO document_versions (tenant_id, document_id, version, content, checksum, created_at, created_by, change_note)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (tenant_id, doc_id, text, checksum, uploaded_at, user_id, change_note),
        )
        self._audit.log_action(
            "document.created",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="document",
            resource_id=str(doc_id),
            metadata={
                "filename": filename,
                "retention_policy": retention_policy or "standard",
                "sensitivity": sensitivity or "internal",
            },
        )
        self.refresh_index(tenant_id)
        return DocumentSummary(
            id=doc_id,
            filename=filename,
            size=len(text),
            uploaded_at=datetime.fromisoformat(uploaded_at[:-1]),
            latest_version=1,
            retention_policy=retention_policy or "standard",
            sensitivity=sensitivity or "internal",
            preview=self._build_preview(text),
            owner_id=user_id,
        )

    def list_documents(self, tenant_id: str, *, user_id: str | None = None) -> List[DocumentSummary]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        query = (
            "SELECT id, filename, content, uploaded_at, latest_version, retention_policy, sensitivity, user_id "
            "FROM documents WHERE "
            + " AND ".join(clauses)
            + " ORDER BY uploaded_at DESC"
        )
        rows = db.query(query, params)
        summaries = [
            DocumentSummary(
                id=row["id"],
                filename=row["filename"],
                size=len(row["content"]),
                uploaded_at=datetime.fromisoformat(row["uploaded_at"].replace("Z", "")),
                latest_version=row["latest_version"],
                retention_policy=row["retention_policy"],
                sensitivity=row["sensitivity"],
                preview=self._build_preview(row["content"]),
                owner_id=row["user_id"],
            )
            for row in rows
        ]
        return summaries

    def get_document(
        self,
        tenant_id: str,
        doc_id: int,
        *,
        requester_id: str | None = None,
    ) -> DocumentDetails:
        rows = db.query(
            """
            SELECT id, filename, content, uploaded_at, latest_version, retention_policy, sensitivity, user_id
            FROM documents
            WHERE tenant_id = ? AND id = ?
            """,
            (tenant_id, doc_id),
        )
        if not rows:
            raise ValueError("Document not found")
        row = rows[0]
        owner_id = row["user_id"]
        sensitivity = row["sensitivity"]
        if sensitivity == "restricted" and requester_id != owner_id:
            raise ValueError("Document access denied for sensitivity level")
        insights = self._analyze(
            row["content"],
            tenant_id=tenant_id,
            user_id=requester_id or owner_id,
        )
        dataset_name = f"document-{doc_id}-v{row['latest_version']}"
        existing = db.query(
            "SELECT id FROM model_datasets WHERE tenant_id = ? AND name = ? LIMIT 1",
            (tenant_id, dataset_name),
            read_only=True,
        )
        if not existing:
            try:
                self._ml_pipeline.create_dataset_snapshot(
                    tenant_id,
                    dataset_name,
                    "documents",
                    {
                        "document_id": doc_id,
                        "filename": row["filename"],
                        "insights": insights.model_dump(),
                        "owner_id": owner_id,
                    },
                    requester_id or owner_id,
                )
            except Exception:  # pragma: no cover - defensive path
                logger.debug("Failed to record document dataset snapshot", exc_info=True)
        versions = self._load_versions(tenant_id, doc_id)
        return DocumentDetails(
            id=row["id"],
            filename=row["filename"],
            size=len(row["content"]),
            uploaded_at=datetime.fromisoformat(row["uploaded_at"].replace("Z", "")),
            preview=self._build_preview(row["content"]),
            content=row["content"],
            insights=insights,
            latest_version=row["latest_version"],
            retention_policy=row["retention_policy"],
            sensitivity=sensitivity,
            versions=versions,
            owner_id=owner_id,
        )

    def create_version(
        self,
        doc_id: int,
        raw_content: bytes,
        *,
        tenant_id: str,
        user_id: str,
        change_note: str | None = None,
    ) -> DocumentDetails:
        rows = db.query(
            "SELECT filename, latest_version FROM documents WHERE tenant_id = ? AND id = ?",
            (tenant_id, doc_id),
        )
        if not rows:
            raise ValueError("Document not found")
        text = self._decode_content(raw_content)
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        uploaded_at = _now_iso()
        next_version = rows[0]["latest_version"] + 1
        db.execute(
            "UPDATE documents SET content = ?, uploaded_at = ?, latest_version = ? WHERE tenant_id = ? AND id = ?",
            (text, uploaded_at, next_version, tenant_id, doc_id),
        )
        db.execute(
            """
            INSERT INTO document_versions (tenant_id, document_id, version, content, checksum, created_at, created_by, change_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, doc_id, next_version, text, checksum, uploaded_at, user_id, change_note),
        )
        self._audit.log_action(
            "document.version_created",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="document",
            resource_id=str(doc_id),
            metadata={"version": next_version, "change_note": change_note},
        )
        self.refresh_index(tenant_id)
        return self.get_document(tenant_id, doc_id, requester_id=user_id)

    def update_metadata(
        self,
        doc_id: int,
        *,
        tenant_id: str,
        user_id: str,
        retention_policy: str | None = None,
        sensitivity: str | None = None,
    ) -> DocumentDetails:
        exists = db.query(
            "SELECT 1 FROM documents WHERE tenant_id = ? AND id = ?",
            (tenant_id, doc_id),
        )
        if not exists:
            raise ValueError("Document not found")
        updates: dict[str, Any] = {}
        params: list[Any] = []
        if retention_policy:
            updates["retention_policy"] = retention_policy
        if sensitivity:
            updates["sensitivity"] = sensitivity
        if not updates:
            return self.get_document(tenant_id, doc_id, requester_id=user_id)
        sets = ", ".join(f"{key} = ?" for key in updates)
        params.extend(updates.values())
        params.extend([tenant_id, doc_id])
        db.execute(f"UPDATE documents SET {sets} WHERE tenant_id = ? AND id = ?", params)
        self._audit.log_action(
            "document.metadata_updated",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="document",
            resource_id=str(doc_id),
            metadata=updates,
        )
        return self.get_document(tenant_id, doc_id, requester_id=user_id)

    def delete_document(
        self,
        doc_id: int,
        *,
        tenant_id: str,
        user_id: str,
        reason: str | None = None,
    ) -> None:
        exists = db.query(
            "SELECT id FROM documents WHERE tenant_id = ? AND id = ?",
            (tenant_id, doc_id),
        )
        if not exists:
            raise ValueError("Document not found")
        db.execute(
            "DELETE FROM document_versions WHERE tenant_id = ? AND document_id = ?",
            (tenant_id, doc_id),
        )
        db.execute("DELETE FROM documents WHERE tenant_id = ? AND id = ?", (tenant_id, doc_id))
        self._audit.log_action(
            "document.deleted",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="document",
            resource_id=str(doc_id),
            metadata={"reason": reason or "privacy_request"},
        )
        self.refresh_index(tenant_id)

    def list_versions(self, tenant_id: str, doc_id: int) -> List[DocumentVersion]:
        exists = db.query(
            "SELECT 1 FROM documents WHERE tenant_id = ? AND id = ?",
            (tenant_id, doc_id),
        )
        if not exists:
            raise ValueError("Document not found")
        rows = db.query(
            """
            SELECT version, checksum, created_at, created_by, change_note
            FROM document_versions
            WHERE tenant_id = ? AND document_id = ?
            ORDER BY version DESC
            """,
            (tenant_id, doc_id),
        )
        versions: list[DocumentVersion] = []
        for row in rows:
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", ""))
            versions.append(
                DocumentVersion(
                    version=row["version"],
                    checksum=row["checksum"],
                    created_at=created_at,
                    created_by=row["created_by"],
                    change_note=row["change_note"],
                )
            )
        return versions

    def _load_versions(self, tenant_id: str, doc_id: int) -> List[DocumentVersion]:
        versions = self.list_versions(tenant_id, doc_id)
        if not versions:
            rows = db.query(
                "SELECT uploaded_at, user_id FROM documents WHERE tenant_id = ? AND id = ?",
                (tenant_id, doc_id),
            )
            timestamp = rows[0]["uploaded_at"] if rows else _now_iso()
            created = datetime.fromisoformat(timestamp.replace("Z", ""))
            created_by = rows[0]["user_id"] if rows else None
            versions.append(
                DocumentVersion(
                    version=1,
                    checksum="legacy-record",
                    created_at=created,
                    created_by=created_by,
                    change_note=None,
                )
            )
        return versions

    def search(self, tenant_id: str, query: str, limit: int = 5) -> List[DocumentSearchResult]:
        matches = self.retrieve_contexts(tenant_id, query, limit)
        return [
            DocumentSearchResult(
                document_id=match.segment.document_id,
                filename=match.segment.filename,
                snippet=textwrap.shorten(match.segment.text.replace("\n", " "), width=200, placeholder="..."),
                score=match.score,
            )
            for match in matches
        ]

    def retrieve_contexts(self, tenant_id: str, query: str, limit: int = 5) -> List[DocumentMatch]:
        store = self._tenant_indexes.get(tenant_id)
        if store is None:
            self.refresh_index(tenant_id)
            store = self._tenant_indexes.get(tenant_id)
        if store is None:
            return []
        matches = store.search(query, limit)
        for match in matches:
            explanation = self._explain_segment(match.segment.text)
            if explanation:
                match.explanation = explanation
        return matches

    def refresh_index(self, tenant_id: str) -> None:
        rows = db.query(
            "SELECT id, filename, content FROM documents WHERE tenant_id = ?",
            (tenant_id,),
        )
        segments: list[DocumentSegment] = []
        for row in rows:
            chunks = self._split_into_segments(row["content"])
            for order, chunk in enumerate(chunks):
                segments.append(
                    DocumentSegment(
                        document_id=row["id"],
                        filename=row["filename"],
                        text=chunk,
                        order=order,
                    )
                )
        store = DocumentVectorStore()
        store.rebuild(segments)
        self._tenant_indexes[tenant_id] = store

    def _decode_content(self, raw: bytes) -> str:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")

    def _build_preview(self, text: str) -> str:
        return textwrap.shorten(" ".join(text.split()), width=self.PREVIEW_LENGTH, placeholder="...")

    def _split_into_segments(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        segment: list[str] = []
        segments: list[str] = []
        for sentence in sentences:
            segment.append(sentence)
            if len(" ".join(segment)) > 320:
                segments.append(" ".join(segment))
                segment = []
        if segment:
            segments.append(" ".join(segment))
        if not segments and text:
            segments.append(text)
        return segments

    def _split_sentences(self, text: str) -> List[str]:
        text = text.replace("\r", " ")
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", text)
        sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]
        return sentences

    def _analyze(self, text: str, *, tenant_id: str, user_id: str | None) -> DocumentInsights:
        sentences = self._split_sentences(text)
        sentence_scores = self._classifier.predict_scores(sentences)
        scored = [SegmentScores(sentence=sentence, scores=scores) for sentence, scores in zip(sentences, sentence_scores)]

        key_points = [item.sentence for item in sorted(scored, key=lambda s: s.overall(), reverse=True)[:5]]
        obligations = [item.sentence for item in scored if item.scores.get("obligation", 0) >= 0.45]
        risks = [item.sentence for item in scored if item.scores.get("risk", 0) >= 0.4]
        deadlines = self._extract_deadlines(scored)

        recommended_actions = self._build_recommendations(obligations, risks, deadlines)
        rationale = self._build_rationale(scored)
        confidence = self._confidence_from_scores(scored)

        insights = DocumentInsights(
            key_points=key_points or ["לא זוהו נקודות מרכזיות מובהקות"],
            obligations=obligations,
            risks=risks,
            deadlines=deadlines,
            recommended_actions=recommended_actions,
            rationale=rationale,
            confidence_score=confidence,
        )
        return self._refine_with_llm(
            text,
            scored,
            insights,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    def _extract_deadlines(self, scored: Sequence[SegmentScores]) -> List[str]:
        deadlines: list[str] = []
        for item in scored:
            if item.scores.get("deadline", 0) >= 0.4 or DEADLINE_PATTERN.search(item.sentence):
                deadlines.append(item.sentence)
        return deadlines

    def _build_recommendations(
        self,
        obligations: Sequence[str],
        risks: Sequence[str],
        deadlines: Sequence[str],
    ) -> List[str]:
        actions: list[str] = []
        if obligations:
            actions.append("להעביר את סעיפי ההתחייבות לבדיקת עורך דין אחראי ולהבטיח תיעוד חתום.")
        if deadlines:
            actions.append("לשריין תזכורות אוטומטיות לפני מועדים קריטיים ולהכין בקשות לדחייה במידת הצורך.")
        if risks:
            actions.append("לנתח את סעיפי הסיכון מול הביטחונות הקיימים ולבחון חיזוק באמצעות ערבויות.")
        if not actions:
            actions.append("לא נמצאו דגלים חריגים – מומלץ לבצע סקירה חוזרת לפני הגשה לבית המשפט.")
        return actions

    def _build_rationale(self, scored: Sequence[SegmentScores]) -> List[str]:
        rationale: list[str] = []
        for label in ("obligation", "risk", "deadline"):
            best = max(scored, key=lambda item: item.scores.get(label, 0), default=None)
            if best and best.scores.get(label, 0) >= 0.35:
                keywords = self._classifier.explain(best.sentence, label)
                explanation = ", ".join(keywords) if keywords else "ביטויים חוזרים בטקסט"
                label_title = LABEL_TITLES.get(label, label)
                rationale.append(
                    f"המודל זיהה {label_title} במשפט: '{best.sentence}' (משקל {best.scores[label]:.0%}) עקב {explanation}."
                )
        return rationale

    def _confidence_from_scores(self, scored: Sequence[SegmentScores]) -> float:
        if not scored:
            return 0.2
        top_scores = sorted((item.overall() for item in scored), reverse=True)[:5]
        confidence = min(0.95, 0.3 + sum(top_scores) / (len(top_scores) * 2))
        return max(0.2, confidence)

    def _explain_segment(self, text: str) -> str | None:
        scores = self._classifier.predict_scores([text])[0]
        if not scores:
            return None
        label, value = max(scores.items(), key=lambda item: item[1])
        if value < 0.35:
            return None
        keywords = self._classifier.explain(text, label)
        summary = ", ".join(keywords[:2]) if keywords else "ביטויים חוזרים"
        label_title = LABEL_TITLES.get(label, label)
        return f"{label_title} ({value:.0%}) – {summary}"

    def as_reference(self, match: DocumentMatch) -> ContextualReference:
        return ContextualReference(
            document_id=match.segment.document_id,
            filename=match.segment.filename,
            snippet=textwrap.shorten(match.segment.text.replace("\n", " "), width=200, placeholder="..."),
            score=match.score,
            explanation=match.explanation,
        )

    def _refine_with_llm(
        self,
        document_text: str,
        scored: Sequence[SegmentScores],
        insights: DocumentInsights,
        *,
        tenant_id: str,
        user_id: str | None,
    ) -> DocumentInsights:
        if not self._llm or not self._llm.is_configured:
            return insights

        exemplar_sentences = [item.sentence for item in sorted(scored, key=lambda s: s.overall(), reverse=True)[:6]]
        payload = {
            "summary": insights.key_points,
            "obligations": insights.obligations,
            "risks": insights.risks,
            "deadlines": insights.deadlines,
            "recommended_actions": insights.recommended_actions,
            "rationale": insights.rationale,
            "confidence": insights.confidence_score,
            "candidate_sentences": exemplar_sentences,
            "document_excerpt": textwrap.shorten(document_text.replace("\n", " "), width=900, placeholder=" ..."),
        }
        system_prompt = (
            "אתה מנתח מסמכים משפטיים עבור צוות ליטיגציה. קבל תמצית קיימת והצע עדכון משופר "
            "בפורמט JSON. שדות החובה: key_points, obligations, risks, deadlines, recommended_actions, rationale, "
            "confidence_adjustment (מספר בין -0.2 ל-0.2), quality_notes (רשימה). החזר תשובה קפדנית בעברית."
        )

        try:
            raw = self._llm.generate(
                system_prompt,
                [
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    }
                ],
                max_tokens=850,
            )
            data = json.loads(raw)
        except LLMGenerationError as exc:  # pragma: no cover - network path
            logger.warning("Claude insight refinement failed, using heuristic output: %s", exc)
            self._reliability.record_failure(
                tenant_id=tenant_id,
                user_id=user_id,
                operation="document_insight_refinement",
                payload={"document_excerpt": textwrap.shorten(document_text, width=200)},
                error=str(exc),
            )
            self._operations.record_event(
                tenant_id=tenant_id,
                event_type="llm.failure",
                severity="warning",
                message="LLM refinement failed; heuristics applied",
                metadata={"error": str(exc)[:160]},
            )
            return insights
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Claude insight refinement returned non-JSON response", exc_info=exc)
            self._reliability.record_failure(
                tenant_id=tenant_id,
                user_id=user_id,
                operation="document_insight_refinement",
                payload={"raw": raw},
                error="Invalid JSON response",
            )
            return insights

        updated = insights.model_dump()
        changed = False

        def _update_list(name: str) -> None:
            nonlocal changed
            value = data.get(name)
            if isinstance(value, list) and value:
                updated[name] = [str(item).strip() for item in value if str(item).strip()]
                changed = True

        _update_list("key_points")
        _update_list("obligations")
        _update_list("risks")
        _update_list("deadlines")
        _update_list("recommended_actions")

        rationale_value = data.get("rationale")
        if isinstance(rationale_value, list) and rationale_value:
            updated["rationale"] = [str(item).strip() for item in rationale_value if str(item).strip()]
            changed = True
        elif isinstance(rationale_value, str) and rationale_value.strip():
            updated["rationale"] = [rationale_value.strip()]
            changed = True

        adjustment = data.get("confidence_adjustment")
        if isinstance(adjustment, (int, float)):
            updated["confidence_score"] = max(0.0, min(1.0, updated["confidence_score"] + float(adjustment)))
            changed = True

        quality_notes = data.get("quality_notes")
        if isinstance(quality_notes, list) and quality_notes:
            updated.setdefault("rationale", []).extend(
                str(item).strip() for item in quality_notes if str(item).strip()
            )
            changed = True
        elif isinstance(quality_notes, str) and quality_notes.strip():
            updated.setdefault("rationale", []).append(quality_notes.strip())
            changed = True

        if not changed:
            return insights

        updated["model_version"] = "lexiai-insights-claude"
        return DocumentInsights(**updated)


class PrivacyService:
    TERMINAL_STATUSES = {"completed", "rejected"}

    def __init__(self, document_service: DocumentService, audit_service: AuditService) -> None:
        self._documents = document_service
        self._audit = audit_service

    def create_request(self, payload: PrivacyRequestCreate) -> PrivacyRequest:
        requested_at = _now_iso()
        cursor = db.execute(
            """
            INSERT INTO privacy_requests (tenant_id, user_id, request_type, resource_type, resource_id, reason, status, requested_at)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                payload.tenant_id,
                payload.user_id,
                payload.request_type,
                payload.resource_type,
                payload.resource_id,
                payload.reason,
                requested_at,
            ),
        )
        request_id = cursor.lastrowid
        self._audit.log_action(
            "privacy.request_created",
            tenant_id=payload.tenant_id,
            user_id=payload.user_id,
            resource_type=payload.resource_type,
            resource_id=payload.resource_id,
            metadata={"request_type": payload.request_type, "reason": payload.reason},
        )
        request = self.get_request(payload.tenant_id, request_id)
        if (
            request.request_type == "erasure"
            and request.resource_type == "document"
            and request.resource_id is not None
        ):
            try:
                self._documents.delete_document(
                    int(request.resource_id),
                    tenant_id=payload.tenant_id,
                    user_id=request.user_id,
                    reason=request.reason,
                )
            except ValueError:
                request = self.update_request(
                    payload.tenant_id,
                    request_id,
                    PrivacyRequestUpdate(
                        status="rejected",
                        resolution_note="המסמך כבר לא קיים בארכיון",
                    ),
                )
            else:
                request = self.update_request(
                    payload.tenant_id,
                    request_id,
                    PrivacyRequestUpdate(
                        status="completed",
                        resolution_note="המסמך הוסר בהתאם לבקשת המשתמש",
                    ),
                )
        return request

    def get_request(self, tenant_id: str, request_id: int) -> PrivacyRequest:
        rows = db.query(
            "SELECT * FROM privacy_requests WHERE tenant_id = ? AND id = ?",
            (tenant_id, request_id),
        )
        if not rows:
            raise ValueError("Privacy request not found")
        return self._row_to_privacy_request(rows[0])

    def list_requests(self, tenant_id: str, status: str | None = None) -> List[PrivacyRequest]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if status:
            clauses.append("status = ?")
            params.append(status)
        query = (
            "SELECT * FROM privacy_requests WHERE "
            + " AND ".join(clauses)
            + " ORDER BY requested_at DESC"
        )
        rows = db.query(query, params)
        return [self._row_to_privacy_request(row) for row in rows]

    def update_request(self, tenant_id: str, request_id: int, payload: PrivacyRequestUpdate) -> PrivacyRequest:
        rows = db.query(
            "SELECT * FROM privacy_requests WHERE tenant_id = ? AND id = ?",
            (tenant_id, request_id),
        )
        if not rows:
            raise ValueError("Privacy request not found")
        resolved_at = _now_iso() if payload.status in self.TERMINAL_STATUSES else None
        db.execute(
            "UPDATE privacy_requests SET status = ?, resolution_note = ?, resolved_at = ? WHERE tenant_id = ? AND id = ?",
            (payload.status, payload.resolution_note, resolved_at, tenant_id, request_id),
        )
        request = self.get_request(tenant_id, request_id)
        self._audit.log_action(
            "privacy.request_updated",
            tenant_id=tenant_id,
            user_id=request.user_id,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            metadata={"status": request.status, "resolution_note": request.resolution_note},
        )
        return request

    def _row_to_privacy_request(self, row: Mapping[str, Any]) -> PrivacyRequest:
        requested_at = datetime.fromisoformat(row["requested_at"].replace("Z", ""))
        resolved_at = (
            datetime.fromisoformat(row["resolved_at"].replace("Z", ""))
            if row["resolved_at"]
            else None
        )
        return PrivacyRequest(
            id=row["id"],
            user_id=row["user_id"],
            request_type=row["request_type"],
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            reason=row["reason"],
            status=row["status"],
            requested_at=requested_at,
            resolved_at=resolved_at,
            resolution_note=row["resolution_note"],
        )


class WorkflowService:
    DEFAULT_STATUS = "backlog"

    def __init__(self, audit_service: AuditService) -> None:
        self._audit = audit_service

    def create_task(self, payload: WorkflowTaskCreate) -> WorkflowTask:
        created_at = _now_iso()
        due_date = payload.due_date.isoformat() + "Z" if payload.due_date else None
        tags = json.dumps(payload.tags, ensure_ascii=False)
        cursor = db.execute(
            """
            INSERT INTO workflow_tasks (tenant_id, case_id, title, status, assignee, due_date, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.tenant_id,
                payload.case_id,
                payload.title,
                self.DEFAULT_STATUS,
                payload.assignee,
                due_date,
                created_at,
                created_at,
                tags,
            ),
        )
        task_id = cursor.lastrowid
        self._audit.log_action(
            "workflow.task_created",
            tenant_id=payload.tenant_id,
            resource_type="workflow_task",
            resource_id=str(task_id),
            metadata={"case_id": payload.case_id, "assignee": payload.assignee},
        )
        return self.get_task(payload.tenant_id, task_id)

    def get_task(self, tenant_id: str, task_id: int) -> WorkflowTask:
        rows = db.query(
            "SELECT * FROM workflow_tasks WHERE tenant_id = ? AND id = ?",
            (tenant_id, task_id),
        )
        if not rows:
            raise ValueError("Task not found")
        return self._row_to_task(rows[0])

    def list_tasks(
        self,
        tenant_id: str,
        case_id: str | None = None,
        status: str | None = None,
    ) -> List[WorkflowTask]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if case_id:
            clauses.append("case_id = ?")
            params.append(case_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        query = (
            "SELECT * FROM workflow_tasks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY due_date IS NULL, due_date ASC, updated_at DESC"
        )
        rows = db.query(query, params)
        return [self._row_to_task(row) for row in rows]

    def update_task(self, tenant_id: str, task_id: int, payload: WorkflowTaskUpdate) -> WorkflowTask:
        rows = db.query(
            "SELECT * FROM workflow_tasks WHERE tenant_id = ? AND id = ?",
            (tenant_id, task_id),
        )
        if not rows:
            raise ValueError("Task not found")
        updates: dict[str, Any] = {}
        if payload.title:
            updates["title"] = payload.title
        if payload.status:
            updates["status"] = payload.status
        if payload.assignee is not None:
            updates["assignee"] = payload.assignee
        if payload.due_date is not None:
            updates["due_date"] = payload.due_date.isoformat() + "Z"
        if payload.tags is not None:
            updates["tags"] = json.dumps(payload.tags, ensure_ascii=False)
        if not updates:
            return self.get_task(tenant_id, task_id)
        updates["updated_at"] = _now_iso()
        set_clause = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [tenant_id, task_id]
        db.execute(f"UPDATE workflow_tasks SET {set_clause} WHERE tenant_id = ? AND id = ?", params)
        task = self.get_task(tenant_id, task_id)
        self._audit.log_action(
            "workflow.task_updated",
            tenant_id=tenant_id,
            resource_type="workflow_task",
            resource_id=str(task_id),
            metadata={"status": task.status, "assignee": task.assignee},
        )
        return task

    def _row_to_task(self, row: Mapping[str, Any]) -> WorkflowTask:
        created_at = datetime.fromisoformat(row["created_at"].replace("Z", ""))
        updated_at = datetime.fromisoformat(row["updated_at"].replace("Z", ""))
        due_date = (
            datetime.fromisoformat(row["due_date"].replace("Z", "")) if row["due_date"] else None
        )
        tags_raw = row["tags"]
        if isinstance(tags_raw, str) and tags_raw:
            try:
                tags = json.loads(tags_raw)
            except json.JSONDecodeError:
                tags = [tags_raw]
        else:
            tags = []
        return WorkflowTask(
            id=row["id"],
            case_id=row["case_id"],
            title=row["title"],
            status=row["status"],
            assignee=row["assignee"],
            due_date=due_date,
            created_at=created_at,
            updated_at=updated_at,
            tags=tags,
        )


class ConversationService:
    def __init__(
        self,
        document_service: DocumentService,
        rag_engine: LegalRagEngine,
        *,
        ml_pipeline_service: MLPipelineService | None = None,
        reliability_service: LLMReliabilityService | None = None,
        operations_service: OperationsService | None = None,
        authorization_service: AuthorizationService | None = None,
    ) -> None:
        self._document_service = document_service
        self._rag_engine = rag_engine
        self._ml_pipeline = ml_pipeline_service or MLPipelineService()
        self._reliability = reliability_service or LLMReliabilityService()
        self._operations = operations_service or OperationsService()
        self._authorization = authorization_service

    def add_message(self, tenant_id: str, user_id: str, role: str, content: str) -> None:
        db.execute(
            "INSERT INTO conversations (tenant_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (tenant_id, user_id, role, content, _now_iso()),
        )

    def get_history(self, tenant_id: str, user_id: str, limit: int = 25) -> List[ChatMessage]:
        rows = db.query(
            """
            SELECT role, content, created_at
            FROM conversations
            WHERE tenant_id = ? AND user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (tenant_id, user_id, limit),
        )
        messages = [
            ChatMessage(role=row["role"], content=row["content"], created_at=datetime.fromisoformat(row["created_at"].replace("Z", "")))
            for row in reversed(rows)
        ]
        return messages

    def generate_response(
        self,
        principal: Principal,
        tenant_id: str,
        prompt: str,
        history: Sequence[ChatMessage],
    ) -> tuple[str, List[ContextualReference]]:
        matches = self._document_service.retrieve_contexts(tenant_id, prompt, limit=5)
        if self._authorization is not None:
            authorised_matches: list[DocumentMatch] = []
            for match in matches:
                try:
                    self._authorization.ensure_document_access(principal, tenant_id, match.segment.document_id)
                except AuthorizationError:
                    continue
                authorised_matches.append(match)
            matches = authorised_matches
        try:
            answer = self._rag_engine.build_answer(prompt, matches)
            response_text = answer.text
        except LLMGenerationError as exc:
            self._reliability.record_failure(
                tenant_id=tenant_id,
                user_id=None,
                operation="chat.generate",
                payload={"prompt": prompt, "matches": len(matches)},
                error=str(exc),
            )
            self._operations.record_event(
                tenant_id=tenant_id,
                event_type="llm.failure",
                severity="warning",
                message="LLM chat generation failed; falling back to heuristic response",
                metadata={"error": str(exc)[:120]},
            )
            response_text = self._fallback_response(prompt, matches)
        references = [self._document_service.as_reference(match) for match in matches]
        snapshot = {
            "prompt": prompt,
            "history": [message.model_dump() for message in history],
            "references": [reference.model_dump() for reference in references],
            "response": response_text,
        }
        try:
            self._ml_pipeline.create_dataset_snapshot(
                tenant_id,
                f"conversation-{_now_iso()}",
                "chat",
                snapshot,
                principal.user_id,
            )
        except Exception:  # pragma: no cover - defensive path
            logger.debug("Failed to record conversation snapshot", exc_info=True)
        return response_text, references

    def _fallback_response(self, prompt: str, matches: Sequence[DocumentMatch]) -> str:
        if matches:
            top = matches[0]
            snippet = textwrap.shorten(top.segment.text.replace("\n", " "), width=280, placeholder="...")
            return (
                "לא הצלחתי להפיק מענה חכם כרגע, אך קטע רלוונטי שנמצא במסמכים הוא: \n"
                f"— {snippet}\nנסה לעדכן את השאלה או לרענן את הדף."
            )
        return "כרגע המערכת לא הצליחה לנתח את השאלה. מומלץ לנסח מחדש או לעיין בתיעוד ההליכים העדכני."


class PredictionService:
    def __init__(
        self,
        classifier: CaseOutcomeClassifier,
        audit_service: AuditService,
        *,
        ml_pipeline_service: MLPipelineService | None = None,
        operations_service: OperationsService | None = None,
        reliability_service: LLMReliabilityService | None = None,
    ) -> None:
        self._classifier = classifier
        self._audit = audit_service
        self._ml_pipeline = ml_pipeline_service or MLPipelineService()
        self._operations = operations_service or OperationsService()
        self._reliability = reliability_service or LLMReliabilityService()

    def create_prediction(self, tenant_id: str, user_id: str, case_details: str) -> PredictionResponse:
        prediction = self._classifier.predict(case_details)
        created_at = datetime.utcnow()
        signals = self._serialize_signals(prediction)
        db.execute(
            """
            INSERT INTO predictions (tenant_id, user_id, case_details, probability, rationale, recommended_actions, signals, quality_warnings, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                user_id,
                case_details,
                prediction.probability,
                prediction.rationale,
                json.dumps(prediction.recommended_actions, ensure_ascii=False),
                json.dumps(signals, ensure_ascii=False),
                json.dumps(prediction.quality_warnings, ensure_ascii=False),
                created_at.isoformat() + "Z",
            ),
        )
        self._audit.log_action(
            "prediction.created",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="prediction",
            resource_id=case_details[:80],
            metadata={"probability": prediction.probability, "signals": len(signals)},
        )
        try:
            dataset = self._ml_pipeline.create_dataset_snapshot(
                tenant_id,
                f"prediction-{created_at.isoformat()}",
                "predictions",
                {
                    "case_details": case_details,
                    "signals": signals,
                    "probability": prediction.probability,
                    "quality": prediction.quality_warnings,
                },
                user_id,
            )
            run = self._ml_pipeline.start_training_run(
                tenant_id,
                dataset.id,
                "online-update",
                notes="Auto-scheduled after prediction",
            )
            self._ml_pipeline.complete_run(
                tenant_id,
                run.id,
                "succeeded",
                {"confidence": prediction.probability, "signals": float(len(signals))},
            )
        except Exception:  # pragma: no cover - defensive path
            logger.debug("Failed to push prediction data into ML pipeline", exc_info=True)
        self._operations.record_event(
            tenant_id=tenant_id,
            event_type="prediction.created",
            severity="info",
            message="נוצר חיזוי חדש למקרה",
            metadata={"user_id": user_id, "probability": prediction.probability},
        )
        return PredictionResponse(
            probability=prediction.probability,
            rationale=prediction.rationale,
            recommended_actions=prediction.recommended_actions,
            created_at=created_at,
            signals=[PredictionSignal(**signal) for signal in signals],
            quality_warnings=prediction.quality_warnings,
        )

    def list_predictions(self, tenant_id: str, user_id: str) -> List[StoredPrediction]:
        rows = db.query(
            """
            SELECT id, case_details, probability, rationale, recommended_actions, signals, quality_warnings, created_at
            FROM predictions
            WHERE tenant_id = ? AND user_id = ?
            ORDER BY created_at DESC
            """,
            (tenant_id, user_id),
        )
        predictions: list[StoredPrediction] = []
        for row in rows:
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", ""))
            probability = (
                float(row["probability"])
                if isinstance(row["probability"], str)
                else row["probability"]
            )
            recommended_actions = json.loads(row["recommended_actions"]) if isinstance(row["recommended_actions"], str) else row["recommended_actions"]
            raw_signals = json.loads(row["signals"]) if isinstance(row["signals"], str) else row["signals"]
            quality = json.loads(row["quality_warnings"]) if isinstance(row["quality_warnings"], str) else row["quality_warnings"]
            predictions.append(
                StoredPrediction(
                    id=row["id"],
                    user_id=user_id,
                    case_details=row["case_details"],
                    probability=probability,
                    rationale=row["rationale"],
                    recommended_actions=recommended_actions,
                    created_at=created_at,
                    signals=[PredictionSignal(**signal) for signal in raw_signals],
                    quality_warnings=quality,
                )
            )
        return predictions

    def _serialize_signals(self, prediction: CasePrediction) -> List[dict[str, object]]:
        signals: list[dict[str, object]] = []
        for token, weight in prediction.signals:
            entry = {"label": token, "weight": round(weight, 4), "direction": "positive"}
            evidence = prediction.signal_explanations.get(token)
            if evidence:
                entry["evidence"] = evidence
            signals.append(entry)
        for token, weight in prediction.negative_signals:
            entry = {"label": token, "weight": round(weight, 4), "direction": "negative"}
            evidence = prediction.signal_explanations.get(token)
            if evidence:
                entry["evidence"] = evidence
            signals.append(entry)
        return signals


class WitnessService:
    def __init__(
        self,
        document_service: DocumentService,
        rag_engine: LegalRagEngine,
        strategy_generator: WitnessStrategyGenerator,
        audit_service: AuditService,
        *,
        ml_pipeline_service: MLPipelineService | None = None,
        operations_service: OperationsService | None = None,
        reliability_service: LLMReliabilityService | None = None,
    ) -> None:
        self._document_service = document_service
        self._rag_engine = rag_engine
        self._strategy_generator = strategy_generator
        self._audit = audit_service
        self._ml_pipeline = ml_pipeline_service or MLPipelineService()
        self._operations = operations_service or OperationsService()
        self._reliability = reliability_service or LLMReliabilityService()

    def create_plan(
        self,
        tenant_id: str,
        user_id: str,
        witness_name: str,
        witness_role: str,
        case_summary: str,
        objectives: Sequence[str],
    ) -> StoredWitnessPlan:
        query = " ".join([case_summary, witness_role, " ".join(objectives)])
        matches = self._document_service.retrieve_contexts(tenant_id, query, limit=6)
        try:
            answer = self._rag_engine.build_answer(case_summary, matches)
        except LLMGenerationError as exc:
            self._reliability.record_failure(
                tenant_id=tenant_id,
                user_id=user_id,
                operation="witness.generate",
                payload={"witness": witness_name, "objectives": list(objectives)},
                error=str(exc),
            )
            self._operations.record_event(
                tenant_id=tenant_id,
                event_type="llm.failure",
                severity="warning",
                message="Witness plan LLM generation failed; returning heuristic plan",
                metadata={"witness_name": witness_name},
            )
            answer = self._rag_engine.build_answer(case_summary, matches)
        payload = self._strategy_generator.generate(witness_role, case_summary, matches, answer.knowledge_references)
        created_at = datetime.utcnow().replace(microsecond=0)

        plan_dict = {
            "strategy": payload.strategy,
            "focus_areas": payload.focus_areas,
            "question_sets": payload.question_sets,
            "risk_controls": payload.risk_controls,
            "quality_notes": payload.quality_notes + answer.guardrails,
            "contextual_references": [self._document_service.as_reference(match).model_dump() for match in matches],
        }
        cursor = db.execute(
            """
            INSERT INTO witness_plans (tenant_id, user_id, witness_name, witness_role, case_summary, plan, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                user_id,
                witness_name,
                witness_role,
                case_summary,
                json.dumps(plan_dict, ensure_ascii=False),
                created_at.isoformat() + "Z",
            ),
        )
        plan_id = cursor.lastrowid
        self._audit.log_action(
            "witness.plan_created",
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="witness_plan",
            resource_id=str(plan_id),
            metadata={"witness_name": witness_name, "objectives": objectives},
        )
        try:
            self._ml_pipeline.create_dataset_snapshot(
                tenant_id,
                f"witness-{plan_id}",
                "witness_plans",
                {
                    "witness": witness_name,
                    "role": witness_role,
                    "objectives": list(objectives),
                    "plan": plan_dict,
                },
                user_id,
            )
        except Exception:  # pragma: no cover - defensive path
            logger.debug("Failed to log witness plan snapshot", exc_info=True)
        self._operations.record_event(
            tenant_id=tenant_id,
            event_type="witness.plan_created",
            severity="info",
            message="נוצרה תוכנית חקירה לעד",
            metadata={"witness_name": witness_name, "user_id": user_id},
        )
        return self._build_plan_response(
            plan_id,
            user_id,
            witness_name,
            witness_role,
            case_summary,
            created_at,
            plan_dict,
        )

    def list_plans(self, tenant_id: str, user_id: str) -> List[StoredWitnessPlan]:
        rows = db.query(
            """
            SELECT id, witness_name, witness_role, case_summary, plan, created_at
            FROM witness_plans
            WHERE tenant_id = ? AND user_id = ?
            ORDER BY created_at DESC
            """,
            (tenant_id, user_id),
        )
        plans: list[StoredWitnessPlan] = []
        for row in rows:
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", ""))
            plan_dict = json.loads(row["plan"]) if isinstance(row["plan"], str) else row["plan"]
            plans.append(
                self._build_plan_response(
                    row["id"],
                    user_id,
                    row["witness_name"],
                    row["witness_role"],
                    row["case_summary"],
                    created_at,
                    plan_dict,
                )
            )
        return plans

    def _build_plan_response(
        self,
        plan_id: int,
        user_id: str,
        witness_name: str,
        witness_role: str,
        case_summary: str,
        created_at: datetime,
        plan_dict: dict,
    ) -> StoredWitnessPlan:
        return StoredWitnessPlan(
            id=plan_id,
            user_id=user_id,
            case_summary=case_summary,
            witness_name=witness_name,
            witness_role=witness_role,
            strategy=plan_dict.get("strategy", ""),
            focus_areas=plan_dict.get("focus_areas", []),
            question_sets=[WitnessQuestionSet(**qs) for qs in plan_dict.get("question_sets", [])],
            risk_controls=plan_dict.get("risk_controls", []),
            contextual_references=[ContextualReference(**ref) for ref in plan_dict.get("contextual_references", [])],
            created_at=created_at,
            quality_notes=plan_dict.get("quality_notes", []),
        )


DATA_DIR = Path(__file__).with_name("data")
knowledge_base = LegalKnowledgeBase(DATA_DIR / "legal_corpus.json")
insight_classifier = LegalInsightClassifier(DATA_DIR / "insight_training.json")
llm_client = ClaudeClient()
case_classifier = CaseOutcomeClassifier(DATA_DIR / "case_outcomes.json", llm_client=llm_client)
witness_generator = WitnessStrategyGenerator(DATA_DIR / "witness_templates.json", llm_client=llm_client)

identity_service = IdentityService()
authorization_service = AuthorizationService(identity_service)
operations_service = OperationsService()
llm_reliability_service = LLMReliabilityService()
ml_pipeline_service = MLPipelineService()
audit_service = AuditService()
document_service = DocumentService(
    insight_classifier,
    DATA_DIR,
    llm_client=llm_client,
    audit_service=audit_service,
    ml_pipeline_service=ml_pipeline_service,
    reliability_service=llm_reliability_service,
    operations_service=operations_service,
)
privacy_service = PrivacyService(document_service, audit_service)
workflow_service = WorkflowService(audit_service)
rag_engine = LegalRagEngine(knowledge_base, llm_client=llm_client)
conversation_service = ConversationService(
    document_service,
    rag_engine,
    ml_pipeline_service=ml_pipeline_service,
    reliability_service=llm_reliability_service,
    operations_service=operations_service,
    authorization_service=authorization_service,
)
prediction_service = PredictionService(
    case_classifier,
    audit_service,
    ml_pipeline_service=ml_pipeline_service,
    operations_service=operations_service,
    reliability_service=llm_reliability_service,
)
witness_service = WitnessService(
    document_service,
    rag_engine,
    witness_generator,
    audit_service,
    ml_pipeline_service=ml_pipeline_service,
    operations_service=operations_service,
    reliability_service=llm_reliability_service,
)


__all__ = [
    "AuthorizationError",
    "identity_service",
    "authorization_service",
    "operations_service",
    "ml_pipeline_service",
    "llm_reliability_service",
    "audit_service",
    "conversation_service",
    "document_service",
    "privacy_service",
    "prediction_service",
    "workflow_service",
    "witness_service",
]

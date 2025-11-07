from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TenantCreate(BaseModel):
    id: str = Field(..., description="Stable tenant identifier", pattern=r"^[a-zA-Z0-9_-]+$")
    name: str
    contact_email: str | None = None
    metadata: dict | None = None


class Tenant(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    contact_email: str | None = None
    metadata: dict | None = Field(default_factory=dict)


class UserCreate(BaseModel):
    tenant_id: str
    user_id: str
    email: str
    password: str
    roles: List[str] = Field(default_factory=lambda: ["analyst"])


class UserUpdate(BaseModel):
    tenant_id: str
    user_id: str
    email: Optional[str] = None
    roles: Optional[List[str]] = None
    status: Optional[str] = None


class UserSummary(BaseModel):
    tenant_id: str
    user_id: str
    email: str
    roles: List[str]
    status: str
    created_at: datetime
    updated_at: datetime


class TokenCreate(BaseModel):
    tenant_id: str
    user_id: str
    scopes: List[str] = Field(default_factory=lambda: ["analyst"])
    expires_in: int | None = Field(default=None, ge=60, description="Requested TTL in seconds")


class TokenResponse(BaseModel):
    token: str
    token_id: str
    expires_at: datetime | None = None


class LoginRequest(BaseModel):
    tenant_id: str
    email: str
    password: str


class LoginResponse(TokenResponse):
    roles: List[str]


class BootstrapRequest(BaseModel):
    bootstrap_secret: str | None = None
    tenant: TenantCreate
    admin: UserCreate


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(..., description="ISO timestamp when the message was stored")


class ContextualReference(BaseModel):
    document_id: int
    filename: str
    snippet: str
    score: float = Field(..., description="Similarity score between 0 and 1")
    explanation: str | None = Field(
        default=None, description="Short rationale describing why the snippet was selected"
    )


class ChatRequest(BaseModel):
    tenant_id: str
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    history: List[ChatMessage]
    contextual_references: List[ContextualReference]


class DocumentUploadRequest(BaseModel):
    tenant_id: str
    user_id: str
    filename: str
    content: str = Field(..., description="Base64 encoded document content")
    retention_policy: str | None = Field(
        default=None,
        description="Organizational retention label such as 'standard', 'litigation_hold', etc.",
    )
    sensitivity: str | None = Field(
        default=None,
        description="Sensitivity classification for privacy governance",
    )
    change_note: str | None = Field(
        default=None,
        description="Optional audit note describing why the document was uploaded or replaced",
    )


class DocumentVersion(BaseModel):
    version: int
    created_at: datetime
    checksum: str
    created_by: str | None = None
    change_note: str | None = None


class DocumentVersionCreate(BaseModel):
    tenant_id: str
    user_id: str
    content: str = Field(..., description="Base64 encoded document content for the new version")
    change_note: str | None = Field(default=None, description="Explanation for the revision")


class DocumentMetadataUpdate(BaseModel):
    tenant_id: str
    user_id: str
    retention_policy: str | None = None
    sensitivity: str | None = None


class DocumentSummary(BaseModel):
    id: int
    filename: str
    size: int = Field(..., description="Size of the stored document in characters")
    uploaded_at: datetime
    latest_version: int = Field(..., ge=1)
    retention_policy: str
    sensitivity: str
    preview: str
    owner_id: str | None = Field(default=None, description="User that uploaded the document")


class DocumentInsights(BaseModel):
    key_points: List[str]
    obligations: List[str]
    risks: List[str]
    deadlines: List[str]
    recommended_actions: List[str]
    rationale: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(default="lexiai-insights-v2")


class DocumentDetails(DocumentSummary):
    content: str
    insights: DocumentInsights
    versions: List[DocumentVersion]


class DocumentSearchResult(BaseModel):
    document_id: int
    filename: str
    snippet: str
    score: float


class DocumentDeletionRequest(BaseModel):
    tenant_id: str
    user_id: str
    reason: str | None = None


class PredictionRequest(BaseModel):
    tenant_id: str
    user_id: str
    case_details: str


class PredictionResponse(BaseModel):
    probability: float
    rationale: str
    recommended_actions: List[str]
    created_at: datetime
    signals: List["PredictionSignal"] = Field(default_factory=list)
    quality_warnings: List[str] = Field(default_factory=list)


class StoredPrediction(PredictionResponse):
    id: int
    user_id: str
    case_details: str


class PredictionSignal(BaseModel):
    label: str
    weight: float
    direction: str = Field(..., pattern="^(positive|negative)$")
    evidence: str | None = None


class WitnessRequest(BaseModel):
    tenant_id: str
    user_id: str
    witness_name: str
    witness_role: str
    case_summary: str
    objectives: List[str] = Field(default_factory=list)


class WitnessQuestionSet(BaseModel):
    stage: str
    questions: List[str]


class WitnessPlan(BaseModel):
    witness_name: str
    witness_role: str
    strategy: str
    focus_areas: List[str]
    question_sets: List[WitnessQuestionSet]
    risk_controls: List[str]
    contextual_references: List[ContextualReference]
    created_at: datetime
    quality_notes: List[str] = Field(default_factory=list)


class StoredWitnessPlan(WitnessPlan):
    id: int
    user_id: str
    case_summary: str


class PrivacyRequestCreate(BaseModel):
    tenant_id: str
    user_id: str
    request_type: str = Field(..., pattern="^(access|export|erasure)$")
    resource_type: str = Field(..., description="e.g. document, prediction, witness_plan")
    resource_id: str | None = None
    reason: str | None = Field(default=None, description="User-provided context")


class PrivacyRequest(BaseModel):
    id: int
    user_id: str
    request_type: str
    resource_type: str
    resource_id: str | None
    reason: str | None
    status: str
    requested_at: datetime
    resolved_at: datetime | None = None
    resolution_note: str | None = None


class PrivacyRequestUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|in_review|completed|rejected)$")
    resolution_note: str | None = None


class AuditLogEntry(BaseModel):
    id: int
    user_id: str | None = None
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    metadata: dict | None = None
    created_at: datetime


class WorkflowTaskCreate(BaseModel):
    tenant_id: str
    case_id: str
    title: str
    assignee: str | None = None
    due_date: datetime | None = None
    tags: List[str] = Field(default_factory=list)


class WorkflowTaskUpdate(BaseModel):
    tenant_id: str
    title: str | None = None
    status: str | None = Field(default=None, pattern="^(backlog|in_progress|blocked|done)$")
    assignee: str | None = None
    due_date: datetime | None = None
    tags: List[str] | None = None


class WorkflowTask(BaseModel):
    id: int
    case_id: str
    title: str
    status: str
    assignee: str | None
    due_date: datetime | None
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    id: int
    tenant_id: str
    name: str
    source: str
    snapshot: dict
    created_at: datetime
    created_by: str


class ModelMetric(BaseModel):
    run_id: int
    metric: str
    value: float
    recorded_at: datetime


class ModelRunRecord(BaseModel):
    id: int
    tenant_id: str
    dataset_id: int
    run_type: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    notes: str | None = None
    metrics: List[ModelMetric] = Field(default_factory=list)


class LlmFailureRecord(BaseModel):
    id: int
    tenant_id: str
    user_id: str | None
    operation: str
    error: str
    payload: dict | None = None
    occurred_at: datetime
    retried: bool


class OpsEventRecord(BaseModel):
    id: int
    tenant_id: str
    event_type: str
    severity: str
    message: str
    metadata: dict | None = None
    created_at: datetime


class BackupRecord(BaseModel):
    id: int
    tenant_id: str
    backup_type: str
    location: str
    checksum: str | None = None
    triggered_by: str
    created_at: datetime


class BackupCreate(BaseModel):
    tenant_id: str
    backup_type: str = "full"
    triggered_by: str
    target_directory: str | None = None

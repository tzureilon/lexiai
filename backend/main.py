from base64 import b64decode
import binascii
import os

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    BackupRecord,
    BackupCreate,
    AuditLogEntry,
    BootstrapRequest,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DatasetRecord,
    DocumentDetails,
    DocumentMetadataUpdate,
    DocumentSearchResult,
    DocumentSummary,
    DocumentUploadRequest,
    DocumentVersion,
    DocumentVersionCreate,
    LoginRequest,
    LoginResponse,
    LlmFailureRecord,
    PredictionRequest,
    PredictionResponse,
    PrivacyRequest,
    PrivacyRequestCreate,
    PrivacyRequestUpdate,
    TokenCreate,
    TokenResponse,
    StoredPrediction,
    StoredWitnessPlan,
    Tenant,
    TenantCreate,
    UserCreate,
    UserSummary,
    UserUpdate,
    OpsEventRecord,
    ModelRunRecord,
    WitnessRequest,
    WorkflowTask,
    WorkflowTaskCreate,
    WorkflowTaskUpdate,
)
from .services import (
    AuthorizationError,
    audit_service,
    authorization_service,
    conversation_service,
    document_service,
    identity_service,
    llm_reliability_service,
    ml_pipeline_service,
    operations_service,
    prediction_service,
    privacy_service,
    witness_service,
    workflow_service,
)
from .security import Principal, get_current_principal

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("LEXIAI_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
BOOTSTRAP_SECRET = os.getenv("LEXIAI_BOOTSTRAP_SECRET")

app = FastAPI(title="LexiAI Backend", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


def _require_tenant(principal: Principal, tenant_id: str) -> None:
    try:
        authorization_service.require_tenant(principal, tenant_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _require_user(principal: Principal, user_id: str) -> None:
    try:
        authorization_service.require_user(principal, user_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _require_roles(principal: Principal, *roles: str) -> None:
    try:
        authorization_service.require_roles(principal, *roles)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/bootstrap", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
async def bootstrap(payload: BootstrapRequest) -> LoginResponse:
    if BOOTSTRAP_SECRET and payload.bootstrap_secret != BOOTSTRAP_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid bootstrap secret")
    if payload.admin.tenant_id != payload.tenant.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tenant mismatch in bootstrap payload")
    try:
        identity_service.create_tenant(payload.tenant)
    except ValueError:
        # Tenant already exists; continue to bootstrap user/token
        pass
    try:
        identity_service.create_user(payload.admin)
    except ValueError:
        pass
    token = identity_service.issue_token(
        TokenCreate(
            tenant_id=payload.admin.tenant_id,
            user_id=payload.admin.user_id,
            scopes=payload.admin.roles,
        )
    )
    return LoginResponse(token=token.token, token_id=token.token_id, expires_at=token.expires_at, roles=payload.admin.roles)


@app.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    try:
        return identity_service.authenticate(payload)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


@app.post("/auth/api-token", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def create_api_token(
    payload: TokenCreate,
    principal: Principal = Depends(get_current_principal),
) -> TokenResponse:
    _require_tenant(principal, payload.tenant_id)
    if payload.user_id != principal.user_id:
        _require_roles(principal, "tenant:admin", "platform:admin")
    return identity_service.issue_token(payload)


@app.post("/tenants", response_model=Tenant, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    payload: TenantCreate,
    principal: Principal = Depends(get_current_principal),
) -> Tenant:
    _require_roles(principal, "platform:admin")
    try:
        return identity_service.create_tenant(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/tenants", response_model=list[Tenant])
async def list_tenants(principal: Principal = Depends(get_current_principal)) -> list[Tenant]:
    _require_roles(principal, "platform:admin")
    return identity_service.list_tenants()


@app.post("/tenants/{tenant_id}/users", response_model=UserSummary, status_code=status.HTTP_201_CREATED)
async def create_user(
    tenant_id: str,
    payload: UserCreate,
    principal: Principal = Depends(get_current_principal),
) -> UserSummary:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin")
    data = payload.model_dump()
    data["tenant_id"] = tenant_id
    try:
        return identity_service.create_user(UserCreate(**data))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/tenants/{tenant_id}/users", response_model=list[UserSummary])
async def list_users(
    tenant_id: str,
    principal: Principal = Depends(get_current_principal),
) -> list[UserSummary]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin")
    return identity_service.list_users(tenant_id)


@app.patch("/tenants/{tenant_id}/users/{user_id}", response_model=UserSummary)
async def update_user(
    tenant_id: str,
    user_id: str,
    payload: UserUpdate,
    principal: Principal = Depends(get_current_principal),
) -> UserSummary:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin")
    data = payload.model_dump(exclude_unset=True)
    data["tenant_id"] = tenant_id
    data["user_id"] = user_id
    try:
        return identity_service.update_user(UserUpdate(**data))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, principal: Principal = Depends(get_current_principal)) -> ChatResponse:
    _require_tenant(principal, req.tenant_id)
    _require_user(principal, req.user_id)
    conversation_service.add_message(req.tenant_id, req.user_id, "user", req.message)
    history = conversation_service.get_history(req.tenant_id, req.user_id, limit=25)
    response_text, contextual_references = conversation_service.generate_response(
        principal, req.tenant_id, req.message, history
    )
    conversation_service.add_message(req.tenant_id, req.user_id, "assistant", response_text)
    updated_history = conversation_service.get_history(req.tenant_id, req.user_id, limit=25)
    return ChatResponse(response=response_text, history=updated_history, contextual_references=contextual_references)


@app.get("/chat/history/{user_id}", response_model=list[ChatMessage])
async def chat_history(
    user_id: str,
    tenant_id: str = Query(...),
    limit: int = 25,
    principal: Principal = Depends(get_current_principal),
):
    _require_tenant(principal, tenant_id)
    _require_user(principal, user_id)
    history = conversation_service.get_history(tenant_id, user_id, limit=limit)
    return history


@app.post("/upload", response_model=DocumentSummary)
async def upload_document(
    req: DocumentUploadRequest,
    principal: Principal = Depends(get_current_principal),
) -> DocumentSummary:
    _require_tenant(principal, req.tenant_id)
    _require_user(principal, req.user_id)
    try:
        content = b64decode(req.content)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid document encoding") from exc
    summary = document_service.save_document(
        req.filename,
        content,
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        retention_policy=req.retention_policy,
        sensitivity=req.sensitivity,
        change_note=req.change_note,
    )
    operations_service.record_event(
        tenant_id=req.tenant_id,
        event_type="document.upload",
        severity="info",
        message="הועלה מסמך חדש",
        metadata={"document_id": summary.id, "owner_id": req.user_id},
    )
    return summary


@app.get("/documents/search", response_model=list[DocumentSearchResult])
async def search_documents(
    tenant_id: str = Query(...),
    query: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=10),
    principal: Principal = Depends(get_current_principal),
):
    _require_tenant(principal, tenant_id)
    results = document_service.search(tenant_id, query, limit=limit)
    filtered: list[DocumentSearchResult] = []
    for result in results:
        try:
            authorization_service.ensure_document_access(principal, tenant_id, result.document_id)
        except AuthorizationError:
            continue
        filtered.append(result)
    return filtered


@app.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    tenant_id: str = Query(...),
    user_id: str | None = Query(default=None),
    principal: Principal = Depends(get_current_principal),
):
    _require_tenant(principal, tenant_id)
    effective_user = user_id
    if user_id and user_id != principal.user_id:
        _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    elif user_id is None and not principal.has_role("tenant:admin", "platform:admin", "compliance"):
        effective_user = principal.user_id
    return document_service.list_documents(tenant_id, user_id=effective_user)


@app.get("/documents/{doc_id}", response_model=DocumentDetails)
async def get_document(
    doc_id: int,
    tenant_id: str = Query(...),
    requester_id: str | None = Query(default=None),
    principal: Principal = Depends(get_current_principal),
):
    _require_tenant(principal, tenant_id)
    try:
        authorization_service.ensure_document_access(principal, tenant_id, doc_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    try:
        return document_service.get_document(tenant_id, doc_id, requester_id=requester_id or principal.user_id)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/documents/{doc_id}/versions", response_model=DocumentDetails)
async def create_document_version(
    doc_id: int,
    req: DocumentVersionCreate,
    principal: Principal = Depends(get_current_principal),
) -> DocumentDetails:
    _require_tenant(principal, req.tenant_id)
    _require_user(principal, req.user_id)
    try:
        authorization_service.ensure_document_access(principal, req.tenant_id, doc_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    try:
        content = b64decode(req.content)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid document encoding") from exc
    try:
        return document_service.create_version(
            doc_id,
            content,
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            change_note=req.change_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/documents/{doc_id}/versions", response_model=list[DocumentVersion])
async def list_document_versions(
    doc_id: int,
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> list[DocumentVersion]:
    _require_tenant(principal, tenant_id)
    try:
        authorization_service.ensure_document_access(principal, tenant_id, doc_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    try:
        return document_service.list_versions(tenant_id, doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/documents/{doc_id}", response_model=DocumentDetails)
async def update_document_metadata(
    doc_id: int,
    update: DocumentMetadataUpdate,
    principal: Principal = Depends(get_current_principal),
) -> DocumentDetails:
    _require_tenant(principal, update.tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    try:
        authorization_service.ensure_document_access(principal, update.tenant_id, doc_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    try:
        return document_service.update_metadata(
            doc_id,
            tenant_id=update.tenant_id,
            user_id=update.user_id,
            retention_policy=update.retention_policy,
            sensitivity=update.sensitivity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest,
    principal: Principal = Depends(get_current_principal),
) -> PredictionResponse:
    _require_tenant(principal, req.tenant_id)
    _require_user(principal, req.user_id)
    return prediction_service.create_prediction(req.tenant_id, req.user_id, req.case_details)


@app.get("/predictions/{user_id}", response_model=list[StoredPrediction])
async def list_predictions(
    user_id: str,
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
):
    _require_tenant(principal, tenant_id)
    if user_id != principal.user_id:
        _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return prediction_service.list_predictions(tenant_id, user_id)


@app.post("/witness", response_model=StoredWitnessPlan)
async def create_witness_plan(
    req: WitnessRequest,
    principal: Principal = Depends(get_current_principal),
) -> StoredWitnessPlan:
    _require_tenant(principal, req.tenant_id)
    _require_user(principal, req.user_id)
    plan = witness_service.create_plan(
        req.tenant_id,
        req.user_id,
        req.witness_name,
        req.witness_role,
        req.case_summary,
        req.objectives,
    )
    return plan


@app.get("/witness/{user_id}", response_model=list[StoredWitnessPlan])
async def list_witness_plans(
    user_id: str,
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
):
    _require_tenant(principal, tenant_id)
    if user_id != principal.user_id:
        _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return witness_service.list_plans(tenant_id, user_id)


@app.post("/privacy/requests", response_model=PrivacyRequest)
async def create_privacy_request(
    req: PrivacyRequestCreate,
    principal: Principal = Depends(get_current_principal),
) -> PrivacyRequest:
    _require_tenant(principal, req.tenant_id)
    _require_user(principal, req.user_id)
    return privacy_service.create_request(req)


@app.get("/privacy/requests", response_model=list[PrivacyRequest])
async def list_privacy_requests(
    tenant_id: str = Query(...),
    status: str | None = Query(default=None),
    principal: Principal = Depends(get_current_principal),
) -> list[PrivacyRequest]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return privacy_service.list_requests(tenant_id, status=status)


@app.patch("/privacy/requests/{request_id}", response_model=PrivacyRequest)
async def update_privacy_request(
    request_id: int,
    update: PrivacyRequestUpdate,
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> PrivacyRequest:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    try:
        return privacy_service.update_request(tenant_id, request_id, update)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/audit", response_model=list[AuditLogEntry])
async def get_audit_log(
    tenant_id: str = Query(...),
    limit: int = Query(100, ge=1, le=500),
    user_id: str | None = None,
    action: str | None = None,
    principal: Principal = Depends(get_current_principal),
) -> list[AuditLogEntry]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return audit_service.list_entries(tenant_id=tenant_id, limit=limit, user_id=user_id, action=action)


@app.post("/workflows/tasks", response_model=WorkflowTask)
async def create_workflow_task(
    payload: WorkflowTaskCreate,
    principal: Principal = Depends(get_current_principal),
) -> WorkflowTask:
    _require_tenant(principal, payload.tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return workflow_service.create_task(payload)


@app.get("/workflows/tasks", response_model=list[WorkflowTask])
async def list_workflow_tasks(
    tenant_id: str = Query(...),
    case_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    principal: Principal = Depends(get_current_principal),
) -> list[WorkflowTask]:
    _require_tenant(principal, tenant_id)
    tasks = workflow_service.list_tasks(tenant_id, case_id=case_id, status=status)
    if not principal.has_role("tenant:admin", "platform:admin", "compliance"):
        tasks = [task for task in tasks if task.assignee in {None, principal.user_id}]
    return tasks


@app.patch("/workflows/tasks/{task_id}", response_model=WorkflowTask)
async def update_workflow_task(
    task_id: int,
    payload: WorkflowTaskUpdate,
    principal: Principal = Depends(get_current_principal),
) -> WorkflowTask:
    _require_tenant(principal, payload.tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    try:
        return workflow_service.update_task(payload.tenant_id, task_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ops/backups", response_model=BackupRecord, status_code=status.HTTP_201_CREATED)
async def perform_backup(
    payload: BackupCreate,
    principal: Principal = Depends(get_current_principal),
) -> BackupRecord:
    _require_tenant(principal, payload.tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin")
    try:
        triggered_by = payload.triggered_by or principal.user_id
        return operations_service.perform_backup(
            payload.tenant_id,
            backup_type=payload.backup_type,
            triggered_by=triggered_by,
            target_directory=payload.target_directory,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.get("/ops/backups", response_model=list[BackupRecord])
async def list_backups(
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> list[BackupRecord]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return operations_service.list_backups(tenant_id)


@app.get("/ops/events", response_model=list[OpsEventRecord])
async def list_events(
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> list[OpsEventRecord]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return operations_service.list_events(tenant_id)


@app.get("/ml/datasets", response_model=list[DatasetRecord])
async def list_datasets(
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> list[DatasetRecord]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return ml_pipeline_service.list_datasets(tenant_id)


@app.get("/ml/runs", response_model=list[ModelRunRecord])
async def list_model_runs(
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> list[ModelRunRecord]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return ml_pipeline_service.list_runs(tenant_id)


@app.get("/llm/failures", response_model=list[LlmFailureRecord])
async def list_llm_failures(
    tenant_id: str = Query(...),
    principal: Principal = Depends(get_current_principal),
) -> list[LlmFailureRecord]:
    _require_tenant(principal, tenant_id)
    _require_roles(principal, "tenant:admin", "platform:admin", "compliance")
    return llm_reliability_service.list_failures(tenant_id)

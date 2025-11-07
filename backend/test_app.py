import base64
import os
from pathlib import Path

os.environ.setdefault("LEXIAI_SECRET", "unit-test-secret-lexiai-1234567890abcdef")

from fastapi.testclient import TestClient

from .db import db
from .main import app
from .schemas import TenantCreate, TokenCreate, UserCreate
from .services import document_service, identity_service


TENANT_ID = "demo-tenant"
USER_ID = "tester"

client = TestClient(app)


def setup_function(_: object) -> None:
    db.reset()
    identity_service.create_tenant(TenantCreate(id=TENANT_ID, name="Test Tenant"))
    identity_service.create_user(
        UserCreate(
            tenant_id=TENANT_ID,
            user_id=USER_ID,
            email="tester@example.com",
            password="Secret123!",
            roles=["tenant:admin", "compliance"],
        )
    )
    token = identity_service.issue_token(
        TokenCreate(tenant_id=TENANT_ID, user_id=USER_ID, scopes=["tenant:admin", "compliance"])
    )
    client.headers.update({"Authorization": f"Bearer {token.token}"})
    document_service.refresh_index(TENANT_ID)


def _upload_sample_document() -> int:
    content = """
    הסכם שכירות בין אלכס לרות. השוכר מתחייב להעביר את התשלום החודשי עד ליום 10 בכל חודש.
    במקרה של איחור בתשלום ייגבה קנס חודשי. הצדדים מאשרים שהמסמך נחתם ונמסר כדין.
    """.strip()
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    response = client.post(
        "/upload",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "filename": "rental-agreement.txt",
            "content": encoded,
            "retention_policy": "standard",
            "sensitivity": "confidential",
            "change_note": "מסמך בסיסי",
        },
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_chat_flow_with_contextual_references():
    _upload_sample_document()
    response = client.post(
        "/chat",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "message": "יש לי חוזה שכירות והמשכיר מאחר בתשלום",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "response" in payload
    assert len(payload["history"]) == 2
    assert payload["contextual_references"]
    assert payload["contextual_references"][0]["explanation"]
    history_response = client.get(
        "/chat/history/tester",
        params={"tenant_id": TENANT_ID},
    )
    assert history_response.status_code == 200
    assert len(history_response.json()) == 2


def test_document_upload_listing_and_insights():
    doc_id = _upload_sample_document()
    listing = client.get("/documents", params={"tenant_id": TENANT_ID})
    assert listing.status_code == 200
    documents = listing.json()
    assert len(documents) == 1
    assert documents[0]["id"] == doc_id
    assert documents[0]["retention_policy"] == "standard"
    assert documents[0]["sensitivity"] == "confidential"
    assert documents[0]["latest_version"] == 1

    details = client.get(
        f"/documents/{doc_id}",
        params={"tenant_id": TENANT_ID, "requester_id": USER_ID},
    )
    assert details.status_code == 200
    payload = details.json()
    assert payload["owner_id"] == "tester"
    assert payload["insights"]["key_points"]
    assert payload["insights"]["obligations"]
    assert 0.0 <= payload["insights"]["confidence_score"] <= 1.0
    assert payload["insights"]["rationale"]
    assert payload["versions"]
    search = client.get(
        "/documents/search",
        params={"tenant_id": TENANT_ID, "query": "קנס"},
    )
    assert search.status_code == 200
    results = search.json()
    assert results
    assert any(result["document_id"] == doc_id for result in results)


def test_backup_endpoint_creates_backup(tmp_path):
    response = client.post(
        "/ops/backups",
        json={
            "tenant_id": TENANT_ID,
            "backup_type": "full",
            "triggered_by": USER_ID,
            "target_directory": str(tmp_path),
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["tenant_id"] == TENANT_ID
    assert Path(payload["location"]).exists()
    assert payload["checksum"]

    events = client.get("/ops/events", params={"tenant_id": TENANT_ID})
    assert events.status_code == 200
    assert any(event["event_type"] == "backup.completed" for event in events.json())

def test_prediction_flow_with_actions():
    _upload_sample_document()
    response = client.post(
        "/predict",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "case_details": "הצגתי הסכם חתום ותיעוד תשלומים, אך קיים עיכוב בהעברת הפיצוי לבית המשפט",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["recommended_actions"]
    assert payload["rationale"]
    assert payload["signals"]
    assert isinstance(payload["quality_warnings"], list)
    history = client.get(
        "/predictions/tester",
        params={"tenant_id": TENANT_ID},
    )
    assert history.status_code == 200
    assert len(history.json()) == 1


def test_witness_plan_generation():
    _upload_sample_document()
    response = client.post(
        "/witness",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "witness_name": "רות כהן",
            "witness_role": "עד מומחה",
            "case_summary": "המחלוקת נסובה סביב נזקי איחור במסירת דירה",
            "objectives": ["לאתגר את חוות הדעת"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["witness_name"] == "רות כהן"
    assert payload["question_sets"]
    assert payload["contextual_references"]
    history = client.get(
        "/witness/tester",
        params={"tenant_id": TENANT_ID},
    )
    assert history.status_code == 200
    history_payload = history.json()
    assert len(history_payload) == 1


def test_document_versioning_and_metadata_controls():
    doc_id = _upload_sample_document()
    update_response = client.patch(
        f"/documents/{doc_id}",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "retention_policy": "litigation_hold",
            "sensitivity": "restricted",
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["retention_policy"] == "litigation_hold"
    assert updated["sensitivity"] == "restricted"

    encoded = base64.b64encode("עדכון התחייבויות".encode("utf-8")).decode("ascii")
    version_response = client.post(
        f"/documents/{doc_id}/versions",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "content": encoded,
            "change_note": "עדכון גרסה",
        },
    )
    assert version_response.status_code == 200
    version_payload = version_response.json()
    assert version_payload["latest_version"] == 2
    versions_list = client.get(
        f"/documents/{doc_id}/versions",
        params={"tenant_id": TENANT_ID},
    )
    assert versions_list.status_code == 200
    assert len(versions_list.json()) >= 2


def test_privacy_request_erasure_flow():
    doc_id = _upload_sample_document()
    response = client.post(
        "/privacy/requests",
        json={
            "tenant_id": TENANT_ID,
            "user_id": USER_ID,
            "request_type": "erasure",
            "resource_type": "document",
            "resource_id": str(doc_id),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"completed", "rejected"}
    documents = client.get("/documents", params={"tenant_id": TENANT_ID})
    assert documents.status_code == 200
    if payload["status"] == "completed":
        assert documents.json() == []


def test_workflow_and_audit_logging():
    _upload_sample_document()
    task_resp = client.post(
        "/workflows/tasks",
        json={
            "tenant_id": TENANT_ID,
            "case_id": "A-123",
            "title": "בדיקת חומרי ראיות",
            "assignee": "qa-team",
            "due_date": "2024-12-31",
            "tags": ["compliance", "evidence"],
        },
    )
    assert task_resp.status_code == 200
    task_id = task_resp.json()["id"]
    update_resp = client.patch(
        f"/workflows/tasks/{task_id}",
        json={"tenant_id": TENANT_ID, "status": "done"},
    )
    assert update_resp.status_code == 200
    tasks = client.get(
        "/workflows/tasks",
        params={"tenant_id": TENANT_ID},
    )
    assert tasks.status_code == 200
    task_entries = tasks.json()
    assert any(task["status"] == "done" for task in task_entries)

    audit_entries = client.get(
        "/audit",
        params={"tenant_id": TENANT_ID, "action": "workflow.task_created"},
    )
    assert audit_entries.status_code == 200
    assert any(entry["resource_id"] == str(task_id) for entry in audit_entries.json())

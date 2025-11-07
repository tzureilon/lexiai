from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from itertools import cycle
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent


def _sanitize_environment(value: str) -> str:
    cleaned = value.strip().lower()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in cleaned) or "development"


ENVIRONMENT = _sanitize_environment(os.getenv("LEXIAI_ENVIRONMENT", "development"))
DEFAULT_SQLITE_PATH = BASE_DIR / f"lexiai_{ENVIRONMENT}.db"
DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")
REPLICA_URLS = [
    url.strip()
    for url in os.getenv("DATABASE_REPLICA_URLS", "").split(",")
    if url.strip()
]
DEFAULT_BACKUP_DIR = Path(os.getenv("LEXIAI_BACKUP_DIR", BASE_DIR / "backups"))
PG_DUMP_PATH = os.getenv("PG_DUMP_PATH", "pg_dump")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

SCHEMA_TABLES = [
    "tenants",
    "users",
    "api_tokens",
    "conversations",
    "documents",
    "document_versions",
    "privacy_requests",
    "audit_logs",
    "workflow_tasks",
    "predictions",
    "witness_plans",
    "model_datasets",
    "model_runs",
    "model_run_metrics",
    "llm_failures",
    "ops_events",
    "data_backups",
]

TENANT_SCOPED_TABLES = [
    "users",
    "api_tokens",
    "conversations",
    "documents",
    "document_versions",
    "privacy_requests",
    "audit_logs",
    "workflow_tasks",
    "predictions",
    "witness_plans",
    "model_datasets",
    "model_runs",
    "model_run_metrics",
    "llm_failures",
    "ops_events",
    "data_backups",
]


class SQLiteDriver:
    dialect = "sqlite"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        # SQLite connections are short-lived per call; nothing to dispose.
        return None


class PostgresDriver:
    dialect = "postgresql"

    def __init__(self, dsn: str, pool_size: int = 5) -> None:
        self._dsn = dsn
        self._pool: Queue = Queue(maxsize=pool_size)
        self._creator = _postgres_connection_factory(dsn)

    @contextmanager
    def connection(self):
        try:
            conn = self._pool.get_nowait()
        except Empty:
            conn = self._creator()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                self._pool.put_nowait(conn)
            except Exception:
                conn.close()

    def close(self) -> None:
        while True:
            try:
                conn = self._pool.get_nowait()
            except Empty:
                break
            else:
                conn.close()


def _postgres_connection_factory(dsn: str):
    try:
        import psycopg  # type: ignore
    except ModuleNotFoundError:
        psycopg = None

    if psycopg is not None:
        def _creator():
            return psycopg.connect(dsn, autocommit=False)

        return _creator

    try:
        import psycopg2  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(
            "PostgreSQL URL supplied but no psycopg/psycopg2 driver is installed. "
            "Install the driver in your environment before starting the API service."
        ) from exc

    def _creator():
        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        return conn

    return _creator


class MigrationManager:
    def __init__(self, database: "Database") -> None:
        self._database = database

    def apply_all(self) -> None:
        with self._database._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            existing = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for version, operation in MIGRATIONS:
                if version in existing:
                    continue
                operation(connection, self._database.dialect)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, _utc_now()),
                )


def _create_base_schema(connection, dialect: str) -> None:
    id_column = (
        "INTEGER PRIMARY KEY AUTOINCREMENT"
        if dialect == "sqlite"
        else "INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY"
    )

    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS conversations (
            id {id_column},
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id {id_column},
            tenant_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            user_id TEXT NOT NULL,
            latest_version INTEGER NOT NULL DEFAULT 1,
            retention_policy TEXT NOT NULL DEFAULT 'standard',
            sensitivity TEXT NOT NULL DEFAULT 'internal'
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS document_versions (
            id {id_column},
            tenant_id TEXT NOT NULL,
            document_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            checksum TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT,
            change_note TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(document_id, version)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS privacy_requests (
            id {id_column},
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            request_type TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            reason TEXT,
            status TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            resolved_at TEXT,
            resolution_note TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id {id_column},
            tenant_id TEXT NOT NULL,
            user_id TEXT,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS workflow_tasks (
            id {id_column},
            tenant_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            assignee TEXT,
            due_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]'
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id {id_column},
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            case_details TEXT NOT NULL,
            probability TEXT NOT NULL,
            rationale TEXT NOT NULL,
            recommended_actions TEXT NOT NULL,
            signals TEXT NOT NULL,
            quality_warnings TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS witness_plans (
            id {id_column},
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            witness_name TEXT NOT NULL,
            witness_role TEXT NOT NULL,
            case_summary TEXT NOT NULL,
            plan TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
    ]

    indexes = [
        "CREATE INDEX IF NOT EXISTS ix_conversations_tenant_user ON conversations (tenant_id, user_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_documents_tenant_uploaded ON documents (tenant_id, uploaded_at)",
        "CREATE INDEX IF NOT EXISTS ix_document_versions_tenant_doc ON document_versions (tenant_id, document_id, version)",
        "CREATE INDEX IF NOT EXISTS ix_privacy_requests_tenant_status ON privacy_requests (tenant_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_audit_logs_tenant_created ON audit_logs (tenant_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_workflow_tasks_tenant_status ON workflow_tasks (tenant_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_predictions_tenant_user ON predictions (tenant_id, user_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_witness_plans_tenant_user ON witness_plans (tenant_id, user_id, created_at)",
    ]

    for statement in statements:
        connection.execute(statement)
    for statement in indexes:
        connection.execute(statement)


def _create_ops_and_security_tables(connection, dialect: str) -> None:
    id_column = (
        "INTEGER PRIMARY KEY AUTOINCREMENT"
        if dialect == "sqlite"
        else "INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY"
    )
    statements = [
        """
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            contact_email TEXT,
            metadata TEXT DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            roles TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            PRIMARY KEY (tenant_id, id),
            UNIQUE(tenant_id, email)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS api_tokens (
            id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            scopes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            last_used_at TEXT,
            PRIMARY KEY (tenant_id, id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS model_datasets (
            id {id_column},
            tenant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            snapshot TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS model_runs (
            id {id_column},
            tenant_id TEXT NOT NULL,
            dataset_id INTEGER NOT NULL,
            run_type TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            notes TEXT,
            FOREIGN KEY(dataset_id) REFERENCES model_datasets(id) ON DELETE CASCADE
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS model_run_metrics (
            id {id_column},
            tenant_id TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES model_runs(id) ON DELETE CASCADE
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS llm_failures (
            id {id_column},
            tenant_id TEXT NOT NULL,
            user_id TEXT,
            operation TEXT NOT NULL,
            payload TEXT,
            error TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            retried INTEGER NOT NULL DEFAULT 0
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS ops_events (
            id {id_column},
            tenant_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS data_backups (
            id {id_column},
            tenant_id TEXT NOT NULL,
            backup_type TEXT NOT NULL,
            location TEXT NOT NULL,
            checksum TEXT,
            triggered_by TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
    ]
    indexes = [
        "CREATE INDEX IF NOT EXISTS ix_users_tenant_status ON users (tenant_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_api_tokens_usage ON api_tokens (tenant_id, user_id)",
        "CREATE INDEX IF NOT EXISTS ix_api_tokens_hash ON api_tokens (token_hash)",
        "CREATE INDEX IF NOT EXISTS ix_model_runs_tenant_status ON model_runs (tenant_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_model_metrics_run ON model_run_metrics (run_id, metric)",
        "CREATE INDEX IF NOT EXISTS ix_llm_failures_tenant ON llm_failures (tenant_id, occurred_at)",
        "CREATE INDEX IF NOT EXISTS ix_ops_events_type ON ops_events (tenant_id, event_type, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_data_backups_tenant ON data_backups (tenant_id, created_at)",
    ]
    for statement in statements:
        connection.execute(statement)
    for statement in indexes:
        connection.execute(statement)


MIGRATIONS = [
    ("0001_base_schema", _create_base_schema),
    (
        "0002_ops_security",
        lambda connection, dialect: _create_ops_and_security_tables(connection, dialect),
    ),
]


def _normalize_params(params: Iterable | Mapping | None) -> Sequence:
    if params is None:
        return ()
    if isinstance(params, Mapping):
        return params  # rely on driver support for dict parameters
    if isinstance(params, (list, tuple)):
        return tuple(params)
    return (params,)


def _convert_placeholders(query: str, dialect: str) -> str:
    if dialect == "postgresql":
        parts = query.split("?")
        if len(parts) == 1:
            return query
        rebuilt = parts[0]
        for idx, part in enumerate(parts[1:]):
            rebuilt += f"%s{part}"
        return rebuilt
    return query


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class DBResult:
    def __init__(self, cursor) -> None:
        self._cursor = cursor
        self.lastrowid = getattr(cursor, "lastrowid", None)

    def __getattr__(self, item: str):  # pragma: no cover - passthrough
        return getattr(self._cursor, item)


class Database:
    def __init__(
        self,
        url: str | None = None,
        *,
        replica_urls: Sequence[str] | None = None,
    ) -> None:
        self._url = url or DEFAULT_DATABASE_URL
        self._driver = self._create_driver(self._url)
        replica_urls = replica_urls if replica_urls is not None else REPLICA_URLS
        self._replica_drivers = [self._create_driver(replica_url) for replica_url in replica_urls]
        self._replica_cycle = cycle(self._replica_drivers) if self._replica_drivers else None
        self.dialect = self._driver.dialect
        MigrationManager(self).apply_all()

    def _create_driver(self, url: str):
        parsed = urlparse(url)
        scheme = parsed.scheme or "sqlite"
        if scheme in {"sqlite", "file"}:
            path = parsed.path or ""
            if path.startswith("//"):
                path = path[2:]
            if path.startswith("/"):
                resolved = Path(path)
            else:
                resolved = DEFAULT_SQLITE_PATH if not path else Path(path)
            if not resolved.is_absolute():
                resolved = (Path(parsed.netloc) / resolved) if parsed.netloc else (BASE_DIR / resolved)
            return SQLiteDriver(resolved)
        if scheme in {"postgres", "postgresql"}:
            dsn = url
            if parsed.username and parsed.password:
                # psycopg expects percent-decoded credentials
                dsn = url.replace("postgres://", "postgresql://")
            return PostgresDriver(dsn)
        raise ValueError(f"Unsupported database scheme: {scheme}")

    @contextmanager
    def _connect(self, *, read_only: bool = False):
        driver = self._driver
        if read_only and self._replica_cycle is not None:
            driver = next(self._replica_cycle)
        with driver.connection() as connection:
            yield connection

    def execute(self, query: str, params: Iterable | Mapping | None = None) -> DBResult:
        sql = _convert_placeholders(query, self.dialect)
        bound = _normalize_params(params)
        with self._connect() as connection:
            cursor = connection.execute(sql, bound)
            return DBResult(cursor)

    def query(
        self,
        query: str,
        params: Iterable | Mapping | None = None,
        *,
        read_only: bool = False,
    ):
        sql = _convert_placeholders(query, self.dialect)
        bound = _normalize_params(params)
        with self._connect(read_only=read_only) as connection:
            cursor = connection.execute(sql, bound)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def reset(self, tenant_id: str | None = None) -> None:
        with self._connect() as connection:
            if tenant_id:
                for table in TENANT_SCOPED_TABLES:
                    connection.execute(
                        _convert_placeholders(
                            f"DELETE FROM {table} WHERE tenant_id = ?",
                            self.dialect,
                        ),
                        _normalize_params((tenant_id,)),
                    )
            else:
                for table in SCHEMA_TABLES:
                    connection.execute(f"DELETE FROM {table}")

    def create_backup(
        self,
        tenant_id: str,
        *,
        backup_type: str = "full",
        target_directory: str | Path | None = None,
    ) -> tuple[str, str]:
        directory = Path(target_directory) if target_directory else DEFAULT_BACKUP_DIR
        destination_dir = _ensure_directory(directory / tenant_id / ENVIRONMENT / backup_type)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if isinstance(self._driver, SQLiteDriver):
            source_path: Path = self._driver._path  # type: ignore[attr-defined]
            if not source_path.exists():
                raise RuntimeError("SQLite database file does not exist; cannot perform backup")
            with tempfile.TemporaryDirectory(prefix="lexiai-backup-") as tmpdir:
                temp_path = Path(tmpdir) / f"sqlite-{timestamp}.db"
                shutil.copy2(source_path, temp_path)
                checksum = _file_checksum(temp_path)
                final_path = destination_dir / temp_path.name
                shutil.move(str(temp_path), final_path)
            return str(final_path), checksum
        if isinstance(self._driver, PostgresDriver):
            final_path = destination_dir / f"postgres-{timestamp}.dump"
            with tempfile.NamedTemporaryFile(prefix="lexiai-backup-", suffix=".dump", delete=False) as tmpfile:
                temp_path = Path(tmpfile.name)
            try:
                command = [
                    PG_DUMP_PATH,
                    f"--dbname={self._url}",
                    "--format=custom",
                    f"--file={temp_path}",
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"pg_dump failed with exit code {completed.returncode}: {completed.stderr.strip()}"
                    )
                checksum = _file_checksum(temp_path)
                shutil.move(str(temp_path), final_path)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "pg_dump executable not found. Install PostgreSQL client tools or set PG_DUMP_PATH."
                ) from exc
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            return str(final_path), checksum
        raise RuntimeError("Unsupported database driver for backups")

    def close(self) -> None:
        self._driver.close()
        for driver in self._replica_drivers:
            driver.close()


_db = Database(DEFAULT_DATABASE_URL)

db = _db

__all__ = ["db", "Database", "DEFAULT_DATABASE_URL"]

"""JSON-backed persistence for the uploaded-document registry and chat sessions.

Kept intentionally simple (flat JSON files) to match the local, single-user
scope of the project. The vector store (Qdrant) remains the source of truth for
document content; these files only track metadata and conversation history.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from app.config import DOCUMENTS_PATH, SESSIONS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Documents registry
# --------------------------------------------------------------------------- #

def _read_documents() -> list:
    if not os.path.exists(DOCUMENTS_PATH):
        return []
    with open(DOCUMENTS_PATH) as f:
        return json.load(f)


def _write_documents(documents: list) -> None:
    os.makedirs(os.path.dirname(DOCUMENTS_PATH), exist_ok=True)
    with open(DOCUMENTS_PATH, "w") as f:
        json.dump(documents, f, indent=2)


def list_documents() -> list:
    """Return all registered documents, newest first."""
    return sorted(
        _read_documents(),
        key=lambda d: d.get("uploaded_at", ""),
        reverse=True,
    )


def add_document(meta: dict) -> dict:
    """Register a newly ingested document. `meta` is the dict from ingest_pdf."""
    entry = dict(meta)
    entry["uploaded_at"] = _now()

    documents = _read_documents()
    documents.append(entry)
    _write_documents(documents)

    return entry


def get_document(document_id: str):
    for doc in _read_documents():
        if doc.get("document_id") == document_id:
            return doc
    return None


def update_document(document_id: str, **fields) -> dict:
    """Merge `fields` into a registered document (e.g. status, counts, error)."""
    documents = _read_documents()
    updated = None
    for doc in documents:
        if doc.get("document_id") == document_id:
            doc.update(fields)
            updated = doc
            break
    _write_documents(documents)
    return updated


# --------------------------------------------------------------------------- #
# Chat sessions
# --------------------------------------------------------------------------- #

def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _read_session(session_id: str):
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_session(session: dict) -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(_session_path(session["id"]), "w") as f:
        json.dump(session, f, indent=2)


def create_session(document_id: str, filename: str, title: str = None) -> dict:
    session = {
        "id": str(uuid.uuid4()),
        "title": title or f"Chat • {filename}",
        "document_id": document_id,
        "filename": filename,
        "created_at": _now(),
        "messages": [],
    }
    _write_session(session)
    return session


def get_session(session_id: str):
    return _read_session(session_id)


def list_sessions() -> list:
    """Return session summaries (without full message bodies), newest first."""
    if not os.path.exists(SESSIONS_DIR):
        return []

    summaries = []
    for name in os.listdir(SESSIONS_DIR):
        if not name.endswith(".json"):
            continue
        session = _read_session(name[: -len(".json")])
        if not session:
            continue
        summaries.append(
            {
                "id": session["id"],
                "title": session["title"],
                "document_id": session["document_id"],
                "filename": session["filename"],
                "created_at": session["created_at"],
                "num_messages": len(session["messages"]),
            }
        )

    return sorted(summaries, key=lambda s: s["created_at"], reverse=True)


def append_message(session_id: str, role: str, content: str):
    """Append a message to a session and return the stored message dict."""
    session = _read_session(session_id)
    if session is None:
        return None

    message = {"role": role, "content": content, "ts": _now()}
    session["messages"].append(message)

    # Use the first user message as a friendlier session title.
    if role == "user" and len([m for m in session["messages"] if m["role"] == "user"]) == 1:
        session["title"] = content[:60] + ("…" if len(content) > 60 else "")

    _write_session(session)
    return message


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

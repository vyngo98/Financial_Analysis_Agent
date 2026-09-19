import os
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import storage
from app.agent.financial_agent import create_financial_agent
from app.config import UPLOAD_DIR
from app.ingestion.pdf_ingestion import ingest_pdf

app = FastAPI(title="Financial RAG Agent")

# The agent is stateless with respect to documents: which document to search is
# passed per request via config, so a single instance is safely reused.
agent = create_financial_agent()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class CreateSessionRequest(BaseModel):
    document_id: str


class MessageRequest(BaseModel):
    content: str


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

@app.get("/api/documents")
def get_documents():
    return storage.list_documents()


def _run_ingest(dest: str, document_id: str):
    """Background job: ingest (may include slow OCR) and update status."""
    try:
        result = ingest_pdf(dest, document_id=document_id)
        storage.update_document(document_id, status="ready", **{
            k: v for k, v in result.items()
            if k not in ("document_id", "filename")
        })
    except Exception as e:  # noqa: BLE001 - surface failure to the UI
        storage.update_document(document_id, status="failed", error=str(e))


@app.post("/api/documents")
def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(file.file.read())

    # Register immediately as "processing"; OCR of scanned reports can take
    # minutes, so the actual ingest runs in the background.
    document_id = str(uuid.uuid4())
    doc = storage.add_document({
        "document_id": document_id,
        "filename": file.filename,
        "status": "processing",
    })

    background_tasks.add_task(_run_ingest, dest, document_id)
    return doc


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

@app.get("/api/sessions")
def get_sessions():
    return storage.list_sessions()


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    document = storage.get_document(req.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail="Document is still processing. Please wait until it is ready.",
        )
    return storage.create_session(document["document_id"], document["filename"])


@app.get("/api/sessions/{session_id}")
def read_session(session_id: str):
    session = storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    if not storage.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": True}


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, req: MessageRequest):
    session = storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    storage.append_message(session_id, "user", req.content)

    # Refresh to include the just-added user message; pass the full history so
    # follow-up questions have conversational context.
    session = storage.get_session(session_id)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in session["messages"]
    ]

    result = agent.invoke(
        {"messages": history},
        config={"configurable": {"document_id": session["document_id"]}},
    )
    answer = result["messages"][-1].content

    return storage.append_message(session_id, "assistant", answer)


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

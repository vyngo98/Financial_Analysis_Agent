const state = {
  documents: [],
  sessions: [],
  currentSession: null, // full session object
};

// --------------------------------------------------------------------------- //
// API helpers
// --------------------------------------------------------------------------- //
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// --------------------------------------------------------------------------- //
// Minimal, safe Markdown rendering (escape first, then a few patterns)
// --------------------------------------------------------------------------- //
function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text);
  const lines = escaped.split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    // Markdown table: a header row followed by a |---| separator row.
    if (line.includes("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const table = [];
      while (i < lines.length && lines[i].includes("|")) {
        table.push(lines[i]);
        i++;
      }
      out.push(renderTable(table));
      continue;
    }
    // Headings: #..###### followed by a space.
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${heading[2]}</h${level}>`);
      i++;
      continue;
    }
    // Unordered list: consecutive "- " / "* " lines.
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${lines[i].replace(/^\s*[-*]\s+/, "")}</li>`);
        i++;
      }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    // Ordered list: consecutive "1. " lines.
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(`<li>${lines[i].replace(/^\s*\d+\.\s+/, "")}</li>`);
        i++;
      }
      out.push(`<ol>${items.join("")}</ol>`);
      continue;
    }
    out.push(line);
    i++;
  }

  let html = out.join("\n");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  return html;
}

function splitRow(row) {
  return row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
}

function renderTable(rows) {
  const header = splitRow(rows[0]);
  const body = rows.slice(2).map(splitRow);
  let html = "<table><thead><tr>";
  html += header.map((h) => `<th>${h}</th>`).join("");
  html += "</tr></thead><tbody>";
  for (const r of body) {
    html += "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";
  return html;
}

// --------------------------------------------------------------------------- //
// Rendering
// --------------------------------------------------------------------------- //
function renderDocuments() {
  const ul = document.getElementById("documents");
  ul.innerHTML = "";
  if (!state.documents.length) {
    ul.innerHTML = `<li class="item-sub" style="padding:6px 10px;">Chưa có tài liệu nào.</li>`;
    return;
  }
  for (const doc of state.documents) {
    const li = document.createElement("li");
    li.className = "list-item";

    const status = doc.status || "ready";
    const isReady = status !== "processing" && status !== "failed";
    let badge, sub;
    if (status === "processing") {
      badge = "⏳";
      sub = "Đang xử lý (OCR có thể mất vài phút)…";
    } else if (status === "failed") {
      badge = "⚠️";
      sub = "Lỗi xử lý: " + escapeHtml(doc.error || "unknown");
    } else {
      badge = "✅";
      const pages = doc.num_pages ?? "?";
      const native = doc.num_text_pages ?? 0;
      const ocr = doc.num_ocr_pages ?? 0;
      const mix = (native || ocr) ? ` (${native} native · ${ocr} OCR)` : "";
      const chunks = doc.num_chunks != null ? ` · ${doc.num_chunks} chunk` : "";
      sub = `${pages} trang${mix} · ${doc.num_tables ?? 0} bảng${chunks}`;
    }

    li.innerHTML = `
      <div class="item-title">${badge} ${escapeHtml(doc.filename)}</div>
      <div class="item-sub">${sub}</div>
    `;

    if (isReady) {
      li.title = "Bấm để tạo phiên hỏi–đáp mới";
      li.onclick = () => startSession(doc.document_id);
    } else {
      li.style.opacity = "0.6";
      li.style.cursor = "default";
    }
    ul.appendChild(li);
  }
}

function renderSessions() {
  const ul = document.getElementById("sessions");
  ul.innerHTML = "";
  if (!state.sessions.length) {
    ul.innerHTML = `<li class="item-sub" style="padding:6px 10px;">Chưa có phiên nào.</li>`;
    return;
  }
  for (const s of state.sessions) {
    const li = document.createElement("li");
    li.className = "list-item";
    if (state.currentSession && s.id === state.currentSession.id) li.classList.add("active");
    li.innerHTML = `
      <div class="session-row">
        <div class="item-main">
          <div class="item-title">${escapeHtml(s.title)}</div>
          <div class="item-sub">${escapeHtml(s.filename)} · ${s.num_messages} tin nhắn</div>
        </div>
        <button class="del-btn" title="Xóa phiên">×</button>
      </div>
    `;
    li.querySelector(".item-main").onclick = () => openSession(s.id);
    li.querySelector(".del-btn").onclick = (e) => {
      e.stopPropagation();
      deleteSession(s.id);
    };
    ul.appendChild(li);
  }
}

function renderMessages() {
  const box = document.getElementById("messages");
  box.innerHTML = "";
  const session = state.currentSession;

  if (!session) {
    box.innerHTML = `<div class="empty-state"><div class="empty-icon">💬</div><p>Upload một PDF rồi chọn nó để tạo phiên hỏi–đáp.</p></div>`;
    return;
  }
  if (!session.messages.length) {
    box.innerHTML = `<div class="empty-state"><div class="empty-icon">✨</div><p>Đặt câu hỏi đầu tiên về <strong>${escapeHtml(session.filename)}</strong>.</p></div>`;
    return;
  }

  for (const m of session.messages) {
    box.appendChild(messageEl(m.role, m.content, m.sources));
  }
  box.scrollTop = box.scrollHeight;
}

function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const chips = sources
    .map((s) => {
      const label = s.content_type === "table" ? "TABLE" : "TEXT";
      const page = s.page != null ? escapeHtml(String(s.page)) : "?";
      return `<span class="source-chip">Trang ${page} · ${label}</span>`;
    })
    .join("");
  return `<div class="sources">${chips}</div>`;
}

function messageEl(role, content, sources) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const avatar = role === "user" ? "🧑" : "🤖";
  const body = role === "assistant" ? renderMarkdown(content) : escapeHtml(content);
  const cite = role === "assistant" ? renderSources(sources) : "";
  div.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${body}${cite}</div>`;
  return div;
}

function updateChatHeader() {
  const title = document.getElementById("chat-title");
  const doc = document.getElementById("chat-doc");
  const input = document.getElementById("composer-input");
  const btn = document.getElementById("send-btn");

  if (state.currentSession) {
    title.textContent = state.currentSession.title;
    doc.textContent = "Tài liệu: " + state.currentSession.filename;
    input.disabled = false;
    btn.disabled = false;
  } else {
    title.textContent = "Chưa chọn phiên";
    doc.textContent = "Chọn một document ở bên trái để bắt đầu hỏi";
    input.disabled = true;
    btn.disabled = true;
  }
}

// --------------------------------------------------------------------------- //
// Actions
// --------------------------------------------------------------------------- //
function listError(ulId, message) {
  const ul = document.getElementById(ulId);
  if (ul) {
    ul.innerHTML = `<li class="item-sub error" style="padding:6px 10px;">${escapeHtml(message)}</li>`;
  }
}

async function loadDocuments() {
  try {
    state.documents = await api("/api/documents");
  } catch (err) {
    listError("documents", "Không tải được danh sách tài liệu: " + err.message);
    return;
  }
  renderDocuments();

  // Keep polling while any document is still processing.
  const processing = state.documents.some((d) => (d.status || "ready") === "processing");
  if (processing && !state._docPoll) {
    state._docPoll = setInterval(loadDocuments, 3000);
  } else if (!processing && state._docPoll) {
    clearInterval(state._docPoll);
    state._docPoll = null;
  }
}

async function loadSessions() {
  try {
    state.sessions = await api("/api/sessions");
  } catch (err) {
    listError("sessions", "Không tải được danh sách phiên: " + err.message);
    return;
  }
  renderSessions();
}

async function uploadFile(file) {
  const status = document.getElementById("upload-status");
  status.classList.remove("hidden", "error");
  status.textContent = `Đang xử lý “${file.name}”… (ingest có thể mất chút thời gian)`;

  const form = new FormData();
  form.append("file", file);
  try {
    await api("/api/documents", { method: "POST", body: form });
    status.textContent = `Đã nạp “${file.name}”.`;
    await loadDocuments();
    setTimeout(() => status.classList.add("hidden"), 4000);
  } catch (err) {
    status.classList.add("error");
    status.textContent = "Lỗi upload: " + err.message;
  }
}

async function startSession(documentId) {
  const session = await api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId }),
  });
  await loadSessions();
  await openSession(session.id);
}

async function openSession(sessionId) {
  try {
    state.currentSession = await api(`/api/sessions/${sessionId}`);
  } catch (err) {
    const box = document.getElementById("messages");
    box.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Không mở được phiên: ${escapeHtml(err.message)}</p></div>`;
    return;
  }
  updateChatHeader();
  renderMessages();
  renderSessions();
}

async function deleteSession(sessionId) {
  await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
  if (state.currentSession && state.currentSession.id === sessionId) {
    state.currentSession = null;
    updateChatHeader();
    renderMessages();
  }
  await loadSessions();
}

async function sendMessage(content) {
  const session = state.currentSession;
  if (!session) return;

  // Optimistic: show the user message + a typing placeholder.
  session.messages.push({ role: "user", content });
  renderMessages();

  const box = document.getElementById("messages");
  const typing = messageEl("assistant", "Đang trả lời…");
  typing.querySelector(".bubble").classList.add("typing");
  box.appendChild(typing);
  box.scrollTop = box.scrollHeight;

  try {
    const msg = await api(`/api/sessions/${session.id}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    session.messages.push(msg);
    renderMessages();
    await loadSessions();
    renderSessions();
  } catch (err) {
    typing.querySelector(".bubble").classList.remove("typing");
    typing.querySelector(".bubble").textContent = "Lỗi: " + err.message;
  }
}

// --------------------------------------------------------------------------- //
// Wiring
// --------------------------------------------------------------------------- //
function init() {
  document.getElementById("file-input").addEventListener("change", (e) => {
    if (e.target.files.length) uploadFile(e.target.files[0]);
    e.target.value = "";
  });

  const input = document.getElementById("composer-input");
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      document.getElementById("composer").requestSubmit();
    }
  });

  document.getElementById("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const content = input.value.trim();
    if (!content) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(content);
  });

  loadDocuments();
  loadSessions();
}

init();

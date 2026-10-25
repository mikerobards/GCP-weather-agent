# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight Cloud Run Web UI that connects to the deployed GCP Agent Runtime."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import google.auth
import google.auth.transport.requests
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DEFAULT_AGENT_RUNTIME_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/"
    "projects/435438911776/locations/us-central1/"
    "reasoningEngines/5324101658383220736"
)

AGENT_RUNTIME_URL = os.getenv("AGENT_RUNTIME_URL", DEFAULT_AGENT_RUNTIME_URL).rstrip("/")

app = FastAPI(title="Weather Agent Web UI")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    user_id: str
    tool_calls: list[dict[str, Any]]


def _get_auth_headers() -> dict[str, str]:
    """Fetch a fresh OAuth2 Bearer token using Application Default Credentials."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }


def _create_session(headers: dict[str, str], user_id: str) -> str:
    """Create a new session on the Agent Runtime instance."""
    resp = requests.post(
        f"{AGENT_RUNTIME_URL}:query",
        headers=headers,
        json={
            "class_method": "async_create_session",
            "input": {"user_id": user_id},
        },
        timeout=30,
    )
    if not resp.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create Agent Runtime session ({resp.status_code}): {resp.text}",
        )
    session_id = (resp.json().get("output") or {}).get("id")
    if not session_id:
        raise HTTPException(
            status_code=502,
            detail="Agent Runtime returned a session response without an ID.",
        )
    return session_id


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "agent_runtime_url": AGENT_RUNTIME_URL}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Send a user message to the deployed Agent Runtime and return the response."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    headers = _get_auth_headers()
    user_id = req.user_id or f"web-{uuid.uuid4().hex[:8]}"
    session_id = req.session_id or _create_session(headers, user_id)

    payload = {
        "class_method": "async_stream_query",
        "input": {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
        },
    }

    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    with requests.post(
        f"{AGENT_RUNTIME_URL}:streamQuery",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        if not resp.ok:
            raise HTTPException(
                status_code=502,
                detail=f"Agent Runtime error ({resp.status_code}): {resp.text}",
            )
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = event.get("content")
            parts = content.get("parts", []) if isinstance(content, dict) else []
            for part in parts:
                if "function_call" in part and part["function_call"]:
                    fc = part["function_call"]
                    tool_calls.append(
                        {"name": fc.get("name"), "args": fc.get("args", {})}
                    )
                elif "text" in part and part["text"]:
                    text_chunks.append(part["text"])

    reply = "".join(text_chunks).strip() or "No response returned from the agent."
    return ChatResponse(
        reply=reply,
        session_id=session_id,
        user_id=user_id,
        tool_calls=tool_calls,
    )


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Weather Agent — GCP Agent Runtime</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --surface: #1e293b;
      --surface-hover: #334155;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-dark: #0284c7;
      --user-bubble: #0284c7;
      --agent-bubble: #1e293b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    header {
      width: 100%;
      max-width: 820px;
      padding: 24px 20px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      font-size: 28px;
    }
    .brand h1 {
      margin: 0;
      font-size: 1.25rem;
      font-weight: 600;
    }
    .brand p {
      margin: 2px 0 0;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .badge {
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    main {
      width: 100%;
      max-width: 820px;
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px 20px 24px;
      gap: 16px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .chip:hover {
      background: var(--surface-hover);
      border-color: var(--accent);
    }
    #chat-box {
      flex: 1;
      min-height: 420px;
      max-height: 62vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 16px;
      background: rgba(15, 23, 42, 0.55);
      border: 1px solid var(--border);
      border-radius: 14px;
    }
    .msg {
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 14px;
      line-height: 1.5;
      font-size: 0.94rem;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--user-bubble);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .msg.agent {
      align-self: flex-start;
      background: var(--agent-bubble);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
    }
    .msg.agent h3, .msg.agent h4 {
      margin: 10px 0 6px;
      color: var(--accent);
    }
    .msg.agent p { margin: 6px 0; }
    .msg.agent ul { margin: 6px 0; padding-left: 20px; }
    .tool-pill {
      display: inline-block;
      margin-bottom: 8px;
      font-size: 0.74rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.25);
    }
    form {
      display: flex;
      gap: 10px;
    }
    input[type="text"] {
      flex: 1;
      padding: 13px 16px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
    }
    input[type="text"]:focus {
      border-color: var(--accent);
    }
    button[type="submit"] {
      padding: 0 22px;
      border-radius: 10px;
      border: none;
      background: var(--accent-dark);
      color: #fff;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    button[type="submit"]:hover:not(:disabled) {
      background: #0369a1;
    }
    button[type="submit"]:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-icon">🌤️</span>
      <div>
        <h1>Weather Agent</h1>
        <p>Powered by Google ADK &amp; GCP Agent Runtime</p>
      </div>
    </div>
    <span class="badge">Live • Open-Meteo</span>
  </header>

  <main>
    <div class="chips">
      <button class="chip" onclick="sendQuick('What is the weather in Seattle, WA?')">🌧️ Seattle, WA</button>
      <button class="chip" onclick="sendQuick('Give me the forecast for zip code 10001')">🗽 NYC (10001)</button>
      <button class="chip" onclick="sendQuick('What is the weather in zip code 90210?')">🌴 Beverly Hills (90210)</button>
      <button class="chip" onclick="sendQuick('What is the weather like in Austin, TX?')">☀️ Austin, TX</button>
      <button class="chip" onclick="sendQuick('What is the weather in Tokyo?')">🗼 Tokyo</button>
    </div>

    <div id="chat-box">
      <div class="msg agent">
        Hello! Ask me for the current weather and 3-day forecast for any <strong>city name</strong> (e.g., <em>Seattle, WA</em> or <em>Tokyo</em>) or <strong>5-digit zip code</strong> (e.g., <em>10001</em> or <em>90210</em>).
      </div>
    </div>

    <form id="chat-form" onsubmit="handleSubmit(event)">
      <input
        id="user-input"
        type="text"
        placeholder="Ask for weather by city or zip code (e.g., 'Weather in Chicago' or '10001')..."
        autocomplete="off"
      />
      <button id="send-btn" type="submit">Send</button>
    </form>
  </main>

  <script>
    let sessionId = null;
    let userId = null;

    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    function appendMessage(role, contentHtml, toolCalls = []) {
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      let prefix = "";
      if (toolCalls && toolCalls.length > 0) {
        const pills = toolCalls
          .map(tc => `<span class="tool-pill">⚡ ${tc.name}(${JSON.stringify(tc.args)})</span>`)
          .join(" ");
        prefix = `<div>${pills}</div>`;
      }
      div.innerHTML = prefix + contentHtml;
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
      return div;
    }

    function sendQuick(text) {
      userInput.value = text;
      document.getElementById("chat-form").requestSubmit();
    }

    async function handleSubmit(e) {
      e.preventDefault();
      const text = userInput.value.trim();
      if (!text) return;

      userInput.value = "";
      sendBtn.disabled = true;
      appendMessage("user", text.replace(/</g, "&lt;").replace(/>/g, "&gt;"));
      const loadingDiv = appendMessage("agent", "<em>Fetching live weather from Agent Runtime...</em>");

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId, user_id: userId }),
        });
        const data = await res.json();
        chatBox.removeChild(loadingDiv);

        if (!res.ok) {
          appendMessage("agent", `<strong>Error:</strong> ${data.detail || "Request failed"}`);
        } else {
          sessionId = data.session_id;
          userId = data.user_id;
          appendMessage("agent", marked.parse(data.reply), data.tool_calls);
        }
      } catch (err) {
        chatBox.removeChild(loadingDiv);
        appendMessage("agent", `<strong>Network error:</strong> ${err.message}`);
      } finally {
        sendBtn.disabled = false;
        userInput.focus();
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page Weather Agent UI."""
    return HTMLResponse(content=HTML_PAGE)

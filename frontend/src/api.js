// Thin client for the Echo generation API. The token is held by the caller
// (App) and persisted in localStorage; every authed call sends it as Bearer.

async function request(path, { method = "GET", token, body } = {}) {
  const headers = {};
  if (body) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const resp = await fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.detail || `Request failed (${resp.status})`);
    err.status = resp.status;
    throw err;
  }
  return data;
}

export const api = {
  register: (username, password, signupSecret) =>
    request("/auth/register", {
      method: "POST",
      body: { username, password, signup_secret: signupSecret },
    }),

  login: (username, password) =>
    request("/auth/login", { method: "POST", body: { username, password } }),

  logout: (token) => request("/auth/logout", { method: "POST", token }),

  listSessions: (token) => request("/sessions", { token }),

  createSession: (token) => request("/sessions", { method: "POST", token }),

  getSession: (token, id) => request(`/sessions/${id}`, { token }),

  deleteSession: (token, id) =>
    request(`/sessions/${id}`, { method: "DELETE", token }),

  sendMessage: (token, id, message) =>
    request(`/sessions/${id}/message`, { method: "POST", token, body: { message } }),

  // Stream the agent's thinking/tool-calls/answer. Calls onEvent(event) for each
  // SSE message. EventSource can't send auth headers, so we read the fetch body.
  async streamMessage(token, id, message, onEvent, signal) {
    const resp = await fetch(`/sessions/${id}/message/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ message }),
      signal,
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      const err = new Error(data.detail || `Stream failed (${resp.status})`);
      err.status = resp.status;
      throw err;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (line) onEvent(JSON.parse(line.slice(6)));
      }
    }
  },
};

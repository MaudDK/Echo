import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { api } from "./api.js";
import Login from "./Login.jsx";
import ChatLayout from "./ChatLayout.jsx";
import ChatView from "./ChatView.jsx";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("echo_token") || "");

  function onAuthed(newToken) {
    localStorage.setItem("echo_token", newToken);
    setToken(newToken);
  }

  async function onLogout() {
    try {
      await api.logout(token);
    } catch {
      /* token may already be gone; clear locally regardless */
    }
    localStorage.removeItem("echo_token");
    setToken("");
  }

  if (!token) return <Login onAuthed={onAuthed} />;

  // A 401 (expired/revoked token) bubbles up to onAuthExpired so we drop to login.
  // The sidebar (ChatLayout) stays mounted across chats; only the active chat —
  // matched by /:sessionId — loads its own transcript lazily.
  return (
    <Routes>
      <Route element={<ChatLayout token={token} onLogout={onLogout} onAuthExpired={onLogout} />}>
        <Route index element={<ChatView />} />
        <Route path=":sessionId" element={<ChatView />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

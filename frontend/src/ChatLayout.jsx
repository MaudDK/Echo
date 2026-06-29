import { useEffect, useState, useCallback } from "react";
import { Outlet, NavLink, useNavigate, useLocation } from "react-router-dom";
import { api } from "./api.js";

// Owns the persistent shell: the session list + new/logout controls. The active
// chat is rendered through <Outlet/> (ChatView), which loads its own transcript.
export default function ChatLayout({ token, onLogout, onAuthExpired }) {
  const [sessions, setSessions] = useState([]);
  const [error, setError] = useState("");
  // Sidebar visibility. Below `md` it behaves as a drawer (closed by default,
  // opened by the hamburger); on `md+` it collapses/expands the column in place.
  // Default open on wide screens, closed on narrow.
  const [open, setOpen] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true
  );
  const navigate = useNavigate();
  const location = useLocation();

  const guard = useCallback(
    (err) => {
      if (err.status === 401) onAuthExpired();
      else setError(err.message);
    },
    [onAuthExpired]
  );

  // Only the lightweight list ({id, name, agent_name}) — never every transcript.
  const refreshSessions = useCallback(async () => {
    try {
      const { sessions } = await api.listSessions(token);
      setSessions(sessions);
      return sessions;
    } catch (err) {
      guard(err);
      return [];
    }
  }, [token, guard]);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  async function newSession() {
    try {
      const { session_id } = await api.createSession(token);
      await refreshSessions();
      navigate(`/${session_id}`);
      setOpen(false);
    } catch (err) {
      guard(err);
    }
  }

  async function removeSession(id, e) {
    e.preventDefault();
    e.stopPropagation();
    try {
      await api.deleteSession(token, id);
      const remaining = await refreshSessions();
      if (location.pathname === `/${id}`) {
        navigate(remaining.length ? `/${remaining[0].id}` : "/", { replace: true });
      }
    } catch (err) {
      guard(err);
    }
  }

  const itemBase =
    "flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg cursor-pointer text-sm transition-colors";

  return (
    <div className="relative flex h-full w-full bg-[#0f1115] text-[#e6e8ec]">
      {/* Backdrop: only rendered (and only visible) when the drawer is open on mobile. */}
      {open && (
        <div
          onClick={() => setOpen(false)}
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
        />
      )}

      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 shrink-0 overflow-hidden bg-[#171a21] transition-[width,transform] duration-200 ${
          open ? "w-60 translate-x-0" : "w-60 -translate-x-full md:w-0 md:translate-x-0"
        }`}
      >
        <div className="flex flex-col h-full w-60 border-r border-[#2a2f3a] p-3 gap-2.5">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-[#e6e8ec] px-1">Echo</span>
          <button
            onClick={() => setOpen(false)}
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
            className="rounded-lg px-2 py-1 text-base leading-none text-[#8b91a0] hover:bg-[#1e222b] hover:text-[#e6e8ec] transition-colors"
          >
            «
          </button>
        </div>

        <button
          onClick={newSession}
          className="w-full px-3 py-2.5 rounded-lg text-sm bg-[#1e222b] border border-[#2a2f3a] hover:bg-[#242833] transition-colors"
        >
          + New chat
        </button>

        <div className="flex-1 flex flex-col gap-1 overflow-y-auto">
          {sessions.map((s) => (
            <NavLink
              key={s.id}
              to={`/${s.id}`}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `${itemBase} ${
                  isActive
                    ? "bg-[#1e222b] text-[#e6e8ec]"
                    : "text-[#8b91a0] hover:bg-[#1e222b]"
                }`
              }
            >
              <span className="truncate">{s.name}</span>
              <button
                onClick={(e) => removeSession(s.id, e)}
                className="shrink-0 px-1.5 leading-none text-base text-[#8b91a0] hover:text-[#e0556a]"
                title="Delete chat"
              >
                ×
              </button>
            </NavLink>
          ))}
          {sessions.length === 0 && (
            <p className="text-sm text-[#8b91a0] px-2 py-2">No chats yet</p>
          )}
        </div>

        {error && <div className="text-[13px] text-[#e0556a] px-1">{error}</div>}

        <button
          onClick={onLogout}
          className="w-full px-3 py-2.5 rounded-lg text-sm bg-[#1e222b] border border-[#2a2f3a] hover:bg-[#242833] transition-colors"
        >
          Log out
        </button>
        </div>
      </aside>

      <div className="flex flex-col flex-1 min-w-0">
        {/* Mobile top bar with the drawer toggle; hidden once the sidebar has room. */}
        <header className="flex items-center gap-3 p-3 border-b border-[#2a2f3a] bg-[#171a21] md:hidden">
          <button
            onClick={() => setOpen(true)}
            aria-label="Open menu"
            className="rounded-lg border border-[#2a2f3a] bg-[#1e222b] px-3 py-1.5 text-lg leading-none hover:bg-[#242833] transition-colors"
          >
            ☰
          </button>
          <span className="text-sm text-[#8b91a0]">Echo</span>
        </header>

        {/* Desktop expand button, shown only when the sidebar is collapsed. */}
        {!open && (
          <button
            onClick={() => setOpen(true)}
            aria-label="Expand sidebar"
            title="Expand sidebar"
            className="hidden md:flex absolute top-3 left-3 z-20 rounded-lg border border-[#2a2f3a] bg-[#1e222b] px-3 py-1.5 text-lg leading-none hover:bg-[#242833] transition-colors"
          >
            ☰
          </button>
        )}

        <Outlet context={{ token, onAuthExpired, refreshSessions }} />
      </div>
    </div>
  );
}

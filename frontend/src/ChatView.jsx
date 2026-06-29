import { useEffect, useState, useRef } from "react";
import { useParams, useOutletContext, useNavigate } from "react-router-dom";
import { api } from "./api.js";
import Markdown from "./Markdown.jsx";

// Rebuild rich display turns from a stored transcript: pair each assistant's
// tool_calls with the tool-result messages that follow it.
function toDisplayTurns(messages) {
  const turns = [];
  let lastAssistant = null;
  for (const m of messages) {
    if (m.role === "user") {
      turns.push({ role: "user", content: m.content });
      lastAssistant = null;
    } else if (m.role === "assistant") {
      const calls = (m.tool_calls || []).map((tc) => ({
        name: tc.function?.name,
        args: tc.function?.arguments,
        result: null,
      }));
      lastAssistant = {
        role: "assistant",
        content: m.content || "",
        thinking: m.thinking || "",
        toolCalls: calls,
      };
      // An assistant turn with only tool calls and no text isn't worth a bubble
      // on its own; merge it into the next assistant turn via lastAssistant.
      turns.push(lastAssistant);
    } else if (m.role === "tool" && lastAssistant) {
      const slot = lastAssistant.toolCalls.find(
        (c) => c.name === m.tool_name && c.result === null
      );
      if (slot) slot.result = m.content;
    }
  }
  // Drop empty assistant shells (tool-only turns with nothing to show).
  return turns.filter(
    (t) => t.role === "user" || t.content || t.thinking || t.toolCalls.length
  );
}

function ToolCall({ call }) {
  return (
    <div className="flex flex-col gap-1 text-[13px]">
      <span className="self-start rounded-md border border-[#2a2f3a] bg-[#2c3340] px-2 py-0.5 text-[#cdd3e0]">
        🔧 {call.name}
      </span>
      <span className="font-mono text-xs text-[#8b91a0]">
        {JSON.stringify(call.args)}
      </span>
      {call.result != null && (
        <details className="rounded-lg border border-[#2a2f3a] bg-[#15181f] px-2.5 py-1.5 text-[#8b91a0]">
          <summary className="cursor-pointer select-none">result</summary>
          <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-[#e6e8ec]">
            {call.result}
          </pre>
        </details>
      )}
    </div>
  );
}

function AssistantBubble({ turn, live }) {
  return (
    <div className="flex flex-col gap-2 self-start max-w-[70%] rounded-xl bg-[#242833] px-3.5 py-2.5 leading-relaxed">
      {turn.thinking && (
        <details
          open={live}
          className="rounded-lg border border-[#2a2f3a] bg-[#15181f] px-2.5 py-1.5 text-[13px] text-[#8b91a0]"
        >
          <summary className="cursor-pointer select-none">💭 Thinking</summary>
          <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-[#e6e8ec]">
            {turn.thinking}
          </pre>
        </details>
      )}
      {turn.toolCalls.map((c, i) => (
        <ToolCall key={i} call={c} />
      ))}
      {turn.content && <Markdown>{turn.content}</Markdown>}
      {live && !turn.content && <span className="animate-blink">▍</span>}
    </div>
  );
}

export default function ChatView() {
  const { sessionId } = useParams();
  const { token, onAuthExpired, refreshSessions } = useOutletContext();
  const navigate = useNavigate();

  const [turns, setTurns] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);
  // When we create a session locally and navigate to it, skip the load that the
  // sessionId change would otherwise trigger — it'd wipe the optimistic turns.
  const skipLoadRef = useRef(null);
  // Lets the Stop button abort the in-flight streaming fetch.
  const abortRef = useRef(null);

  function stop() {
    abortRef.current?.abort();
  }

  function guard(err) {
    if (err.status === 401) onAuthExpired();
    else setError(err.message);
  }

  // Lazily load ONLY the active chat's transcript when the route changes.
  useEffect(() => {
    setError("");
    if (!sessionId) {
      setTurns([]);
      return;
    }
    if (skipLoadRef.current === sessionId) {
      skipLoadRef.current = null;
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const { messages } = await api.getSession(token, sessionId);
        if (!cancelled) setTurns(toDisplayTurns(messages));
      } catch (err) {
        if (!cancelled) guard(err);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, token]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function send(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    let id = sessionId;
    if (!id) {
      try {
        const { session_id } = await api.createSession(token);
        id = session_id;
        skipLoadRef.current = id; // the navigate below must not reload over us
        navigate(`/${id}`);
        refreshSessions();
      } catch (err) {
        return guard(err);
      }
    }

    setInput("");
    setError("");
    setSending(true);

    // Append the user turn and a live assistant turn we mutate as events stream.
    const assistant = { role: "assistant", content: "", thinking: "", toolCalls: [] };
    setTurns((t) => [...t, { role: "user", content: text }, assistant]);
    const flush = () => setTurns((t) => [...t.slice(0, -1), { ...assistant }]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.streamMessage(token, id, text, (ev) => {
        switch (ev.type) {
          case "thinking":
            assistant.thinking += ev.text;
            break;
          case "token":
            assistant.content += ev.text;
            break;
          case "tool_call":
            assistant.toolCalls = [
              ...assistant.toolCalls,
              { name: ev.name, args: ev.args, result: null },
            ];
            break;
          case "tool_result": {
            const slot = [...assistant.toolCalls]
              .reverse()
              .find((c) => c.name === ev.name && c.result === null);
            if (slot) slot.result = ev.result;
            break;
          }
          case "answer":
            assistant.content = ev.text;
            break;
          case "error":
            assistant.content = `⚠️ ${ev.detail}`;
            break;
          default:
            break;
        }
        flush();
      }, controller.signal);
    } catch (err) {
      // Aborting via the Stop button is intentional, not an error; keep
      // whatever partial answer already streamed in.
      if (err.name !== "AbortError") guard(err);
    } finally {
      abortRef.current = null;
      setSending(false);
      refreshSessions();
    }
  }

  return (
    <main className="flex flex-col flex-1 min-h-0 min-w-0">
      <div className="flex-1 overflow-y-auto">
        <div className="flex flex-col gap-3 w-full max-w-4xl min-h-full mx-auto px-4 py-6">
        {turns.length === 0 && (
          <div className="m-auto text-[#8b91a0]">Ask your agent anything.</div>
        )}
        {turns.map((t, i) =>
          t.role === "user" ? (
            <div
              key={i}
              className="self-end max-w-[70%] rounded-xl bg-[#2b4a8b] px-3.5 py-2.5 whitespace-pre-wrap break-words leading-relaxed"
            >
              {t.content}
            </div>
          ) : (
            <AssistantBubble
              key={i}
              turn={t}
              live={sending && i === turns.length - 1}
            />
          )
        )}
        <div ref={endRef} />
        </div>
      </div>

      {error && (
        <div className="w-full max-w-4xl mx-auto px-4 py-2 text-[13px] text-[#e0556a]">
          {error}
        </div>
      )}

      <form
        onSubmit={send}
        className="border-t border-[#2a2f3a] bg-[#171a21]"
      >
        <div className="flex gap-2 w-full max-w-4xl mx-auto px-4 py-4">
          <input
            placeholder="Message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 rounded-lg border border-[#2a2f3a] bg-[#1e222b] px-3 py-2.5 text-sm text-[#e6e8ec] outline-none focus:border-[#5b8cff]"
          />
          {sending ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-lg bg-[#e0556a] px-5 py-2.5 text-sm text-white"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded-lg bg-[#5b8cff] px-5 py-2.5 text-sm text-white disabled:opacity-50 disabled:cursor-default"
            >
              Send
            </button>
          )}
        </div>
      </form>
    </main>
  );
}

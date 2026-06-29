import { useState } from "react";
import { api } from "./api.js";

export default function Login({ onAuthed }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [signupSecret, setSignupSecret] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "register") {
        await api.register(username, password, signupSecret);
      }
      const { token } = await api.login(username, password);
      onAuthed(token);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const inputCls =
    "rounded-lg border border-[#2a2f3a] bg-[#1e222b] px-3 py-2.5 text-sm text-[#e6e8ec] outline-none focus:border-[#5b8cff]";

  return (
    <div className="flex h-full w-full items-center justify-center bg-[#0f1115] text-[#e6e8ec]">
      <form
        onSubmit={submit}
        className="flex flex-col gap-3 w-80 p-8 rounded-2xl bg-[#171a21] border border-[#2a2f3a]"
      >
        <h1 className="m-0 text-center text-2xl font-semibold">Echo</h1>
        <p className="m-0 mb-2 text-center text-[#8b91a0]">
          {mode === "login" ? "Sign in to your agents" : "Create an account"}
        </p>

        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
          className={inputCls}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className={inputCls}
        />

        {mode === "register" && (
          <input
            type="password"
            placeholder="Signup secret"
            value={signupSecret}
            onChange={(e) => setSignupSecret(e.target.value)}
            className={inputCls}
          />
        )}

        {error && <div className="text-[13px] text-[#e0556a]">{error}</div>}

        <button
          type="submit"
          disabled={busy || !username || !password}
          className="rounded-lg bg-[#5b8cff] px-3 py-2.5 text-sm text-white disabled:opacity-50 disabled:cursor-default"
        >
          {busy ? "…" : mode === "login" ? "Log in" : "Register"}
        </button>

        <button
          type="button"
          onClick={() => {
            setError("");
            setMode(mode === "login" ? "register" : "login");
          }}
          className="bg-transparent p-1 text-sm text-[#8b91a0] hover:text-[#e6e8ec]"
        >
          {mode === "login"
            ? "Need an account? Register"
            : "Have an account? Log in"}
        </button>
      </form>
    </div>
  );
}

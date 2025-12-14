import { useState } from "react";

const API_BASE = "https://ml-api-ec2k.onrender.com";

export default function Login({ onLoggedIn }) {
  const [nationalId, setNationalId] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setErr("");

    if (!/^\d{10}$/.test(nationalId)) {
      setErr("National ID must be exactly 10 digits.");
      return;
    }
    if (password.length < 6) {
      setErr("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ national_id: nationalId, password }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        setErr(data?.error || "Login failed.");
        return;
      }

      sessionStorage.setItem("token", data.access_token);
      onLoggedIn?.();
    } catch {
      setErr("Unable to reach the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-card">
      <h2 className="login-title">System Login</h2>
      <p className="login-sub">Sign in to access the Equipment Monitoring Dashboard.</p>
  
      {err && <div className="login-error">{err}</div>}
  
      <form onSubmit={submit}>
        <label className="login-label">National ID</label>
        <input
          className="login-input"
          placeholder="e.g., 1234567890"
          value={nationalId}
          onChange={(e) => setNationalId(e.target.value)}
          maxLength={10}
          inputMode="numeric"
        />
  
        <label className="login-label">Password</label>
        <input
          className="login-input"
          type="password"
          placeholder="Enter your password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
  
        <button className="login-btn" disabled={loading}>
          {loading ? "Signing in..." : "Sign In"}
        </button>
      </form>
    </div>
  );

}

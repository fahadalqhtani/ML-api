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
    <div style={{ maxWidth: 420, margin: "70px auto", padding: 24 }}>
      <h2 style={{ marginBottom: 8 }}>System Login</h2>
      <p style={{ marginTop: 0, marginBottom: 16, opacity: 0.8 }}>
        Sign in to access the Equipment Monitoring Dashboard.
      </p>

      {err && (
        <div
          style={{
            background: "#ffe5e5",
            padding: 12,
            borderRadius: 10,
            marginBottom: 12,
            border: "1px solid #ffb3b3",
          }}
        >
          {err}
        </div>
      )}

      <form onSubmit={submit}>
        <label style={{ display: "block", marginBottom: 6 }}>National ID</label>
        <input
          placeholder="e.g., 1234567890"
          value={nationalId}
          onChange={(e) => setNationalId(e.target.value)}
          maxLength={10}
          inputMode="numeric"
          style={{
            width: "100%",
            padding: 10,
            marginBottom: 12,
            borderRadius: 10,
            border: "1px solid #ddd",
          }}
        />

        <label style={{ display: "block", marginBottom: 6 }}>Password</label>
        <input
          type="password"
          placeholder="Enter your password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{
            width: "100%",
            padding: 10,
            marginBottom: 12,
            borderRadius: 10,
            border: "1px solid #ddd",
          }}
        />

        <button
          disabled={loading}
          style={{
            width: "100%",
            padding: 10,
            borderRadius: 10,
            border: "none",
            cursor: "pointer",
          }}
        >
          {loading ? "Signing in..." : "Sign In"}
        </button>
      </form>
    </div>
  );
}

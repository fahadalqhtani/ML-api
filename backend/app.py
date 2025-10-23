import eventlet
eventlet.monkey_patch()


import os
from datetime import datetime
from pathlib import Path
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from sqlalchemy import create_engine, text
from joblib import load as joblib_load

# ===================================================
# Configuration
# ===================================================

def _with_sslmode_require(url: str) -> str:
    """Ensure Render Postgres uses SSL."""
    if not url:
        return url
    if "sslmode=" in url:
        return url
    return url + ("&sslmode=require" if "?" in url else "?sslmode=require")

DATABASE_URL = _with_sslmode_require(os.getenv("DATABASE_URL", ""))
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL. Add it in Render Environment settings.")

MODEL_PATH = os.getenv("MODEL_PATH", "best_DT.pkl")
RISK_THRESHOLD = int(os.getenv("RISK_THRESHOLD", "85"))  # % threshold for failure

# ===================================================
# Initialize App, Database, Model
# ===================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = joblib_load(MODEL_PATH)


# ===================================================
# Utility Functions
# ===================================================

def equipment_code_from_name(name: str) -> int:
    """Map equipment name to code (compressor=0, pump=1, turbine=2)."""
    n = name.lower().strip()
    if n.startswith("compressor"):
        return 0
    if n.startswith("pump"):
        return 1
    if n.startswith("turbine") or n.startswith("turpin"):
        return 2
    raise ValueError(f"Unknown equipment type for name: {name!r}")

def parse_timestamp(ts_str: str) -> datetime:
    """Parse date like 'YYYY/MM/DD hh:mm' or 'YYYY-MM-DD hh:mm:ss'."""
    ts_str = ts_str.replace("-", "/").strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass
    raise ValueError("Invalid date format. Use 'YYYY/MM/DD hh:mm'.")

def compute_risk_score(temperature, vibration, pressure, equipment_code) -> int:
    """Predict failure probability using the trained model."""
    X = np.array([[float(temperature), float(pressure), float(vibration), int(equipment_code)]], dtype=float)
    if hasattr(model, "predict_proba"):
        proba_faulty = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        from math import exp
        z = float(model.decision_function(X)[0])
        proba_faulty = 1.0 / (1.0 + exp(-z))
    else:
        proba_faulty = float(model.predict(X)[0])  # fallback
    return int(round(proba_faulty * 100))


# ===================================================
# API Routes
# ===================================================

@app.get("/health")
def health():
    return "OK", 200


@app.post("/ingest")
def ingest():
    """
    Receives JSON:
    {
      "date": "YYYY/MM/DD hh:mm",
      "equipment_name": "pump101",
      "temperature": 73.4,
      "vibration": 1.57,
      "pressure": 29.9,
      "equipment_code": 1   # optional
    }
    """
    try:
        data = request.get_json(force=True)
        ts = parse_timestamp(str(data["date"]))
        name = str(data["equipment_name"]).strip()
        temperature = float(data["temperature"])
        vibration = float(data["vibration"])
        pressure = float(data["pressure"])
        equipment_code = int(data["equipment_code"]) if "equipment_code" in data else equipment_code_from_name(name)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid input: {e}"}), 400

    risk_score = compute_risk_score(temperature, vibration, pressure, equipment_code)
    failurety = 1 if risk_score >= RISK_THRESHOLD else 0
    ts_db = ts.strftime("%Y-%m-%d %H:%M:%S")

    # ===================== DB Write =====================
    try:
        with engine.begin() as conn:
            # Ensure equipment exists (no deletion)
            conn.execute(
                text("INSERT INTO equipment (name) VALUES (:name) ON CONFLICT (name) DO NOTHING"),
                {"name": name},
            )

            # Insert reading
            # Insert reading (without equipment_code)
            reading_id = conn.execute(
                text("""
                    INSERT INTO reading (equipment_name, temperature, pressure, vibration, timestamp)
                    VALUES (:equipment_name, :temperature, :pressure, :vibration, :timestamp)
                    RETURNING id
                """),
                {
                    "equipment_name": name,
                    "temperature": temperature,
                    "pressure": pressure,
                    "vibration": vibration,
                    "timestamp": ts_db,
                },
            ).scalar_one()


            # Insert prediction
            conn.execute(
                text("""
                    INSERT INTO prediction (reading_id, prediction, probability, timestamp)
                    VALUES (:reading_id, :prediction, :probability, :timestamp)
                """),
                {
                    "reading_id": reading_id,
                    "prediction": failurety,
                    "probability": risk_score / 100.0,
                    "timestamp": ts_db,
                },
            )
    except Exception as e:
        return jsonify({"ok": False, "error": f"Database write failed: {e}"}), 500

    # ===================== WebSocket Broadcast =====================
    socketio.emit("reading_update", {
        "date": ts.strftime("%H:%M"),
        "equipment_name": name,
        "equipment_code": equipment_code,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "risk_score": risk_score,
    })

    # ===================== HTTP Response =====================
    return jsonify({
        "ok": True,
        "data": {
            "date": ts.strftime("%Y/%m/%d %H:%M"),
            "equipment_name": name,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "risk_score": risk_score,
            "failurety": failurety
        }
    }), 200

@app.get("/equipment")
def list_equipment():
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT name FROM equipment ORDER BY name ASC")).all()
        names = [r[0] for r in rows]
        return jsonify({"ok": True, "equipment": names})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/latest")
def latest():
    name = request.args.get("equipment_name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "equipment_name is required"}), 400

    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT r.temperature, r.vibration, r.pressure, r.timestamp,
                       p.probability, p.prediction
                FROM reading r
                JOIN prediction p ON p.reading_id = r.id
                WHERE r.equipment_name = :name
                ORDER BY r.id DESC
                LIMIT 1
            """), {"name": name}).mappings().first()

        if row is None:
            return jsonify({"ok": True, "data": None}), 200

        data = {
            "temperature": float(row["temperature"]),
            "vibration": float(row["vibration"]),
            "pressure": float(row["pressure"]),
            "timestamp": row["timestamp"].isoformat(),
            "risk_score": round(float(row["probability"]) * 100),
            "prediction": int(row["prediction"]),
        }
        return jsonify({"ok": True, "data": data}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500




# ===================================================
# Entry Point
# ===================================================

if __name__ == "__main__":
    import eventlet
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

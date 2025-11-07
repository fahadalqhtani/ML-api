# app.py
import eventlet
eventlet.monkey_patch()

import os
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from sqlalchemy import create_engine, text
from joblib import load as joblib_load

# ===================================================
# Configuration
# ===================================================

def _with_sslmode_require(url: str) -> str:
    """Ensure Postgres uses SSL on Render-like platforms."""
    if not url:
        return url
    if "sslmode=" in url:
        return url
    return url + ("&sslmode=require" if "?" in url else "?sslmode=require")

DATABASE_URL = _with_sslmode_require(os.getenv("DATABASE_URL", ""))
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL. Set it in your environment.")

MODEL_PATH     = os.getenv("MODEL_PATH", "best_rf.pkl")  # اسم ملف الموديل
RISK_THRESHOLD = int(os.getenv("RISK_THRESHOLD", "85"))  # %
TEST_CSV_PATH  = os.getenv("TEST_CSV_PATH", "test.csv")  # بيانات السيموليشن
SIM_INTERVAL   = float(os.getenv("SIM_INTERVAL", "5"))   # ثواني

# خريطة الكود إلى اسم الجهاز
CODE_TO_NAME = {
    0: "Compressor",
    1: "Pump",
    2: "Turbine",
}

# ===================================================
# Initialize App, DB, Model
# ===================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    ping_interval=20,   # نبض كل 20 ثانية
    ping_timeout=30,    # مهلة الرد 30 ثانية
    path="/socket.io",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
model = joblib_load(MODEL_PATH)

# ===================================================
# Utilities
# ===================================================

def equipment_code_from_name(name: str) -> int:
    n = name.lower().strip()
    if n.startswith("compressor"):
        return 0
    if n.startswith("pump"):
        return 1
    if n.startswith("turbine") or n.startswith("turpin"):
        return 2
    return 1  # default pump

def parse_timestamp(ts_str: str) -> datetime:
    ts = ts_str.replace("-", "/").strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            pass
    return datetime.utcnow()

def compute_risk_score(temperature, vibration, pressure, equipment_code) -> int:
    """Use trained model to produce failure probability in %."""
    X = np.array([[float(temperature), float(pressure), float(vibration), int(equipment_code)]], dtype=float)
    if hasattr(model, "predict_proba"):
        proba_faulty = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        from math import exp
        z = float(model.decision_function(X)[0])
        proba_faulty = 1.0 / (1.0 + exp(-z))
    else:
        proba_faulty = float(model.predict(X)[0])
    return int(round(proba_faulty * 100))

def upsert_and_insert_reading(name, ts, temperature, vibration, pressure, risk_score):
    """Write reading+prediction to DB and notify frontend."""
    failurety = 1 if risk_score >= RISK_THRESHOLD else 0
    ts_db = ts.strftime("%Y-%m-%d %H:%M:%S")

    with engine.begin() as conn:
        # تأكد من وجود الجهاز
        conn.execute(
            text("INSERT INTO equipment (name) VALUES (:name) ON CONFLICT (name) DO NOTHING"),
            {"name": name},
        )
        # إدخال القراءة
        reading_id = conn.execute(
            text("""
                INSERT INTO reading (equipment_name, temperature, pressure, vibration, timestamp)
                VALUES (:equipment_name, :temperature, :pressure, :vibration, :timestamp)
                RETURNING id
            """),
            {
                "equipment_name": name,
                "temperature": float(temperature),
                "pressure": float(pressure),
                "vibration": float(vibration),
                "timestamp": ts_db,
            },
        ).scalar_one()

        # إدخال التنبؤ
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

    # بث للفرونت-إند (تحديث فوري)
    socketio.emit("reading_update", {
        "date": ts.strftime("%H:%M:%S"),
        "equipment_name": name,
        "temperature": float(temperature),
        "vibration": float(vibration),
        "pressure": float(pressure),
        "risk_score": int(risk_score),
    })

# ===================================================
# Simulation: one row per equipment code every interval
# ===================================================

def simulate_from_csv_triplet(csv_path: str = TEST_CSV_PATH, interval: float = SIM_INTERVAL):
    """
    Every `interval` seconds, emit/store one reading per equipment_code group.
    CSV columns required: temperature, pressure, vibration, equipment_code
    """
    print(f"📡 Simulation starting from {csv_path} (every {interval}s)")
    if not Path(csv_path).exists():
        print(f"⚠️ CSV not found: {csv_path}. Simulation aborted.")
        return

    df = pd.read_csv(csv_path)
    required = {"temperature", "pressure", "vibration", "equipment_code"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{csv_path} must contain columns: {required}. Found: {set(df.columns)}")

    # cycles per equipment_code
    groups = {}
    for code, g in df.groupby("equipment_code"):
        g = g[["temperature", "pressure", "vibration"]].reset_index(drop=True)
        if len(g) == 0:
            continue
        groups[int(code)] = itertools.cycle(g.to_dict("records"))

    if not groups:
        print("⚠️ No groups found in CSV. Simulation aborted.")
        return

    codes = sorted(groups.keys())

    # warm-up: ensure devices exist
    try:
        with engine.begin() as conn:
            for code in codes:
                conn.execute(
                    text("INSERT INTO equipment (name) VALUES (:name) ON CONFLICT (name) DO NOTHING"),
                    {"name": CODE_TO_NAME.get(code, f"device_{code}")},
                )
    except Exception as e:
        print("DB warmup error:", e)

    print(f"▶️ Simulation groups: {codes} | interval={interval}s")

    while True:
        for code in codes:
            try:
                sample = next(groups[code])
                name = CODE_TO_NAME.get(code, f"device_{code}")
                temp = float(sample["temperature"])
                pres = float(sample["pressure"])
                vib  = float(sample["vibration"])

                risk = compute_risk_score(temp, vib, pres, code)
                upsert_and_insert_reading(name, datetime.utcnow(), temp, vib, pres, risk)

            except Exception as e:
                print(f"⚠️ Simulation error for code={code}: {e}")
                continue

        eventlet.sleep(interval)

# ===================================================
# API Routes
# ===================================================

@app.get("/health")
def health():
    return "OK", 200

@app.get("/")
def root():
    # يُستخدم للتسخين ولفحص جاهزية الخدمة من الواجهة
    return jsonify({"ok": True, "service": "equipment-monitoring", "socket": True}), 200

@app.post("/ingest")
def ingest():
    """
    JSON:
    {
      "date": "YYYY/MM/DD hh:mm[:ss]" (optional),
      "equipment_name": "Pump",
      "temperature": 73.4,
      "vibration": 1.57,
      "pressure": 29.9
    }
    """
    try:
        data = request.get_json(force=True)
        ts = parse_timestamp(str(data.get("date", ""))) if data.get("date") else datetime.utcnow()
        name = str(data["equipment_name"]).strip()
        temperature = float(data["temperature"])
        vibration = float(data["vibration"])
        pressure = float(data["pressure"])
        code = equipment_code_from_name(name)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid input: {e}"}), 400

    risk_score = compute_risk_score(temperature, vibration, pressure, code)
    upsert_and_insert_reading(name, ts, temperature, vibration, pressure, risk_score)

    return jsonify({
        "ok": True,
        "data": {
            "date": ts.strftime("%Y/%m/%d %H:%M:%S"),
            "equipment_name": name,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "risk_score": risk_score,
            "failurety": 1 if risk_score >= RISK_THRESHOLD else 0
        }
    }), 200

@app.get("/equipment")
def list_equipment():
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT name FROM equipment ORDER BY name ASC")).all()
        return jsonify({"ok": True, "equipment": [r[0] for r in rows]})
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
            return jsonify({"ok": True, "data": None})

        data = {
            "temperature": float(row["temperature"]),
            "vibration": float(row["vibration"]),
            "pressure": float(row["pressure"]),
            "timestamp": row["timestamp"].isoformat(),
            "risk_score": round(float(row["probability"]) * 100),
            "prediction": int(row["prediction"]),
        }
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ===================================================
# Start background simulation once (Flask 3 friendly)
# ===================================================

_SIM_STARTED = False
def _ensure_simulation_started():
    """Run CSV simulation once per process (skip if DISABLE_SIM=1)."""
    global _SIM_STARTED
    if not _SIM_STARTED and os.getenv("DISABLE_SIM", "0") != "1":
        _SIM_STARTED = True
        socketio.start_background_task(simulate_from_csv_triplet, TEST_CSV_PATH, SIM_INTERVAL)

# ابدأ المحاكاة فور تحميل التطبيق (بديل before_first_request في Flask 3)
_ensure_simulation_started()

# Socket.IO logging helpers (اختياري)
@socketio.on('connect')
def on_connect():
    print('🔌 client connected')

# Local run
if __name__ == "__main__":
    _ensure_simulation_started()
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

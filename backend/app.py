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
    """Ensure Postgres uses SSL (required for Render and similar platforms)."""
    if not url:
        return url
    if "sslmode=" in url:
        return url
    return url + ("&sslmode=require" if "?" in url else "?sslmode=require")

DATABASE_URL = _with_sslmode_require(os.getenv("DATABASE_URL", ""))
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL. Set it in your environment.")


MODEL_PATH     = os.getenv("MODEL_PATH", "best_xgb_fold_model.pkl")
RISK_THRESHOLD = int(os.getenv("RISK_THRESHOLD", "85"))  # percentage
TEST_CSV_PATH  = os.getenv("TEST_CSV_PATH", "test.csv")
SIM_INTERVAL   = float(os.getenv("SIM_INTERVAL", "5"))   # seconds

CODE_TO_NAME = {0: "Compressor", 1: "Pump", 2: "Turbine"}

# ===================================================
# Initialize App, DB, Model
# ===================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    ping_interval=20,
    ping_timeout=30,
    path="/socket.io",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib_load(MODEL_PATH)

# ===================================================
# SHAP Initialization
# ===================================================

FEATURE_NAMES = ["temperature", "pressure", "vibration", "humidity", "equipment_code"]
SENSOR_FEATURES = ["temperature", "pressure", "vibration", "humidity"]

try:
    import shap
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    _SHAP_OK = True
except Exception:
    explainer = None
    _SHAP_OK = False

# ===================================================
# Baselines (for direction high/low)
# ===================================================

GLOBAL_BASELINE = {
    "temperature": None,
    "pressure": None,
    "vibration": None,
    "humidity": None,
}
try:
    if Path(TEST_CSV_PATH).exists():
        _df_base = pd.read_csv(TEST_CSV_PATH)
        for k in ["temperature", "pressure", "vibration", "humidity"]:
            if k in _df_base.columns and _df_base[k].notna().any():
                GLOBAL_BASELINE[k] = float(_df_base[k].median())
except Exception:
    # keep None if anything goes wrong
    pass

def feature_direction(feature: str, value: float) -> str:
    """Return 'high' or 'low' against a global median; falls back to 'abnormal'."""
    base = GLOBAL_BASELINE.get(feature)
    if base is None:
        return "abnormal"
    return "high" if float(value) > base else "low"

# ===================================================
# Utility Functions
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

#  model input: [temp, pressure, vibration, humidity, equipment_code]
def compute_risk_score(temperature, vibration, pressure, humidity, equipment_code) -> int:
    """Use the trained model to produce a failure probability in %."""
    X = np.array(
        [[float(temperature), float(pressure), float(vibration), float(humidity), int(equipment_code)]],
        dtype=float,
    )
    if hasattr(model, "predict_proba"):
        proba_faulty = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        from math import exp
        z = float(model.decision_function(X)[0])
        proba_faulty = 1.0 / (1.0 + exp(-z))
    else:
        proba_faulty = float(model.predict(X)[0])
    return int(round(proba_faulty * 100))

# ---------- SHAP helpers (multi-cause) ----------

def _extract_shap_values(x_row: np.ndarray):
    """Return list of (feature, shap_value) for sensor features only."""
    if not _SHAP_OK:
        return []
    exp = explainer(x_row)
    vals = getattr(exp, "values", None)
    if vals is None:
        return []
    if vals.ndim == 3:               # (n_samples, n_features, n_classes)
        cls_idx = 1 if vals.shape[-1] > 1 else 0
        raw = vals[0, :, cls_idx]
    elif vals.ndim == 2:             # (n_samples, n_features)
        raw = vals[0, :]
    else:                            # (n_features,)
        raw = vals
    return [(f, float(v)) for f, v in zip(FEATURE_NAMES, raw.tolist()) if f in SENSOR_FEATURES]

def shap_top_causes(temperature, vibration, pressure, humidity, equipment_code,
                    min_share=0.20, coverage=0.80, max_causes=3):
    """
    Return a ranked list of (feature, contribution, share) for 1..3 causes.
    """
    if not _SHAP_OK:
        return []

    x_row = np.array(
        [[float(temperature), float(pressure), float(vibration), float(humidity), int(equipment_code)]],
        dtype=float
    )
    pairs = _extract_shap_values(x_row)
    if not pairs:
        return []

    positives = [(f, v) for f, v in pairs if v > 0]
    if not positives:
        positives = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:1]

    positives.sort(key=lambda t: t[1], reverse=True)
    total_pos = sum(v for _, v in positives) or 1e-9

    ranked = []
    cum_share = 0.0
    for i, (f, v) in enumerate(positives):
        share = v / total_pos
        if i == 0:
            ranked.append((f, v, share))
            cum_share += share
            continue
        if len(ranked) < max_causes and (share >= min_share or cum_share < coverage):
            ranked.append((f, v, share))
            cum_share += share
        if len(ranked) >= max_causes or cum_share >= coverage:
            break
    return ranked

def build_warning_message_multi(device_name: str, causes: list, cur_values: dict) -> str:
    """
    Compose a clean message using 1..3 causes (no time, no 'Warning:' prefix).
    """
    label = {
        "temperature": "temperature",
        "vibration": "vibration",
        "pressure": "pressure",
        "humidity": "humidity",
    }
    parts = []
    for f, _, _ in causes:
        direction = feature_direction(f, cur_values.get(f))
        token = "abnormal" if direction == "abnormal" else direction
        parts.append(f"{token} {label.get(f, f)}")
    if len(parts) == 1:
        cause_text = parts[0]
    elif len(parts) == 2:
        cause_text = " and ".join(parts)
    else:
        cause_text = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"{cause_text[0].upper() + cause_text[1:]} detected on {device_name}"

# --------------------------------------------------

def upsert_and_insert_reading(name, ts, temperature, vibration, pressure, humidity, risk_score):
    """Insert reading + prediction into DB and emit to frontend.
       Returns True on success, False on failure.
    """
    try:
        failurety = 1 if risk_score >= RISK_THRESHOLD else 0
        ts_db = ts.strftime("%Y-%m-%d %H:%M:%S")

        message = None
        if failurety == 1:
            code = equipment_code_from_name(name)
            causes = shap_top_causes(temperature, vibration, pressure, humidity, code)
            if causes:
                cur_vals = {
                    "temperature": float(temperature),
                    "pressure": float(pressure),
                    "vibration": float(vibration),
                    "humidity": float(humidity),
                }
                message = build_warning_message_multi(name, causes, cur_vals)

        with engine.begin() as conn:
             exists = conn.execute(
                        text("SELECT 1 FROM equipment WHERE name = :name"),
                        {"name": name},
                    ).scalar_one_or_none()

            if exists is None:
           
                return False
            # Insert reading
            reading_id = conn.execute(
                text("""
                    INSERT INTO reading (equipment_name, temperature, pressure, vibration, humidity, timestamp)
                    VALUES (:equipment_name, :temperature, :pressure, :vibration, :humidity, :timestamp)
                    RETURNING id
                """),
                {
                    "equipment_name": name,
                    "temperature": float(temperature),
                    "pressure": float(pressure),
                    "vibration": float(vibration),
                    "humidity": float(humidity),
                    "timestamp": ts_db,
                },
            ).scalar_one()

            # Insert prediction
            conn.execute(
                text("""
                    INSERT INTO prediction (reading_id, prediction, probability, timestamp, message)
                    VALUES (:reading_id, :prediction, :probability, :timestamp, :message)
                """),
                {
                    "reading_id": reading_id,
                    "prediction": failurety,
                    "probability": risk_score / 100.0,
                    "timestamp": ts_db,
                    "message": message,
                },
            )

        # Emit socket event
        payload = {
            "date": ts.strftime("%H:%M:%S"),
            "equipment_name": name,
            "temperature": float(temperature),
            "vibration": float(vibration),
            "pressure": float(pressure),
            "humidity": float(humidity),
            "risk_score": int(risk_score),
        }
        if message:
            payload["message"] = message
        socketio.emit("reading_update", payload)

        return True   # SUCCESS

    except Exception as e:
        print("DB error:", e)
        return False  # FAILURE


# ===================================================
# Simulation
# ===================================================


_SIM_TASK = None
_SIM_RUNNING = False


_SIM_GROUPS = {}   # code -> itertools.cycle([...])
_SIM_CODES  = []   # list of codes المتاحة

def _init_sim_groups(csv_path: str = TEST_CSV_PATH):
    """Load CSV and prepare global cycles once (for resume)."""
    global _SIM_GROUPS, _SIM_CODES

    if _SIM_GROUPS:
       
        return

    print(f"📄 Initializing simulation groups from: {csv_path}")
    if not Path(csv_path).exists():
        print(f"⚠️ CSV not found: {csv_path}. Cannot initialize simulation groups.")
        _SIM_GROUPS = {}
        _SIM_CODES = []
        return

    df = pd.read_csv(csv_path)
    required = {"temperature", "pressure", "vibration", "humidity", "equipment_code"}
    if not required.issubset(df.columns):
        print(f"❌ CSV missing required columns. Found: {set(df.columns)}")
        _SIM_GROUPS = {}
        _SIM_CODES = []
        return

    groups = {}
    for code, g in df.groupby("equipment_code"):
        g = g[["temperature", "pressure", "vibration", "humidity"]].reset_index(drop=True)
        if len(g) == 0:
            continue
        groups[int(code)] = itertools.cycle(g.to_dict("records"))

    if not groups:
        print("⚠️ No groups found in CSV. Cannot initialize simulation groups.")
        _SIM_GROUPS = {}
        _SIM_CODES = []
        return

    _SIM_GROUPS = groups
    _SIM_CODES = sorted(groups.keys())

    # Ensure devices exist once
    try:
        with engine.begin() as conn:
            for code in _SIM_CODES:
                conn.execute(
                    text("INSERT INTO equipment (name) VALUES (:name) ON CONFLICT (name) DO NOTHING"),
                    {"name": CODE_TO_NAME.get(code, f"device_{code}")},
                )
    except Exception as e:
        print("DB warmup error:", e)

    print(f" Simulation groups ready: codes={_SIM_CODES}")

def simulate_from_csv_triplet(csv_path: str = TEST_CSV_PATH, interval: float = SIM_INTERVAL):
   
    global _SIM_RUNNING, _SIM_GROUPS, _SIM_CODES

    print(f"📡 Simulation loop starting (interval={interval}s)")
    _init_sim_groups(csv_path)

    if not _SIM_GROUPS or not _SIM_CODES:
        print("⚠️ Simulation groups not ready. Aborting simulation loop.")
        _SIM_RUNNING = False
        return

    print(f"▶️ Simulation active for codes={_SIM_CODES} | interval={interval}s")

    while _SIM_RUNNING:
        for code in _SIM_CODES:
            if not _SIM_RUNNING:
                break
            try:
                sample = next(_SIM_GROUPS[code])
                name = CODE_TO_NAME.get(code, f"device_{code}")
                temp = float(sample["temperature"])
                pres = float(sample["pressure"])
                vib  = float(sample["vibration"])
                hum  = float(sample["humidity"])

                risk = compute_risk_score(temp, vib, pres, hum, code)
                upsert_and_insert_reading(name, datetime.utcnow(), temp, vib, pres, hum, risk)

            except Exception as e:
                print(f" Simulation error for code={code}: {e}")
                continue

        eventlet.sleep(interval)

    print(" Simulation stopped.")

def start_simulation():
    
    global _SIM_TASK, _SIM_RUNNING
    if _SIM_RUNNING:
        return False

    
    _init_sim_groups(TEST_CSV_PATH)

    if not _SIM_GROUPS:
        print("❌ Cannot start simulation: no groups.")
        _SIM_RUNNING = False
        return False

    _SIM_RUNNING = True
    _SIM_TASK = socketio.start_background_task(
        simulate_from_csv_triplet,
        TEST_CSV_PATH,
        SIM_INTERVAL
    )
    return True

def stop_simulation():
   
    global _SIM_RUNNING
    if not _SIM_RUNNING:
        return False
    _SIM_RUNNING = False
    return True

# ===================================================
# API Routes
# ===================================================

@app.get("/health")
def health():
    return "OK", 200

@socketio.on('connect')
def on_connect():
    print('Client connected')

@app.get("/db-test")
def db_test():
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT current_database(), inet_server_addr();")
        )
        db_name, host_ip = result.fetchone()
    return {"db_name": db_name, "host_ip": str(host_ip)}

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "equipment-monitoring",
        "socket": True,
        "shap": _SHAP_OK,
        "simulation_running": _SIM_RUNNING,
    }), 200


@app.post("/simulation/start")
def api_sim_start():
    started = start_simulation()
    msg = "Simulation started." if started else "Simulation already running or cannot start."
    return jsonify({"ok": True, "running": _SIM_RUNNING, "message": msg}), 200

@app.post("/simulation/stop")
def api_sim_stop():
    stopped = stop_simulation()
    msg = "Simulation stopped." if stopped else "Simulation already stopped."
    return jsonify({"ok": True, "running": _SIM_RUNNING, "message": msg}), 200

@app.post("/ingest")
def ingest():
    """Receive a single reading from IoT sensor."""
    try:
        data = request.get_json(force=True)
        ts = parse_timestamp(str(data.get("date", ""))) if data.get("date") else datetime.utcnow()
        name = str(data["equipment_name"]).strip()
        temperature = float(data["temperature"])
        vibration = float(data["vibration"])
        pressure = float(data["pressure"])
        humidity = float(data["humidity"])
        code = equipment_code_from_name(name)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid input: {e}"}), 400

    risk_score = compute_risk_score(temperature, vibration, pressure, humidity, code)
    saved=upsert_and_insert_reading(name, ts, temperature, vibration, pressure, humidity, risk_score)

    return jsonify({
        "ok": True,
        "db_stored":saved,
        "data": {
            "date": ts.strftime("%Y/%m/%d %H:%M:%S"),
            "equipment_name": name,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "humidity": humidity,
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
                SELECT r.temperature, r.vibration, r.pressure, r.humidity, r.timestamp,
                       p.probability, p.prediction, p.message
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
            "humidity": float(row["humidity"]),
            "timestamp": row["timestamp"].isoformat(),
            "risk_score": round(float(row["probability"]) * 100),
            "prediction": int(row["prediction"]),
            "message": row["message"],
        }
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

 
@app.get("/records")
def records():
    """Return readings + prediction info for a specific equipment."""
    name = request.args.get("equipment_name", "").strip()
    limit_str = request.args.get("limit")  # 👈 بدون قيمة افتراضية

    if not name:
        return jsonify({"ok": False, "error": "equipment_name is required"}), 400

    
    limit = None
    if limit_str is not None:
        try:
            limit = int(limit_str)
           
        except ValueError:
            limit = None  

    try:
        
        base_sql = """
            SELECT
                r.temperature,
                r.vibration,
                r.pressure,
                r.humidity,
                r.timestamp,
                p.prediction,
                p.probability,
                p.message
            FROM reading r
            JOIN prediction p ON p.reading_id = r.id
            WHERE r.equipment_name = :name
            ORDER BY r.id DESC
        """

        params = {"name": name}
        if limit is not None:
            base_sql += " LIMIT :limit"
            params["limit"] = limit

        with engine.begin() as conn:
            rows = conn.execute(text(base_sql), params).mappings().all()

        data = []
        for row in rows:
            data.append({
                "temperature": float(row["temperature"]),
                "vibration": float(row["vibration"]),
                "pressure": float(row["pressure"]),
                "humidity": float(row["humidity"]),
                "timestamp": row["timestamp"].isoformat(),
                "prediction": int(row["prediction"]),
                "probability": float(row["probability"]),
                "message": row["message"],
            })

        return jsonify({"ok": True, "data": data}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

   
      

# ===================================================
# Entry point
# ===================================================

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

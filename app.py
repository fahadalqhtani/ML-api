from flask import Flask, request, jsonify
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# ------------------ ML MODELS ------------------ #

encoder = joblib.load("encoder.pkl")
selected_features = joblib.load("selected_features.pkl")
model = joblib.load("decision_tree.pkl")

# ------------------ FIREBASE SETUP ------------------ #
import json, os
cred_data = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
cred = credentials.Certificate(cred_data)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-sensors-48dda-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app = Flask(__name__)

# ------------------ ML PROCESS FUNCTION ------------------ #
def process_sensor_data(equipment_name, temp, vib, pres, last_temp, last_vib, last_pres):
    # حساب التغيرات
    temp_change = temp - last_temp
    vib_change  = vib - last_vib
    pres_change = pres - last_pres

    temp_change_pct = (temp_change / last_temp) * 100 if last_temp != 0 else 0
    vib_change_pct  = (vib_change / last_vib) * 100 if last_vib != 0 else 0
    pres_change_pct = (pres_change / last_pres) * 100 if last_pres != 0 else 0

    eq_code = encoder.transform([equipment_name])[0]

    feature_row = {
        'equipment_code': eq_code,
        'temperature': temp,
        'vibration': vib,
        'pressure': pres,
        'temp_change': temp_change,
        'vibration_change': vib_change,
        'pressure_change': pres_change,
        'temp_change_pct': temp_change_pct,
        'vibration_change_pct': vib_change_pct,
        'pressure_change_pct': pres_change_pct
    }

    X = np.array([[feature_row[f] for f in selected_features]])
    risk = model.predict_proba(X)[0][1] * 100

    recommended_action = "None"
    status = "normal"

    if risk > 85:
        status = "warning"

        # 🔎 تحديد أهم ميزة سببت التحذير
        node_indicator = model.decision_path(X)
        feature_index = model.tree_.feature
        node_ids = node_indicator.indices
        last_split = node_ids[-2] if len(node_ids) > 1 else node_ids[0]
        dominant_feature = selected_features[feature_index[last_split]]

        mapping = {
            "temperature": "temperature",
            "temp_change": "temperature",
            "temp_change_pct": "temperature",

            "vibration": "vibration",
            "vibration_change": "vibration",
            "vibration_change_pct": "vibration",

            "pressure": "pressure",
            "pressure_change": "pressure",
            "pressure_change_pct": "pressure",
        }

        sensor_problem = "unknown"
        for key in mapping:
            if key in dominant_feature:
                sensor_problem = mapping[key]
                break

        recommended_action = f"High {sensor_problem}"

    return {
        "risk_score": float(risk),
        "status": status,
        "recommended_action": recommended_action
    }

# ------------------ API ENDPOINT ------------------ #
@app.route("/api/sensors", methods=["POST"])
def receive_data():
    try:
        # 👀 اطبع البيانات الخام
        raw_data = request.data.decode("utf-8")
        print("📥 RAW request:", raw_data)

        # حاول تحويلها JSON
        data = request.get_json(force=True, silent=True)
        print("📥 Parsed JSON:", data)

        if not data:
            return jsonify({"error": "Invalid JSON", "raw": raw_data}), 400

        equipment_name = data.get("equipment")
        temp = float(data.get("temperature", 0))
        vib = float(data.get("vibration", 0))
        pres = float(data.get("pressure", 0))

        last_temp = float(data.get("last_temperature", temp))
        last_vib = float(data.get("last_vibration", vib))
        last_pres = float(data.get("last_pressure", pres))

        # ML prediction
        result = process_sensor_data(equipment_name, temp, vib, pres, last_temp, last_vib, last_pres)

        # تحديث Firebase
        ref = db.reference(f"sensors/{equipment_name}")
        ref.update({
            "temperature": temp,
            "vibration": vib,
            "pressure": pres,
            "last_temperature": last_temp,
            "last_vibration": last_vib,
            "last_pressure": last_pres,
            **result
        })

        return jsonify({"message": "Data processed", "result": result}), 200

    except Exception as e:
     import traceback
     print("🔥 Exception in /api/sensors:", traceback.format_exc())  # 👀 اطبع التفاصيل
     return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



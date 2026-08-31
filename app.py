# app.py - BP Flask Server
from flask import Flask, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime, timezone
from threading import Lock
import os
import json
import uuid
import firebase_admin
from firebase_admin import credentials, messaging

app = Flask(__name__)

# --- Firebase Admin SDK, for sending push notifications -------------
_firebase_app = None
try:
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        cred = credentials.Certificate(json.loads(service_account_json))
        _firebase_app = firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized — push notifications enabled.")
    else:
        print("FIREBASE_SERVICE_ACCOUNT_JSON not set — push notifications disabled.")
except Exception as e:
    print(f"Firebase Admin SDK failed to initialize: {e}")

DEVICE_TOKENS = set()
_token_lock = Lock()


def send_alert_push(patient_id, status, sbp, dbp):
    if _firebase_app is None:
        return
    with _token_lock:
        tokens = list(DEVICE_TOKENS)
    if not tokens:
        return
    message_body = (
        f"ALERT: Patient {patient_id} — {status}, BP {round(sbp)}/{round(dbp)}. "
        f"Immediate attention required."
    )
    message = messaging.MulticastMessage(
        notification=messaging.Notification(title="Blood Pressure Alert", body=message_body),
        data={"patient_id": patient_id, "status": status, "sbp": str(sbp), "dbp": str(dbp)},
        tokens=tokens,
    )
    try:
        response = messaging.send_multicast(message)
        print(f"Push sent: {response.success_count} succeeded, {response.failure_count} failed")
    except Exception as e:
        print(f"Failed to send push notification: {e}")


@app.route('/register_token', methods=['POST'])
def register_token():
    data = request.get_json()
    token = data.get('token')
    if not token:
        return jsonify({"error": "token is required"}), 400
    with _token_lock:
        DEVICE_TOKENS.add(token)
    return jsonify({"status": "registered"})


# Load model and scalers
model = tf.keras.models.load_model('bp_lstm_model.keras', compile=False)
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
x_scaler   = scalers['x_scaler']
y_scaler   = scalers['y_scaler']
phy_scaler = scalers['phy_scaler']
print("Model and scalers loaded successfully.")

WINDOW_SIZE = 250

# --- Patient registry ------------------------------------------------------
PATIENTS = {}
_patients_lock = Lock()

# The one patient the ESP32's readings should currently be attributed to.
ACTIVE_PATIENT_ID = None
_active_lock = Lock()

LATEST_READINGS = {}
_lock = Lock()


def classify_bp(sbp, dbp):
    if sbp >= 180 or dbp >= 120:
        return "Hypertensive Crisis"
    elif sbp >= 140 or dbp >= 90:
        return "Stage 2"
    elif sbp >= 130 or dbp >= 80:
        return "Stage 1"
    else:
        return "Normal"


@app.route('/patients', methods=['POST'])
def register_patient():
    data = request.get_json()
    name = data.get('name')
    ward = data.get('ward', '')
    age = data.get('age')
    sex = data.get('sex')

    if not name or age is None or sex is None:
        return jsonify({"error": "name, age, and sex are required"}), 400

    patient_id = data.get('patient_id') or f"P{uuid.uuid4().hex[:6].upper()}"

    with _patients_lock:
        PATIENTS[patient_id] = {
            "patient_id": patient_id,
            "name": name,
            "ward": ward,
            "age": int(age),
            "sex": int(sex),
        }

    return jsonify(PATIENTS[patient_id]), 201


@app.route('/patients', methods=['GET'])
def list_patients():
    with _patients_lock:
        patients = list(PATIENTS.values())
    with _lock:
        readings = dict(LATEST_READINGS)

    result = []
    for p in patients:
        merged = dict(p)
        reading = readings.get(p['patient_id'])
        if reading:
            merged.update({
                "sbp": reading["sbp"],
                "dbp": reading["dbp"],
                "status": reading["status"],
                "alert": reading["alert"],
                "heart_rate": reading["heart_rate"],
                "timestamp": reading["timestamp"],
            })
        result.append(merged)
    return jsonify(result)


@app.route('/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    with _patients_lock:
        patient = PATIENTS.get(patient_id)
    if patient is None:
        return jsonify({"error": f"No patient registered with id '{patient_id}'"}), 404
    with _lock:
        reading = LATEST_READINGS.get(patient_id)
    merged = dict(patient)
    if reading:
        merged.update(reading)
    return jsonify(merged)


@app.route('/active_patient', methods=['POST'])
def set_active_patient():
    data = request.get_json()
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400
    with _patients_lock:
        if patient_id not in PATIENTS:
            return jsonify({"error": f"No patient registered with id '{patient_id}'"}), 404
    global ACTIVE_PATIENT_ID
    with _active_lock:
        ACTIVE_PATIENT_ID = patient_id
    return jsonify({"active_patient_id": patient_id})


@app.route('/active_patient', methods=['GET'])
def get_active_patient():
    with _active_lock:
        return jsonify({"active_patient_id": ACTIVE_PATIENT_ID})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    with _active_lock:
        patient_id = ACTIVE_PATIENT_ID

    if patient_id is None:
        return jsonify({"error": "No active patient selected. Nurse must select a patient in the app first."}), 409

    with _patients_lock:
        patient = PATIENTS.get(patient_id)
    if patient is None:
        return jsonify({"error": f"Active patient '{patient_id}' is no longer registered."}), 409

    ppg_ir  = np.array(data['ppg_ir'], dtype=np.float32)
    ppg_red = np.array(data['ppg_red'], dtype=np.float32)
    hr      = float(data['heart_rate'])
    age     = float(patient['age'])
    sex     = float(patient['sex'])

    ppg = np.stack([ppg_ir, ppg_red], axis=1).reshape(1, WINDOW_SIZE, 2)
    ppg_flat   = ppg.reshape(-1, 2)
    ppg_scaled = x_scaler.transform(ppg_flat).reshape(1, WINDOW_SIZE, 2)
    phy        = np.array([[age, sex, hr]], dtype=np.float32)
    phy_scaled = phy_scaler.transform(phy)

    pred_scaled = model.predict([ppg_scaled, phy_scaled], verbose=0)
    pred        = y_scaler.inverse_transform(pred_scaled)
    sbp = float(pred[0][0])
    dbp = float(pred[0][1])
    status = classify_bp(sbp, dbp)
    alert = status in ["Stage 2", "Hypertensive Crisis"]

    result = {"sbp": round(sbp, 1), "dbp": round(dbp, 1), "status": status, "alert": alert}

    with _lock:
        previous = LATEST_READINGS.get(patient_id)
        was_alerting = bool(previous and previous.get("alert"))
        LATEST_READINGS[patient_id] = {
            "patient_id": patient_id,
            "sbp": result["sbp"],
            "dbp": result["dbp"],
            "status": result["status"],
            "alert": result["alert"],
            "heart_rate": int(hr),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    if alert and not was_alerting:
        send_alert_push(patient_id, status, result["sbp"], result["dbp"])

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
# app.py - BP Flask Server
from flask import Flask, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime, timezone
from threading import Lock
import os
import json
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
        notification=messaging.Notification(
            title="Blood Pressure Alert",
            body=message_body,
        ),
        data={
            "patient_id": patient_id,
            "status": status,
            "sbp": str(sbp),
            "dbp": str(dbp),
        },
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


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"[DEBUG] Received: age={data.get('age')}, sex={data.get('sex')}, heart_rate={data.get('heart_rate')}, patient_id={data.get('patient_id')}")
    ppg_ir  = np.array(data['ppg_ir'], dtype=np.float32)
    ppg_red = np.array(data['ppg_red'], dtype=np.float32)
    age     = float(data['age'])
    sex     = float(data['sex'])
    hr      = float(data['heart_rate'])
    patient_id = str(data.get('patient_id', 'P001'))

    ppg = np.stack([ppg_ir, ppg_red], axis=1)
    ppg = ppg.reshape(1, WINDOW_SIZE, 2)
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

    result = {
        "sbp": round(sbp, 1),
        "dbp": round(dbp, 1),
        "status": status,
        "alert": alert
    }

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
            "age": int(age),
            "sex": int(sex),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    if alert and not was_alerting:
        send_alert_push(patient_id, status, result["sbp"], result["dbp"])

    return jsonify(result)


@app.route('/patients', methods=['GET'])
def list_patients():
    with _lock:
        readings = list(LATEST_READINGS.values())
    return jsonify(readings)


@app.route('/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    with _lock:
        reading = LATEST_READINGS.get(patient_id)
    if reading is None:
        return jsonify({"error": f"No readings yet for patient_id '{patient_id}'"}), 404
    return jsonify(reading)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
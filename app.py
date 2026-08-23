# app.py - BP Flask Server
# Block 1 - Imports and setup

from flask import Flask, request, jsonify
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load model and scalers
model = tf.keras.models.load_model('bp_lstm_model.keras', compile=False)

with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

x_scaler   = scalers['x_scaler']
y_scaler   = scalers['y_scaler']
phy_scaler = scalers['phy_scaler']

print("Model and scalers loaded successfully.")

# Block 2 - Predict endpoint

WINDOW_SIZE = 250

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

    ppg_ir  = np.array(data['ppg_ir'], dtype=np.float32)
    ppg_red = np.array(data['ppg_red'], dtype=np.float32)
    age     = float(data['age'])
    sex     = float(data['sex'])
    hr      = float(data['heart_rate'])

    ppg = np.stack([ppg_ir, ppg_red], axis=1)  # (250, 2)
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

    return jsonify({
        "sbp": round(sbp, 1),
        "dbp": round(dbp, 1),
        "status": status,
        "alert": status in ["Stage 2", "Hypertensive Crisis"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
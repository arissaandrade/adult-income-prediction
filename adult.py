import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, abort

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "adult.pkl"

app = Flask(__name__, template_folder="templates")

# -------------------------------
# Load trained model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing {MODEL_PATH}. Train and save the pipeline first.")

PIPE = joblib.load(MODEL_PATH)

# -------------------------------
# Feature Columns
# -------------------------------
NUMERIC = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
CATEG = ["sex", "workclass", "education", "marital_status", "occupation"]
ALL_COLS = NUMERIC + CATEG

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("adult.html")

def _num(name):
    """Helper to safely parse numeric values."""
    try:
        return float(request.form.get(name))
    except Exception:
        abort(400, description=f"Invalid numeric value for '{name}'")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and make prediction."""
    # Parse inputs
    row = {
        "age": _num("age"),
        "education_num": _num("education_num"),
        "hours_per_week": _num("hours_per_week"),
        "capital_gain": _num("capital_gain"),
        "capital_loss": _num("capital_loss"),
        "sex": (request.form.get("sex") or "").strip(),
        "workclass": (request.form.get("workclass") or "").strip(),
        "education": (request.form.get("education") or "").strip(),
        "marital_status": (request.form.get("marital_status") or "").strip(),
        "occupation": (request.form.get("occupation") or "").strip(),
    }

    if any(v == "" for k, v in row.items() if k in CATEG):
        abort(400, description="Please select all categorical fields.")

    X = pd.DataFrame([row], columns=ALL_COLS)

    # Predict
    try:
        y_pred = PIPE.predict(X)[0]
        y_prob = float(PIPE.predict_proba(X)[:, 1][0])
    except Exception as e:
        abort(500, description=f"Inference failed: {e}")

    label = ">50K" if int(y_pred) == 1 else "<=50K"

    # Return HTML result page
    return f"""
    <html><head><meta charset="utf-8"><title>Result</title></head>
    <body style="font-family:Inter,system-ui,Segoe UI,Roboto,Arial">
      <div style="max-width:700px;margin:40px auto;padding:24px;border:1px solid #D7DFEA;border-radius:12px;background:#fff">
        <h2 style="color:#0A66C2;margin-top:0">Adult Income Prediction (&gt;50K)</h2>
        <p><b>Predicted Class:</b> <code>{label}</code></p>
        <p><b>P(income &gt; 50K):</b> <code>{y_prob:.4f}</code></p>
        <hr/>
        <p><b>Inputs</b></p>
        <pre style="background:#F4F7FC;padding:12px;border-radius:8px">{row}</pre>
        <p><a href="/" style="text-decoration:none;font-weight:800;color:white;background:linear-gradient(90deg,#0A66C2,#2B81E0);padding:10px 16px;border-radius:8px">← Back</a></p>
      </div>
    </body></html>
    """

# -------------------------------
# Run the Flask app
# -------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

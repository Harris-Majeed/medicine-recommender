"""
Medicine Recommendation System - Flask backend
Loads the trained model and lookup tables saved by the notebook.
"""
import json
import os

import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# ---------- Load artifacts once at startup ----------
model = joblib.load(os.path.join(MODEL_DIR, "medicine_model.joblib"))
encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

with open(os.path.join(MODEL_DIR, "symptoms.json")) as f:
    SYMPTOMS = json.load(f)
with open(os.path.join(MODEL_DIR, "lookups.json")) as f:
    LOOKUPS = json.load(f)


def pretty(symptom: str) -> str:
    """'skin_rash' -> 'Skin Rash'"""
    return symptom.replace("_", " ").strip().title()


SYMPTOM_OPTIONS = sorted(
    [{"value": s, "label": pretty(s)} for s in SYMPTOMS],
    key=lambda x: x["label"],
)


def recommend(symptom_list, top_n=3):
    valid = [s for s in symptom_list if s in SYMPTOMS]
    if not valid:
        return None

    vec = np.zeros((1, len(SYMPTOMS)))
    for s in valid:
        vec[0, SYMPTOMS.index(s)] = 1

    proba = model.predict_proba(vec)[0]
    order = np.argsort(proba)[::-1][:top_n]
    top = [
        {"disease": encoder.classes_[i], "score": round(float(proba[i]) * 100, 1)}
        for i in order
    ]
    best = top[0]["disease"]

    return {
        "symptoms_used": [pretty(s) for s in valid],
        "top_predictions": top,
        "best": best,
        "description": LOOKUPS["description"][best],
        "precautions": LOOKUPS["precautions"][best],
        "medications": LOOKUPS["medications"][best],
        "diet": LOOKUPS["diets"][best],
        "workout": LOOKUPS["workout"][best],
    }


@app.route("/")
def index():
    return render_template("index.html", symptoms=SYMPTOM_OPTIONS, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    raw = request.form.get("symptoms", "")
    chosen = [s.strip() for s in raw.split(",") if s.strip()]

    result = recommend(chosen)
    if result is None:
        return render_template(
            "index.html",
            symptoms=SYMPTOM_OPTIONS,
            error="Please select at least one symptom.",
        )
    return render_template("result.html", r=result)


if __name__ == "__main__":
    app.run(debug=True)
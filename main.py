from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "job_placement.csv"
MODEL_PATH = BASE_DIR / "model.json"

FEATURE_ORDER = [
	"age",
	"gpa",
	"years_of_experience",
	"stream_encoded",
	"salary>52500.0",
	"college_target_enc",
]


def build_stream_encoder(dataset: pd.DataFrame) -> dict[str, int]:
	streams = sorted(dataset["stream"].dropna().astype(str).unique().tolist())
	return {name: idx for idx, name in enumerate(streams)}


def build_college_target_encoding(dataset: pd.DataFrame, smoothing: float = 10.0) -> tuple[dict[str, float], float]:
	data = dataset.copy()
	data["target"] = (data["placement_status"].astype(str).str.lower() == "placed").astype(int)

	global_mean = float(data["target"].mean())
	stats = data.groupby("college_name")["target"].agg(["mean", "count"])

	smoother = 1.0 / (1.0 + np.exp(-(stats["count"] - smoothing)))
	smoothed = smoother * stats["mean"] + (1.0 - smoother) * global_mean

	return smoothed.to_dict(), global_mean


def load_model() -> XGBClassifier:
	model = XGBClassifier()
	model.load_model(str(MODEL_PATH))
	return model


def make_features(form: dict[str, str]) -> list[float]:
	age = int(form["age"])
	gpa_out_of_10 = float(form["gpa"])
	years_of_experience = float(form["years_of_experience"])

	stream = form["stream"]
	stream_encoded = STREAM_ENCODER.get(stream, 0)

	expected_salary = float(form["expected_salary"])
	salary_binary = 1 if expected_salary >= 52500.0 else 0

	college_name = form.get("college_name", "").strip()
	use_custom_rank = form.get("use_custom_college_rank") == "on"
	custom_rank_raw = form.get("custom_college_rank", "").strip()

	if use_custom_rank or (college_name not in COLLEGE_TARGET_ENC_MAP and custom_rank_raw):
		custom_rank = float(custom_rank_raw)
		if not 0.0 <= custom_rank <= 100.0:
			raise ValueError("Custom rank must be between 0 and 100")
		college_target_enc = custom_rank / 100.0
	else:
		college_target_enc = COLLEGE_TARGET_ENC_MAP.get(college_name, GLOBAL_TARGET_MEAN)

	features = [
		age,
		gpa_out_of_10,
		years_of_experience,
		stream_encoded,
		salary_binary,
		college_target_enc,
	]
	return [float(value) for value in features]


app = Flask(__name__)

dataset_df = pd.read_csv(DATASET_PATH)
STREAM_ENCODER = build_stream_encoder(dataset_df)
COLLEGE_TARGET_ENC_MAP, GLOBAL_TARGET_MEAN = build_college_target_encoding(dataset_df)
COLLEGE_OPTIONS = sorted(dataset_df["college_name"].dropna().astype(str).unique().tolist())
STREAM_OPTIONS = sorted(STREAM_ENCODER.keys())

model = load_model()

PLACEMENT_CONFIDENCE_THRESHOLD = 75.0


@app.route("/", methods=["GET", "POST"])
def index():
	prediction = None
	probability = None
	error = None
	improvement_tips = []

	if request.method == "POST":
		try:
			features = make_features(request.form)
			feature_frame = pd.DataFrame([features], columns=FEATURE_ORDER)

			proba = float(model.predict_proba(feature_frame)[0][1])

			probability = round(proba * 100.0, 2)
			prediction = "Likely Placed" if probability >= PLACEMENT_CONFIDENCE_THRESHOLD else "Placement Risk"

			if probability < PLACEMENT_CONFIDENCE_THRESHOLD:
				improvement_tips = [
					"Build stronger job-ready skills through projects, certifications, and interview preparation.",
					"Apply consistently across multiple platforms like Naukri, LinkedIn, Indeed, Internshala, and Unstop.",
					"Improve your resume and portfolio with measurable outcomes and relevant keywords.",
					"Practice aptitude, coding rounds, and mock interviews every week.",
					"Increase networking through alumni, faculty referrals, and professional communities.",
				]
		except (KeyError, ValueError):
			error = "Please enter valid values for all fields."

	return render_template(
		"index.html",
		prediction=prediction,
		probability=probability,
		threshold=PLACEMENT_CONFIDENCE_THRESHOLD,
		improvement_tips=improvement_tips,
		error=error,
		stream_options=STREAM_OPTIONS,
		college_options=COLLEGE_OPTIONS,
	)


if __name__ == "__main__":
	app.run(debug=True)

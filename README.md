# AI/ML Placement Prediction Project

This project predicts student placement outcomes using an XGBoost classifier trained in the notebook and deployed through a Flask web application.

This project is prepared for my Vityarthi submission.

## Project Goal

Build a practical placement predictor that:
- Learns from student profile and academic signals.
- Gives a placement probability for new candidates.
- Exposes the model through a professional website for easy use.

## Project Structure

- `train.ipynb` - end-to-end data cleaning, feature engineering, model training, and model export.
- `model.json` - trained XGBoost model saved from the notebook.
- `main.py` - Flask backend that loads the model and serves predictions.
- `templates/index.html` - frontend form and prediction dashboard.
- `dataset/job_placement.csv` - source dataset.
- `requirements.txt` - Python dependencies.

## Notebook Workflow (Detailed)

The training notebook follows a clear sequence.

### 1. Load dependencies
Imported:
- pandas
- numpy
- sklearn LabelEncoder
- sklearn KFold and train_test_split
- xgboost XGBClassifier

Reason:
- pandas and numpy for data handling and numerical operations.
- sklearn utilities for encoding and train/test split.
- XGBoost for strong tabular classification performance.

### 2. Read dataset
The CSV file is loaded from `dataset/job_placement.csv` into a DataFrame.

Reason:
- Establishes the working data source for all preprocessing and modeling.

### 3. Target cleaning and conversion
`placement_status` is converted into a numeric target:
- Placed -> 1
- Not Placed -> 0

Reason:
- Classification models require numeric target labels.
- Binary target makes the objective unambiguous.

### 4. GPA scaling
I multiplied GPA by 2:
- `data['gpa'] = data['gpa'] * 2`

Reason (as I noted):
- To align GPA representation with a 10-point style interpretation used in my context.

### 5. Stream encoding
`stream` is converted using LabelEncoder into `stream_encoded`.

Reason:
- `stream` is categorical text and cannot be used directly by the model.
- Label encoding provides a compact numeric representation.

### 6. Salary feature transformation
I computed average salary and then created a binary indicator:
- Column name generated as `salary>52500.0`.
- Value is 1 if salary is greater than or equal to rounded mean salary, else 0.

Reason (as I noted):
- Simplifies salary into a decision-style signal (above/below benchmark).
- Reduces raw salary sensitivity and keeps interpretation straightforward.

### 7. College signal engineering
I engineered college-level reputation features:

#### a) Target encoding with smoothing
Function: `target_encode_with_smoothing(...)`
- Uses KFold strategy to generate out-of-fold encoded values.
- Combines college-wise placement mean with global mean using smoothing.

Reason:
- Captures placement reputation of each college.
- Smoothing prevents overconfidence on low-frequency colleges.
- Out-of-fold method reduces leakage risk during encoding.

#### b) Frequency encoding
Function: `frequency_encode(...)`
- Creates `college_freq_enc` based on occurrence ratio of each college.

Reason:
- Encodes how common a college is in dataset.
- Gives a popularity/representation signal.

#### c) Drop original text column
Dropped `college_name` after encoding.

Reason:
- Avoids feeding raw high-cardinality text directly to model.
- Keeps model input numeric and compact.

### 8. Final training feature selection
Final training subset used:
- `age`
- `gpa`
- `years_of_experience`
- `stream_encoded`
- `salary>52500.0`
- `college_target_enc`
- target: `target`

Reason:
- These columns represent academic strength, experience, domain background, salary benchmark signal, and college placement reputation.

### 9. Train/test split
Used:
- `train_test_split(..., test_size=.2)`

Reason:
- Keeps 20% held out for evaluation.
- Basic, fast validation for initial model quality check.

### 10. Model training
Model:
- `XGBClassifier()` with default settings.
- Fitted on training split.

Reason:
- XGBoost performs well on structured/tabular data.
- Good baseline before tuning.

### 11. Evaluation
Prediction done on test set and evaluated with `accuracy_score`.

Reason:
- Provides quick baseline metric to verify model learning.

### 12. Model export
Saved trained model as:
- `model.json`

Reason:
- JSON format is easy to load later for inference in Flask.

## Website Overview

The Flask website in `main.py` + `templates/index.html` provides:

- A professional UI for entering candidate inputs.
- Real-time prediction using the saved model.
- Probability output as placement confidence percentage.
- Decision threshold set to 75%.
- Labeling logic:
  - `Likely Placed` if probability >= 75
  - `Placement Risk` if probability < 75
- Improvement suggestions when below 75%.

### Input fields in website
- Age
- GPA (out of 10)
- Years of experience
- Expected salary
- Stream
- College selection

### Custom college option
If college is not listed, the website allows:
- Enabling custom college placement rank (0-100).
- This rank is converted internally to model scale (0-1) and used as `college_target_enc`.

Reason:
- Supports real users whose college may not exist in training dataset.
- Gives controlled manual fallback instead of hard failure.

## Consistency Between Training and Inference

The web app replicates notebook feature behavior:
- Uses same input feature order expected by model.
- Reconstructs stream encoding from dataset categories.
- Reconstructs college target encoding map from dataset.
- Uses same salary benchmark logic (`>= 52500` -> 1 else 0).
- Falls back to global college mean when needed.

This alignment is critical so model sees inference data in the same format it was trained on.

## Setup and Run

### 1. Create and activate environment (recommended)
Windows PowerShell example:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the Flask app

```powershell
python main.py
```

### 4. Open in browser

- http://127.0.0.1:5000

## How to Retrain and Update Model

1. Open `train.ipynb`.
2. Run all cells in order.
3. Confirm new model is saved to `model.json`.
4. Restart Flask app so it loads updated model.

## Practical Notes

- The current model is a strong baseline but can be improved with:
  - Hyperparameter tuning for XGBoost.
  - More robust validation (Stratified K-Fold, precision/recall/F1/AUC).
  - Better handling for unseen categories through explicit encoder persistence.
  - Expanded and more diverse dataset.

## Suggested Future Enhancements

- Save preprocessing artifacts (stream mapping, college encoding) as versioned files.
- Add model monitoring and confidence calibration.
- Add admin page for batch predictions via CSV upload.
- Add explainability block (feature contribution summary).

---

Built as a full ML-to-web workflow: notebook experimentation, model export, and production-style Flask inference UI.

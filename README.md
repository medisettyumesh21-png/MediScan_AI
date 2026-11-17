# 🩺 MediScan AI - Intelligent Clinical Decision Support System

> **Educational prototype - not a medical device.** MediScan AI is built on a public educational dataset,
> has **not** been clinically validated, and must **never** be used to diagnose or treat real patients.
> All outputs require review by a qualified healthcare professional.

## Project overview
MediScan AI is a Streamlit web app that demonstrates how machine learning can *support* (never replace)
healthcare professionals. It analyses the Wisconsin Diagnostic Breast Cancer dataset, estimates the probability of
each class, groups cases into **illustrative** review-priority categories, and explains which measurements influenced
each estimate.

## Problem statement
*"Develop a decision-support tool that analyzes approved medical images or clinical datasets to identify potential
signs of disease and help healthcare professionals prioritize cases for further assessment."*

## Features
| Page | What it offers |
|---|---|
| **Home Dashboard** | Record/feature counts, evaluation metric cards, class distribution, priority groups, metrics chart, ROC curve, navigation shortcuts |
| **Case Analysis** | Pick any case (or a random one), see probabilities, gauge, illustrative priority, per-case feature explanation and a human-review recommendation |
| **Dataset Explorer** | Search + filters, feature statistics, class distribution, CSV downloads, **batch CSV upload** with validation and report download |
| **AI Model Insights** | Metric table, confusion matrix, ROC curve, probability histogram, feature importance, model limitations |
| **About & Safety** | Purpose, technologies, dataset source, how the model works, privacy, responsible AI |

Also: responsive layout, interactive Plotly charts, loading spinners, empty-state messages, helpful CSV error messages.

## Technology stack
Python · Streamlit · Pandas · NumPy · scikit-learn · Plotly · GitHub · Streamlit Community Cloud

## Dataset information
* **Breast Cancer Wisconsin (Diagnostic)** - 569 records, 30 numeric features computed from digitised images of
  fine-needle-aspirate samples (creators: W. H. Wolberg, W. N. Street, O. L. Mangasarian; UCI ML Repository).
* Loaded automatically with `sklearn.datasets.load_breast_cancer`. Class labels: **0 = malignant (212), 1 = benign (357)**.
* Start-up checks: missing values, duplicates, negative values, label sanity.

## Model methodology
1. Stratified **80/20 train/test split**, fixed `random_state=42` (reproducible).
2. `Pipeline(StandardScaler -> LogisticRegression)` - the scaler is fitted on **training data only**, so there is no data leakage.
3. Evaluation on the unseen test set: accuracy, precision, recall, F1 (malignant = positive class), confusion matrix, ROC-AUC.
   A 5-fold cross-validation ROC-AUC on the training split is shown as a stability check.
4. **Explainability:** exact logistic-regression arithmetic. Each feature's contribution to the malignant log-odds is
   `-coefficient x standardised value`; the contributions plus the intercept term reproduce the model output exactly.
5. **Illustrative priority:** P(malignant) ≥ 0.80 → Priority 1, ≥ 0.40 → Priority 2, otherwise Priority 3
   (`utils/priority.py`). These thresholds are for demonstration only. **Every** category recommends human review.

## Project structure
```
MediScan_AI/
├── app.py                 # Streamlit web app (5 pages)
├── model.py               # data loading, training, evaluation, explanations
├── requirements.txt
├── README.md
├── .streamlit/config.toml # theme + upload-size limit
├── assets/logo.svg
├── data/                  # sample_upload.csv (+ notes)
├── utils/
│   ├── analysis.py        # scoring, CSV reports
│   ├── charts.py          # Plotly figures
│   ├── data_utils.py      # CSV upload validation
│   ├── priority.py        # illustrative priority rules
│   └── styles.py          # CSS + HTML helpers
└── tests/                 # unit tests + Streamlit smoke test
```

## Installation and running locally
Requires **Python 3.10+** (3.11 or 3.12 recommended).
```bash
# 1. open a terminal in the MediScan_AI folder
python -m venv .venv

# 2. activate it
#    Windows (PowerShell):  .venv\Scripts\Activate.ps1
#    Windows (cmd):         .venv\Scripts\activate.bat
#    macOS / Linux:         source .venv/bin/activate

# 3. install packages and start the app
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

### Run the tests
```bash
python -m unittest discover -s tests -v
```

## Push to GitHub
1. Create a free account at https://github.com and click **New repository** (name e.g. `MediScan_AI`, **Public**,
   do *not* tick "Add a README").
2. In a terminal inside the project folder:
```bash
git init
git add .
git commit -m "MediScan AI: clinical decision-support prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/MediScan_AI.git
git push -u origin main
```
(No Git installed? On the new repository page choose **uploading an existing file** and drag the project files in -
include the hidden `.streamlit` folder.)

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-username>/MediScan_AI` · Branch: `main` · Main file path: `app.py`.
4. (Optional) *Advanced settings* → choose Python 3.11 or 3.12.
5. Click **Deploy**. The first build takes a few minutes; you then get a public `*.streamlit.app` link.

## Privacy
* The built-in dataset is public and anonymous.
* Uploaded CSVs are processed **in memory only** and never written to disk or a database by this app.
* Files with name/ID/contact/date-of-birth style columns are rejected. **Never upload real patient information.**

## Limitations and safety disclaimer
* Small, clean, single-source public dataset; real clinical data is messier and more varied.
* No clinical validation, external validation, bias audit or regulatory approval. No real-world accuracy is claimed.
* Linear model on tabular measurements only; it does not see images, history or symptoms.
* Probabilities are estimates and may not be well calibrated; priority thresholds are illustrative.
* Low scores do not rule out disease and high scores do not confirm it.
* Records used in training look better than new cases would - use held-out test records for demonstrations.

**This software is provided for education and demonstration only. It must not be used for diagnosis, treatment or any clinical decision.**

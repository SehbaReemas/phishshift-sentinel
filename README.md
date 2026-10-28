# PhishShift Sentinel — Message Risk Scanner

A student-friendly cybersecurity + NLP + ML prototype aligned to the ML-07 project requirements.

## What it demonstrates

- Message risk classification for email/SMS text
- Near-duplicate removal before splitting
- Rules baseline
- TF-IDF + Logistic Regression (linear classifier)
- TF-IDF + Complement Naive Bayes (third model)
- Campaign-separated train/test holdout
- Reject/abstain behavior for uncertain cases
- Safe explanations of risk-cue groups
- Benchmark dashboard
- No live URL opening/fetching or attachment execution

## Project structure

```text
phishshift_sentinel/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── messages.csv
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── linear_tfidf.joblib
│   ├── complement_nb.joblib
│   ├── metrics.json
│   └── model_card.json
├── reports/
└── tests/
    └── test_project.py
```

## Run locally

### Windows PowerShell

```powershell
cd phishshift_sentinel
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, run the Python commands without activating the environment, or use:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_model.py
.venv\Scripts\python.exe -m streamlit run app.py
```

## Data provenance

The included `messages.csv` is **synthetically generated for this student prototype**. It contains 640 generated examples and provenance fields. No real credentials, mailbox dumps, or live phishing URLs are included.

For a formal submission, replace or augment it with permitted public/licensed data if your mentor/instructor requires external corpus evidence, and document the license/source.

## Important limitation

The model is a decision-support prototype. It does not prove that a message is safe or malicious.

## Suggested 3–5 minute demo

1. Open the dashboard.
2. Paste a legitimate account notification.
3. Paste an urgent verification message.
4. Change its wording/punctuation and analyze again.
5. Show the three evidence scores.
6. Demonstrate an uncertain case and the `REVIEW / ABSTAIN` outcome.
7. Open Model Benchmarks and explain the campaign-separated holdout.

## Project name

**PhishShift Sentinel**
Tagline: **"Spot the shift. Explain the risk."**

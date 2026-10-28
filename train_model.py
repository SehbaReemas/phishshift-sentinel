
from pathlib import Path
import json, re
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "messages.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def deduplicate_near_duplicates(df):
    # Deterministic normalization-based duplicate removal for the student prototype.
    x = df.copy()
    x["norm_body"] = x["body_text"].map(normalize)
    x = x.drop_duplicates(subset=["norm_body"], keep="first")
    return x.drop(columns=["norm_body"])

def split_by_campaign(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["campaign_group"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

def evaluate(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

df = pd.read_csv(DATA)
df = deduplicate_near_duplicates(df)
train, test = split_by_campaign(df)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
    max_df=0.98
)
X_train = vectorizer.fit_transform(train["body_text"].map(normalize))
X_test = vectorizer.transform(test["body_text"].map(normalize))

linear = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
nb = ComplementNB(alpha=0.5)

linear.fit(X_train, train["label"])
nb.fit(X_train, train["label"])

pred_linear = linear.predict(X_test)
pred_nb = nb.predict(X_test)

metrics = {
    "dataset_rows_after_dedup": int(len(df)),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "unique_train_campaigns": sorted(train["campaign_group"].unique().tolist()),
    "heldout_campaigns": sorted(test["campaign_group"].unique().tolist()),
    "linear_tfidf": evaluate(test["label"], pred_linear),
    "complement_nb": evaluate(test["label"], pred_nb),
}

joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")
joblib.dump(linear, MODELS / "linear_tfidf.joblib")
joblib.dump(nb, MODELS / "complement_nb.joblib")

with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(MODELS / "model_card.json", "w", encoding="utf-8") as f:
    json.dump({
        "project": "PhishShift Sentinel",
        "purpose": "Educational message-risk triage",
        "data": "Synthetic, locally generated examples with provenance fields",
        "split": "Campaign-separated GroupShuffleSplit",
        "models": ["Rules", "TF-IDF + Logistic Regression", "TF-IDF + ComplementNB"],
        "safety": [
            "No URL opening/fetching",
            "No attachments",
            "No real credentials or mailbox dumps",
            "Predictions are decision support, not guarantees"
        ]
    }, f, indent=2)

print("Training complete.")
print(json.dumps(metrics, indent=2))

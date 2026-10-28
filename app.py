
import json, re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.set_page_config(
    page_title="PhishShift Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
  --ink:#0b1220;
  --muted:#64748b;
  --line:#e2e8f0;
  --panel:#ffffff;
  --soft:#f8fafc;
}
.main .block-container {max-width:1180px; padding-top:1.2rem; padding-bottom:3rem;}
.hero {
  padding: 28px 30px; border:1px solid #dbeafe; border-radius:24px;
  background: linear-gradient(135deg,#eff6ff 0%,#f8fafc 52%,#ecfeff 100%);
  margin-bottom:18px;
}
.badge {display:inline-block;padding:6px 10px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:12px;font-weight:700;letter-spacing:.4px;}
.hero h1 {font-size:42px; margin:10px 0 6px; color:var(--ink);}
.hero p {font-size:16px; color:#475569; max-width:780px;}
.card {background:white;border:1px solid var(--line);border-radius:18px;padding:18px 20px;height:100%;}
.metric {font-size:30px;font-weight:800;color:var(--ink);}
.label {font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;}
.risk-high {border-left:6px solid #dc2626;background:#fef2f2;padding:16px;border-radius:14px;}
.risk-medium {border-left:6px solid #d97706;background:#fffbeb;padding:16px;border-radius:14px;}
.risk-low {border-left:6px solid #059669;background:#ecfdf5;padding:16px;border-radius:14px;}
.risk-unknown {border-left:6px solid #64748b;background:#f1f5f9;padding:16px;border-radius:14px;}
.small {font-size:13px;color:#64748b;}
.stButton>button {border-radius:12px;font-weight:700;}
textarea {border-radius:14px !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    vec = joblib.load(MODELS/"tfidf_vectorizer.joblib")
    linear = joblib.load(MODELS/"linear_tfidf.joblib")
    nb = joblib.load(MODELS/"complement_nb.joblib")
    metrics = json.loads((MODELS/"metrics.json").read_text(encoding="utf-8"))
    return vec, linear, nb, metrics

try:
    vectorizer, linear, nb, metrics = load_assets()
except Exception:
    st.error("Model files are missing. Run `python train_model.py` once, then restart Streamlit.")
    st.stop()

def normalize(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()

RISK_PATTERNS = {
    "Urgency / pressure": [r"\burgent\b", r"\bimmediately\b", r"\bwithout delay\b", r"\bfinal notice\b", r"\bdeadline\b", r"\btoday\b"],
    "Account restriction": [r"\bsuspended\b", r"\block(ed)?\b", r"\bdisabled\b", r"\brestricted\b"],
    "Credential / verification request": [r"\bverify\b", r"\bverification\b", r"\bconfirm your\b", r"\bidentity check\b", r"\baccount details\b"],
    "Payment / reward lure": [r"\bpayment\b", r"\breward\b", r"\bbenefit\b", r"\bbilling\b"],
    "Unfamiliar sender signal": [r"\bunknown\b", r"\buntrusted\b"]
}

def rule_score(text):
    t=normalize(text)
    hits=[]
    for name, patterns in RISK_PATTERNS.items():
        if any(re.search(p,t) for p in patterns):
            hits.append(name)
    score=min(0.95, 0.12*len(hits))
    if len(hits)>=3: score=min(0.98, score+0.25)
    return score, hits

def analyze(text):
    clean=normalize(text)
    X=vectorizer.transform([clean])
    lin=float(linear.predict_proba(X)[0,1])
    nbp=float(nb.predict_proba(X)[0,1])
    rules, hits=rule_score(clean)
    # Balanced ensemble for a student-facing triage score.
    ensemble=0.50*lin + 0.30*nbp + 0.20*rules
    # Abstain when models disagree strongly or confidence is middling.
    disagreement=abs(lin-nbp)
    abstain=(0.40 <= ensemble <= 0.60) or disagreement > 0.45
    if abstain:
        label="REVIEW / ABSTAIN"
        css="risk-unknown"
    elif ensemble >= 0.68:
        label="HIGH RISK"
        css="risk-high"
    elif ensemble >= 0.45:
        label="MEDIUM RISK"
        css="risk-medium"
    else:
        label="LOW RISK"
        css="risk-low"
    return {
        "label":label,"css":css,"score":ensemble,"linear":lin,"nb":nbp,"rules":rules,
        "hits":hits,"disagreement":disagreement
    }

with st.sidebar:
    st.markdown("## 🛡️ PhishShift")
    st.caption("Message Risk Scanner")
    st.divider()
    st.markdown("### How it works")
    st.markdown("1. Normalize message\n2. Score risk cues\n3. Compare ML models\n4. Combine evidence\n5. Abstain when uncertain")
    st.divider()
    st.markdown("### Safety")
    st.caption("This prototype never opens, fetches, or executes URLs/attachments. It is decision support, not a safety guarantee.")

st.markdown("""
<div class="hero">
  <span class="badge">CYBERSECURITY · NLP · ROBUSTNESS</span>
  <h1>PhishShift Sentinel</h1>
  <p>Analyze email and SMS wording for phishing risk, see the evidence behind the score, and test how robust the decision is to unfamiliar phrasing.</p>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_bench, tab_about = st.tabs(["🔎 Analyze Message", "📊 Model Benchmarks", "ℹ️ Project"])

with tab_scan:
    c1,c2=st.columns([1.55,1])
    with c1:
        st.markdown("### Paste a message")
        sample = st.selectbox("Quick example", [
            "Choose an example…",
            "Your account notification is ready. Review it in your usual app.",
            "Urgent: your account will be suspended unless you verify your details immediately.",
            "The support team has received your request and will reply through the normal support channel."
        ])
        default="" if sample=="Choose an example…" else sample
        message=st.text_area("Message text", value=default, height=190, placeholder="Paste the message here. Do not paste real passwords or private information.")
        analyze_btn=st.button("Analyze risk", type="primary", use_container_width=True)
    with c2:
        st.markdown("### What this scanner checks")
        for x in ["Unusual urgency", "Account restriction language", "Verification / credential cues", "Payment or reward lures", "Model disagreement"]:
            st.markdown(f"• {x}")
        st.info("Tip: test a legitimate password-reset message too. A good detector should minimize false positives.")

    if analyze_btn:
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            r=analyze(message)
            st.markdown("### Result")
            st.markdown(f"""
            <div class="{r['css']}">
              <div class="label">Risk decision</div>
              <div style="font-size:28px;font-weight:850;">{r['label']}</div>
              <div style="margin-top:6px;">Combined risk score: <b>{r['score']:.0%}</b></div>
            </div>
            """, unsafe_allow_html=True)

            m1,m2,m3,m4=st.columns(4)
            m1.metric("TF-IDF + Linear", f"{r['linear']:.0%}")
            m2.metric("Complement NB", f"{r['nb']:.0%}")
            m3.metric("Rules", f"{r['rules']:.0%}")
            m4.metric("Model gap", f"{r['disagreement']:.0%}")

            st.markdown("### Evidence")
            if r["hits"]:
                for h in r["hits"]:
                    st.markdown(f"**• {h}**")
            else:
                st.success("No predefined high-risk cue group was triggered by the rules layer.")

            if r["label"]=="REVIEW / ABSTAIN":
                st.warning("The models are uncertain or disagree materially. Treat this as a manual-review case rather than a definitive verdict.")

            st.caption("Decision support only — never use the score as proof that a message is safe or malicious.")

with tab_bench:
    st.markdown("### Campaign-separated evaluation")
    st.caption("The holdout is separated by campaign group so the test includes unseen campaign templates rather than only near-duplicates of training examples.")
    a,b,c=st.columns(3)
    a.metric("Rows after dedup", metrics["dataset_rows_after_dedup"])
    b.metric("Train rows", metrics["train_rows"])
    c.metric("Held-out rows", metrics["test_rows"])
    table=pd.DataFrame([
        ["TF-IDF + Logistic Regression", metrics["linear_tfidf"]["accuracy"], metrics["linear_tfidf"]["precision"], metrics["linear_tfidf"]["recall"], metrics["linear_tfidf"]["f1"]],
        ["TF-IDF + ComplementNB", metrics["complement_nb"]["accuracy"], metrics["complement_nb"]["precision"], metrics["complement_nb"]["recall"], metrics["complement_nb"]["f1"]],
    ], columns=["Model","Accuracy","Precision","Recall","F1"])
    st.dataframe(table.style.format({c:"{:.1%}" for c in ["Accuracy","Precision","Recall","F1"]}), use_container_width=True, hide_index=True)
    st.markdown("**Held-out campaign groups:** " + ", ".join(metrics["heldout_campaigns"]))

with tab_about:
    st.markdown("### Project goal")
    st.write("Detect suspicious messages when attackers shift wording, use unfamiliar phrasing, or change superficial message patterns.")
    st.markdown("### Required project coverage")
    st.write("Rules baseline • TF-IDF linear classifier • third model • campaign-separated holdout • reject/abstain • risk-cue explanations • false-positive checks • robustness testing.")
    st.markdown("### Safety boundaries")
    st.write("Synthetic/local training data is used in this prototype. No real credentials, mailbox dumps, attachments, or live URL fetching are required.")
    st.markdown("### Suggested demo")
    st.write("1) Run a normal legitimate notification. 2) Run an urgent verification message. 3) Modify its wording and punctuation. 4) Show the model scores and explain why an uncertain case is sent to review.")

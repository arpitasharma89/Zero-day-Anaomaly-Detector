import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from zero_day_detector import (CONFIG, FEATURES, generate_dataset, preprocess,
                               train_model, score_record, _random_record)

st.set_page_config(page_title="Zero Day Anomaly Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Zero Day Anomaly Detector")
st.caption("Isolation Forest trained only on normal traffic, so it can flag unknown attacks.")


@st.cache_resource
def load_everything():
    df = generate_dataset(CONFIG["n_normal"], CONFIG["n_attack"], CONFIG["random_state"])
    X_normal, scaler = preprocess(df[df.label == 0], fit=True)
    model = train_model(X_normal)
    return df, model, scaler


df, model, scaler = load_everything()
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📡 Live Simulation", "🔍 Check One Connection"])

# ---------------- Tab 1 ----------------
with tab1:
    X_all, _ = preprocess(df, scaler=scaler)
    y_true = df["label"].values
    y_pred = (model.predict(X_all) == -1).astype(int)
    scores = model.decision_function(X_all)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(df))
    c2.metric("Attack recall", f"{(y_pred[y_true == 1] == 1).mean():.0%}")
    c3.metric("ROC-AUC", f"{roc_auc_score(y_true, -scores):.3f}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(range(len(scores)), scores,
                   c=["red" if l else "steelblue" for l in y_true], alpha=0.35, s=8)
        ax.axhline(0, color="black", ls="--")
        ax.set_title("Anomaly scores (red = attack)"); ax.set_xlabel("Record"); ax.set_ylabel("Score")
        st.pyplot(fig)

    st.text(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))

# ---------------- Tab 2 ----------------
with tab2:
    n = st.slider("Number of connections", 10, 200, 40)
    p = st.slider("Attack probability", 0.0, 1.0, 0.3)
    if st.button("▶ Run simulation"):
        rng = np.random.default_rng()
        rows = []
        for _ in range(n):
            rec = _random_record(rng, rng.random() < p)
            res = score_record(model, scaler, rec)
            rows.append({"time": res["timestamp"], "level": res["alert_level"],
                         "score": res["score"], "bytes_sent": round(rec["bytes_sent"]),
                         "failed_logins": rec["failed_logins"],
                         "error_rate": round(rec["error_rate"], 2)})
        out = pd.DataFrame(rows)
        counts = out["level"].value_counts()
        m = st.columns(4)
        for col, lvl in zip(m, ["HIGH", "MEDIUM", "LOW", "NONE"]):
            col.metric(lvl if lvl != "NONE" else "NORMAL", int(counts.get(lvl, 0)))
        st.dataframe(out.style.apply(
            lambda r: ["background-color:#ffcccc" if r["level"] == "HIGH" else
                       "background-color:#fff0b3" if r["level"] in ("MEDIUM", "LOW") else ""] * len(r),
            axis=1), width="stretch")

# ---------------- Tab 3 ----------------
with tab3:
    st.write("Enter connection details. Defaults are a normal connection.")
    a, b = st.columns(2)
    rec = {
        "duration": a.number_input("duration (s)", 0.0, 1000.0, 30.0),
        "bytes_sent": a.number_input("bytes_sent", 0.0, 1e7, 5000.0),
        "bytes_received": a.number_input("bytes_received", 0.0, 1e7, 8000.0),
        "packets_sent": a.number_input("packets_sent", 0.0, 1e5, 50.0),
        "packets_received": a.number_input("packets_received", 0.0, 1e5, 80.0),
        "failed_logins": b.number_input("failed_logins", 0, 100, 0),
        "src_port": b.number_input("src_port", 1, 65535, 40000),
        "protocol_type": b.selectbox("protocol_type", [0, 1, 2],
                                     format_func=lambda x: ["TCP", "UDP", "ICMP"][x]),
        "connection_count": b.number_input("connection_count", 0.0, 10000.0, 10.0),
        "error_rate": b.slider("error_rate", 0.0, 1.0, 0.02),
    }
    if st.button("Analyse"):
        res = score_record(model, scaler, rec)
        msg = f"{res['alert_msg']}  (score {res['score']})"
        {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.warning, "NONE": st.success}[res["alert_level"]](msg)

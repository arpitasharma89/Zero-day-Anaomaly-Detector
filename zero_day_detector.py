# ==============================================================
#  Zero Day Anomaly Detector  -  Full Implementation
#  Language : Python 3.x
#  Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
# ==============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")          # save plots to files (no pop-up windows needed)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, datetime, joblib, os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# MODULE 1 - CONFIGURATION
# ──────────────────────────────────────────────────────────────

CONFIG = {
    "n_normal"         : 2000,   # normal training samples
    "n_attack"         : 200,    # labelled attack samples (eval only)
    "n_estimators"     : 200,    # Isolation Forest trees
    "contamination"    : 0.05,   # expected anomaly fraction
    "random_state"     : 42,
    "model_path"       : "zd_model.pkl",
    "scaler_path"      : "zd_scaler.pkl",
    "alert_thresholds" : {       # anomaly score cut-offs
        "HIGH"  : -0.15,
        "MEDIUM": -0.05,
    },
}

FEATURES = [
    "duration",           # session length (seconds)
    "bytes_sent",         # bytes client -> server
    "bytes_received",     # bytes server -> client
    "packets_sent",       # packet count client -> server
    "packets_received",   # packet count server -> client
    "failed_logins",      # consecutive auth failures
    "src_port",           # source TCP/UDP port number
    "protocol_type",      # 0=TCP, 1=UDP, 2=ICMP
    "connection_count",   # recent connections to same host
    "error_rate",         # fraction of connections with errors
]

ALERT_MESSAGES = {
    "HIGH"  : "*** HIGH ALERT ***  POSSIBLE ZERO-DAY ATTACK DETECTED",
    "MEDIUM": "*** MEDIUM ALERT *** SUSPICIOUS ACTIVITY DETECTED",
    "LOW"   : "*** LOW ALERT ***  MINOR ANOMALY - MONITOR CLOSELY",
    "NONE"  : "Status: NORMAL",
}


# ──────────────────────────────────────────────────────────────
# MODULE 2 - DATA GENERATION (synthetic traffic)
# ──────────────────────────────────────────────────────────────

def generate_dataset(n_normal=2000, n_attack=200, seed=42):
    """Return a labelled DataFrame (label 0=normal, 1=attack)."""
    rng = np.random.default_rng(seed)

    normal = pd.DataFrame({
        "duration"        : rng.normal(30,   10,  n_normal).clip(1),
        "bytes_sent"      : rng.normal(5000, 1000, n_normal).clip(100),
        "bytes_received"  : rng.normal(8000, 2000, n_normal).clip(100),
        "packets_sent"    : rng.normal(50,   15,  n_normal).clip(1),
        "packets_received": rng.normal(80,   20,  n_normal).clip(1),
        "failed_logins"   : rng.poisson(0.2,       n_normal),
        "src_port"        : rng.integers(1024, 65535, n_normal),
        "protocol_type"   : rng.choice([0, 1, 2], n_normal, p=[0.7, 0.2, 0.1]),
        "connection_count": rng.normal(10,   5,   n_normal).clip(1),
        "error_rate"      : rng.uniform(0,   0.05, n_normal),
        "label"           : 0,
    })

    attack = pd.DataFrame({
        "duration"        : rng.normal(2,    1,    n_attack).clip(0.1),
        "bytes_sent"      : rng.normal(60000, 15000, n_attack).clip(1000),
        "bytes_received"  : rng.normal(100,  50,   n_attack).clip(10),
        "packets_sent"    : rng.normal(600,  100,  n_attack).clip(10),
        "packets_received": rng.normal(8,    3,    n_attack).clip(1),
        "failed_logins"   : rng.poisson(8,          n_attack),
        "src_port"        : rng.integers(1, 1023,  n_attack),
        "protocol_type"   : rng.choice([0, 1, 2], n_attack, p=[0.4, 0.3, 0.3]),
        "connection_count": rng.normal(200,  50,   n_attack).clip(50),
        "error_rate"      : rng.uniform(0.5, 1.0,  n_attack),
        "label"           : 1,
    })

    df = (pd.concat([normal, attack], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True))
    return df


# ──────────────────────────────────────────────────────────────
# MODULE 3 - PRE-PROCESSING
# ──────────────────────────────────────────────────────────────

def preprocess(df, scaler=None, fit=False):
    """Impute, clip and z-score normalise the FEATURES columns."""
    X = df[FEATURES].copy()
    X = X.fillna(X.median())
    X = X.clip(lower=0)

    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


# ──────────────────────────────────────────────────────────────
# MODULE 4 - MODEL TRAINING (Isolation Forest)
# ──────────────────────────────────────────────────────────────

def train_model(X_normal_scaled, cfg=CONFIG):
    model = IsolationForest(
        n_estimators  = cfg["n_estimators"],
        contamination = cfg["contamination"],
        max_samples   = "auto",
        random_state  = cfg["random_state"],
        n_jobs        = -1,
    )
    model.fit(X_normal_scaled)
    print(f"[OK] Model trained  |  trees={cfg['n_estimators']}  "
          f"contamination={cfg['contamination']}")
    return model


def save_artifacts(model, scaler, cfg=CONFIG):
    joblib.dump(model,  cfg["model_path"])
    joblib.dump(scaler, cfg["scaler_path"])
    print(f"[OK] Saved model  -> {cfg['model_path']}")
    print(f"[OK] Saved scaler -> {cfg['scaler_path']}")


def load_artifacts(cfg=CONFIG):
    model  = joblib.load(cfg["model_path"])
    scaler = joblib.load(cfg["scaler_path"])
    print("[OK] Loaded pre-trained model and scaler.")
    return model, scaler


# ──────────────────────────────────────────────────────────────
# MODULE 5 - ANOMALY SCORING & ALERT GENERATION
# ──────────────────────────────────────────────────────────────

def score_record(model, scaler, record: dict) -> dict:
    X = pd.DataFrame([record])[FEATURES]
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)[0]            # 1=normal, -1=anomaly
    score = model.decision_function(X_scaled)[0]  # lower = more anomalous

    thresholds = CONFIG["alert_thresholds"]

    if pred == -1:
        if   score < thresholds["HIGH"]:   level = "HIGH"
        elif score < thresholds["MEDIUM"]: level = "MEDIUM"
        else:                              level = "LOW"
        is_anomaly = True
    else:
        level      = "NONE"
        is_anomaly = False

    return {
        "timestamp"  : datetime.datetime.now().isoformat(timespec="seconds"),
        "prediction" : "ANOMALY" if is_anomaly else "NORMAL",
        "score"      : round(float(score), 4),
        "alert_level": level,
        "alert_msg"  : ALERT_MESSAGES[level],
    }


def print_result(res):
    ts, msg, sc = res["timestamp"], res["alert_msg"], res["score"]
    if res["prediction"] == "ANOMALY":
        print(f"[{ts}]  {msg}  |  anomaly_score={sc}")
    else:
        print(f"[{ts}]  {msg}  |  score={sc}")


# ──────────────────────────────────────────────────────────────
# MODULE 6 - REAL-TIME MONITORING SIMULATION
# ──────────────────────────────────────────────────────────────

def _random_record(rng, is_attack: bool) -> dict:
    if is_attack:
        r = {
            "duration"        : float(rng.normal(2, 1)),
            "bytes_sent"      : float(rng.normal(65000, 12000)),
            "bytes_received"  : float(rng.normal(75, 25)),
            "packets_sent"    : float(rng.normal(650, 90)),
            "packets_received": float(rng.normal(7, 2)),
            "failed_logins"   : int(rng.poisson(10)),
            "src_port"        : int(rng.integers(1, 1023)),
            "protocol_type"   : int(rng.choice([0, 1, 2])),
            "connection_count": float(rng.normal(260, 45)),
            "error_rate"      : float(rng.uniform(0.65, 1.0)),
        }
    else:
        r = {
            "duration"        : float(rng.normal(30, 10)),
            "bytes_sent"      : float(rng.normal(5000, 800)),
            "bytes_received"  : float(rng.normal(8000, 1500)),
            "packets_sent"    : float(rng.normal(50, 12)),
            "packets_received": float(rng.normal(80, 18)),
            "failed_logins"   : int(rng.poisson(0.2)),
            "src_port"        : int(rng.integers(1024, 65535)),
            "protocol_type"   : int(rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])),
            "connection_count": float(rng.normal(10, 4)),
            "error_rate"      : float(rng.uniform(0, 0.04)),
        }
    r["duration"]         = max(0.1, r["duration"])
    r["bytes_sent"]       = max(100,  r["bytes_sent"])
    r["bytes_received"]   = max(10,   r["bytes_received"])
    r["packets_sent"]     = max(1,    r["packets_sent"])
    r["packets_received"] = max(1,    r["packets_received"])
    r["failed_logins"]    = max(0,    r["failed_logins"])
    r["connection_count"] = max(1,    r["connection_count"])
    r["error_rate"]       = min(1.0, max(0.0, r["error_rate"]))
    return r


def simulate_realtime(model, scaler, n_records=25, attack_prob=0.30, seed=99):
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 70)
    print("   REAL-TIME NETWORK MONITORING  -  Zero Day Detection System")
    print("=" * 70)
    alerts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}

    for _ in range(n_records):
        is_attack = rng.random() < attack_prob
        record    = _random_record(rng, is_attack)
        result    = score_record(model, scaler, record)
        print_result(result)
        alerts[result["alert_level"]] += 1
        time.sleep(0.04)

    print("=" * 70)
    print(f"  Summary  |  HIGH={alerts['HIGH']}  MEDIUM={alerts['MEDIUM']}"
          f"  LOW={alerts['LOW']}  NORMAL={alerts['NONE']}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# MODULE 7 - EVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate(model, scaler, df):
    X_all, _ = preprocess(df, scaler=scaler, fit=False)
    raw_preds = model.predict(X_all)
    y_pred = [1 if p == -1 else 0 for p in raw_preds]
    y_true = df["label"].tolist()

    scores = -model.decision_function(X_all)
    auc    = roc_auc_score(y_true, scores)

    print("\n--- Classification Report -------------------------------------")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    print(f"  ROC-AUC Score : {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix - Zero Day Anomaly Detector")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  [OK] Confusion matrix -> confusion_matrix.png")


def plot_score_distribution(model, scaler, df):
    X_all, _ = preprocess(df, scaler=scaler, fit=False)
    scores = model.decision_function(X_all)
    labels = df["label"].values

    plt.figure(figsize=(11, 4))
    colors = ["red" if l == 1 else "steelblue" for l in labels]
    plt.scatter(range(len(scores)), scores, c=colors, alpha=0.35, s=8)
    plt.axhline(0, color="black", linewidth=1.2, linestyle="--")
    plt.title("Anomaly Score Distribution - Network Connection Records")
    plt.xlabel("Record Index"); plt.ylabel("Anomaly Score")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=8, label='Normal'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Attack'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Decision boundary'),
    ])
    plt.tight_layout()
    plt.savefig("anomaly_score_distribution.png", dpi=150)
    plt.close()
    print("  [OK] Score plot -> anomaly_score_distribution.png")


def plot_feature_importance(model, scaler, df):
    X_all, _ = preprocess(df, scaler=scaler, fit=False)
    base_scores = model.decision_function(X_all)
    importance = {}
    rng = np.random.default_rng(0)

    for i, feat in enumerate(FEATURES):
        X_perm = X_all.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_scores = model.decision_function(X_perm)
        importance[feat] = float(np.abs(base_scores - perm_scores).mean())

    feat_df = (pd.Series(importance).sort_values(ascending=True).reset_index())
    feat_df.columns = ["feature", "importance"]

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    plt.title("Approximate Feature Importance (Permutation)")
    plt.xlabel("Mean Absolute Score Change")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("  [OK] Feature importance -> feature_importance.png")


# ──────────────────────────────────────────────────────────────
# MODULE 8 - MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n==========================================================")
    print("   Zero Day Anomaly Detector - AI Project")
    print("==========================================================\n")

    print("[1] Generating synthetic network traffic dataset ...")
    df = generate_dataset(CONFIG["n_normal"], CONFIG["n_attack"], CONFIG["random_state"])
    print(f"    Total records : {len(df)}")
    print(f"    Normal        : {(df.label == 0).sum()}")
    print(f"    Attack        : {(df.label == 1).sum()}")
    print("\n    Sample feature statistics (normal traffic):")
    print(df[df.label == 0][FEATURES].describe().round(2).to_string())

    print("\n[2] Pre-processing (imputation + z-score normalisation) ...")
    df_normal = df[df["label"] == 0].copy()
    X_normal, scaler = preprocess(df_normal, fit=True)
    print(f"    Fitted scaler on {len(df_normal)} normal records.")

    print("\n[3] Training Isolation Forest model ...")
    model = train_model(X_normal)

    print("\n[4] Saving model and scaler ...")
    save_artifacts(model, scaler)

    print("\n[5] Evaluating on full labelled dataset ...")
    evaluate(model, scaler, df)

    print("\n[6] Generating visualisation plots ...")
    plot_score_distribution(model, scaler, df)
    plot_feature_importance(model, scaler, df)

    print("\n[7] Launching real-time monitoring simulation ...")
    simulate_realtime(model, scaler, n_records=25, attack_prob=0.30)

    print("\n[Done] All steps completed successfully.")
    print("       Output files: confusion_matrix.png, "
          "anomaly_score_distribution.png, feature_importance.png")

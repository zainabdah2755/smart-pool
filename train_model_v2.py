"""
Smart Pool AI - Advanced Training Pipeline v2
==============================================
Trains multiple specialized models:
  1. Pool Status Classifier (Normal / Warning / Danger)
  2. Water Health Scorer (0-100 regression)
  3. Algae Growth Risk Predictor (0-100 risk score)
  4. Predictive Maintenance Classifier
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error
import joblib
import json
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for pool monitoring.
    Covers normal, warning, and danger scenarios based on
    domain knowledge of pool water chemistry.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        # ── Sensor readings ──────────────────────────────────────
        temp_eau   = rng.uniform(18, 40)      # °C
        temp_air   = rng.uniform(15, 45)      # °C
        humidite   = rng.uniform(30, 95)      # %
        ph         = rng.uniform(6.0, 9.0)    # pH
        luminosite = rng.uniform(0, 1023)     # LDR raw (0-1023)
        ir         = rng.uniform(0, 1)        # turbidity (0=clear, 1=opaque)

        # ── Derived / engineered features ────────────────────────
        temp_delta      = temp_eau - temp_air
        heat_index      = temp_air + 0.33 * (humidite / 100 * 6.105 *
                          np.exp(17.27 * temp_air / (237.7 + temp_air))) - 4
        ph_deviation    = abs(ph - 7.4)           # ideal pH 7.4
        clarity_score   = max(0, 1 - ir)          # 1=crystal clear
        uv_index_est    = luminosite / 1023 * 11  # 0-11 UV index estimate

        # ── Algae risk score (domain formula) ────────────────────
        algae_risk = (
            max(0, (temp_eau - 24) * 3)        # warm water promotes algae
            + max(0, (ph - 7.8) * 25)          # high pH favors algae
            + max(0, ir * 40)                  # turbidity = existing growth
            + max(0, (humidite - 70) * 0.5)    # high humidity
            + max(0, uv_index_est * 2)         # UV drives photosynthesis
        )
        algae_risk = float(np.clip(algae_risk, 0, 100))

        # ── Water health score (0-100) ────────────────────────────
        health_score = 100
        health_score -= ph_deviation * 15
        health_score -= ir * 30
        health_score -= max(0, (temp_eau - 32) * 3)
        health_score -= max(0, (28 - temp_eau) * 2)
        health_score -= algae_risk * 0.2
        health_score = float(np.clip(health_score + rng.normal(0, 3), 0, 100))

        # ── Status label ─────────────────────────────────────────
        if ph < 6.8 or ph > 8.5 or temp_eau > 35 or ir > 0.8:
            status = "Danger"
        elif ph_deviation > 0.5 or ir > 0.4 or temp_eau > 32 or algae_risk > 60:
            status = "Warning"
        else:
            status = "Normal"

        # ── Maintenance label ─────────────────────────────────────
        # Flags if pool needs immediate attention
        needs_maintenance = int(
            ir > 0.5 or ph < 7.0 or ph > 8.0
            or temp_eau > 34 or algae_risk > 70
        )

        rows.append({
            "temp_eau":         round(temp_eau, 2),
            "temp_air":         round(temp_air, 2),
            "humidite":         round(humidite, 2),
            "ph":               round(ph, 2),
            "luminosite":       round(luminosite, 2),
            "ir":               round(ir, 3),
            # engineered
            "temp_delta":       round(temp_delta, 2),
            "heat_index":       round(heat_index, 2),
            "ph_deviation":     round(ph_deviation, 3),
            "clarity_score":    round(clarity_score, 3),
            "uv_index_est":     round(uv_index_est, 2),
            # targets
            "algae_risk":       round(algae_risk, 2),
            "health_score":     round(health_score, 2),
            "status":           status,
            "needs_maintenance": needs_maintenance,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING HELPER
# ─────────────────────────────────────────────

BASE_FEATURES = [
    "temp_eau", "temp_air", "humidite", "ph",
    "luminosite", "ir",
    "temp_delta", "heat_index", "ph_deviation",
    "clarity_score", "uv_index_est",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a dataframe that may only have raw sensor cols."""
    df = df.copy()
    if "temp_delta" not in df.columns:
        df["temp_delta"] = df["temp_eau"] - df["temp_air"]
    if "heat_index" not in df.columns:
        df["heat_index"] = (
            df["temp_air"]
            + 0.33 * (df["humidite"] / 100 * 6.105
                      * np.exp(17.27 * df["temp_air"] / (237.7 + df["temp_air"])))
            - 4
        )
    if "ph_deviation" not in df.columns:
        df["ph_deviation"] = (df["ph"] - 7.4).abs()
    if "clarity_score" not in df.columns:
        df["clarity_score"] = (1 - df["ir"]).clip(0, 1)
    if "uv_index_est" not in df.columns:
        df["uv_index_est"] = df["luminosite"] / 1023 * 11
    return df


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train_all_models(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[BASE_FEATURES]
    results = {}

    # ── 3a. Status Classifier ─────────────────────────────────────
    le = LabelEncoder()
    y_status = le.fit_transform(df["status"])   # Danger=0, Normal=1, Warning=2
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_status, test_size=0.2, random_state=42)

    clf_status = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42
    )
    clf_status.fit(X_tr, y_tr)
    preds = clf_status.predict(X_te)
    print("\n=== STATUS CLASSIFIER ===")
    print(classification_report(y_te, preds, target_names=le.classes_))
    results["status_model"] = clf_status
    results["status_encoder"] = le

    # ── 3b. Health Score Regressor ────────────────────────────────
    y_health = df["health_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_health, test_size=0.2, random_state=42)

    reg_health = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
    )
    reg_health.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, reg_health.predict(X_te))
    print(f"\n=== HEALTH SCORE REGRESSOR ===  MAE = {mae:.2f} pts")
    results["health_model"] = reg_health

    # ── 3c. Algae Risk Regressor ──────────────────────────────────
    y_algae = df["algae_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_algae, test_size=0.2, random_state=42)

    reg_algae = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
    )
    reg_algae.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, reg_algae.predict(X_te))
    print(f"\n=== ALGAE RISK REGRESSOR ===  MAE = {mae:.2f} pts")
    results["algae_model"] = reg_algae

    # ── 3d. Maintenance Classifier ────────────────────────────────
    y_maint = df["needs_maintenance"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_maint, test_size=0.2, random_state=42)

    clf_maint = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    )
    clf_maint.fit(X_tr, y_tr)
    preds = clf_maint.predict(X_te)
    print("\n=== MAINTENANCE CLASSIFIER ===")
    print(classification_report(y_te, preds, target_names=["OK", "Needs Maintenance"]))
    results["maintenance_model"] = clf_maint

    # ── 3e. Feature importance ────────────────────────────────────
    importances = dict(zip(BASE_FEATURES,
                           clf_status.feature_importances_.round(4).tolist()))
    print("\n=== FEATURE IMPORTANCES (Status model) ===")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:<20} {imp:.4f}")
    results["feature_importances"] = importances
    results["feature_names"] = BASE_FEATURES

    return results


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic dataset…")
    df = generate_dataset(n_samples=5000)
    df.to_csv("smartpool_v2_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df.columns.tolist()}")
    print(df["status"].value_counts())

    print("\nTraining models…")
    models = train_all_models(df)

    # Save each model separately for clean API loading
    joblib.dump(models["status_model"],      "model_status.pkl")
    joblib.dump(models["status_encoder"],    "model_status_encoder.pkl")
    joblib.dump(models["health_model"],      "model_health.pkl")
    joblib.dump(models["algae_model"],       "model_algae.pkl")
    joblib.dump(models["maintenance_model"], "model_maintenance.pkl")

    # Save metadata
    meta = {
        "feature_names": models["feature_names"],
        "feature_importances": models["feature_importances"],
        "status_classes": models["status_encoder"].classes_.tolist(),
        "version": "2.0",
    }
    with open("model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ All models saved:")
    print("  model_status.pkl")
    print("  model_status_encoder.pkl")
    print("  model_health.pkl")
    print("  model_algae.pkl")
    print("  model_maintenance.pkl")
    print("  model_metadata.json")

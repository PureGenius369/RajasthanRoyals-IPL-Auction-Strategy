#!/usr/bin/env python3
"""
rr2_model_selection_calibrated.py

Calibrated model to predict "capped vs uncapped" from the engineered features,
then estimate cap probability for today's uncapped players and export a
Top-40 list.

Inputs:
    output/player_features_with_phase.parquet
    players.csv

Outputs (in ./output):
    rr2_player_cap_probs_full_calibrated.csv
    rr2_top40_uncapped_probs_calibrated.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ----------------- 1. LOAD DATA -----------------
print("[1] Loading feature table and players...")
features_path = OUT_DIR / "player_features_with_phase.parquet"
df = pd.read_parquet(features_path)

players = pd.read_csv("players.csv")[["player_id", "player_name"]]
players["player_id"] = players["player_id"].astype(str)

df["player_id"] = df["player_id"].astype(str)
df = df.merge(players, on="player_id", how="left")

# ----------------- 2. TARGET -----------------
target_col = "career_is_capped"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in feature table.")

y = df[target_col].astype(int)
print(f"[info] Total players: {len(df)}, capped: {y.sum()}, uncapped: {(1-y).sum()}")

# ----------------- 3. FEATURE MATRIX X -----------------
print("[2] Building feature matrix...")

drop_cols = [
    "player_id",
    "player_name",
    "career_is_capped",
    "career_is_uncapped",
]
# drop any other *_is_capped / *_is_uncapped flags
drop_cols += [c for c in df.columns if "is_capped" in c.lower() or "is_uncapped" in c.lower()]

cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ["player_name"]]
num_cols = [c for c in df.columns
            if c not in drop_cols + cat_cols
            and pd.api.types.is_numeric_dtype(df[c])]

X_num = df[num_cols].copy().fillna(0.0)
if cat_cols:
    X_cat = pd.get_dummies(df[cat_cols].fillna("Unknown"), drop_first=True)
else:
    X_cat = pd.DataFrame(index=df.index)

X = pd.concat([X_num, X_cat], axis=1)

print(f"[info] X shape: {X.shape[0]} rows × {X.shape[1]} cols")
print(f"[info] Numeric feats: {len(num_cols)}, categorical (one-hot): {len(X_cat.columns)}")

# ----------------- 4. TRAIN/TEST SPLIT & CALIBRATED MODEL -----------------
print("[3] Train/test split + calibrated RF training...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

base_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Calibrated classifier (isotonic calibration for better probability shape)
cal_clf = CalibratedClassifierCV(
    estimator=base_clf,
    cv=3,                # 3-fold CV on the training set
    method="isotonic"
)

cal_clf.fit(X_train, y_train)

# Evaluate on held-out test data
y_prob_test = cal_clf.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob_test)
acc = accuracy_score(y_test, y_pred_test)

print(f"[metrics] Test AUC (calibrated): {auc:.3f}")
print(f"[metrics] Test ACC (calibrated): {acc:.3f}")
print("[metrics] Classification report (test):")
print(classification_report(y_test, y_pred_test))

# ----------------- 5. REFIT CALIBRATED MODEL ON FULL DATA -----------------
print("[4] Fitting calibrated model on full dataset...")

# We fit a base model on full data, then calibrate with cv='prefit'
base_full = RandomForestClassifier(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
base_full.fit(X, y)

cal_full = CalibratedClassifierCV(
    estimator=base_full,
    cv="prefit",
    method="isotonic"
)
cal_full.fit(X, y)  # calibration splits X,y internally

# feature importance (from base RF, just for interpretation)
importances = base_full.feature_importances_
fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\n[top 20 features by importance (base RF)]:")
print(fi.head(20))

# ----------------- 6. PREDICT CALIBRATED PROBABILITIES -----------------
print("[5] Predicting calibrated probabilities for all players...")
df["cap_prob_calibrated"] = cal_full.predict_proba(X)[:, 1]

# ----------------- 7. AGE-BASED DECAY -----------------
print("[6] Applying age-based decay...")

age_col = "career_age_years"
if age_col in df.columns:
    age = df[age_col].fillna(df[age_col].median())
else:
    # if age isn't available, no decay
    age = pd.Series(28.0, index=df.index)

# piecewise decay:
# <= 32 → 1.0
# 33–35 → 0.85
# > 35  → 0.70
decay_factor = np.where(
    age <= 32, 1.0,
    np.where(age <= 35, 0.85, 0.70)
)

df["cap_prob_calibrated_age"] = df["cap_prob_calibrated"] * decay_factor

print("[info] Probability range before age decay:",
      float(df["cap_prob_calibrated"].min()),
      "to",
      float(df["cap_prob_calibrated"].max()))
print("[info] Probability range after age decay:",
      float(df["cap_prob_calibrated_age"].min()),
      "to",
      float(df["cap_prob_calibrated_age"].max()))

# ----------------- 8. SAVE FULL TABLE -----------------
print("[7] Saving full calibrated probability table...")

full_cols = [
    "player_id", "player_name",
    target_col,
    "cap_prob_calibrated",
    "cap_prob_calibrated_age",
]
for c in [
    "career_role_guess",
    "career_batting_runs", "career_bat_avg", "career_bat_sr",
    "career_bowling_wickets", "career_bow_econ",
    "venue_fit_score",
    "runs_slope", "wk_slope",
    "career_matches_played",
    "career_age_years"
]:
    if c in df.columns and c not in full_cols:
        full_cols.append(c)

full_cols = [c for c in full_cols if c in df.columns]
df[full_cols].to_csv(OUT_DIR / "rr2_player_cap_probs_full_calibrated.csv", index=False)
print(f"[saved] rr2_player_cap_probs_full_calibrated.csv")

# ----------------- 9. TOP-40 UNCAPPED (AGE-ADJUSTED) -----------------
print("[8] Building Top-40 uncapped list (age-adjusted prob)...")

uncapped_mask = df[target_col] == 0
df_uncapped = df[uncapped_mask].copy()

df_uncapped = df_uncapped.sort_values("cap_prob_calibrated_age", ascending=False)

top40 = df_uncapped.head(40)

top_cols = [
    "player_id", "player_name",
    "cap_prob_calibrated", "cap_prob_calibrated_age",
    "career_role_guess",
    "career_batting_runs", "career_bat_avg", "career_bat_sr",
    "career_bowling_wickets", "career_bow_econ",
    "venue_fit_score",
    "runs_slope", "wk_slope",
    "career_matches_played",
    "career_age_years",
]
top_cols = [c for c in top_cols if c in top40.columns]

top40[top_cols].to_csv(OUT_DIR / "rr2_top40_uncapped_probs_calibrated.csv", index=False)
print(f"[saved] rr2_top40_uncapped_probs_calibrated.csv")

print("\n[Top-20 preview]:")
print(top40[top_cols].head(20))
print("\nDone.")

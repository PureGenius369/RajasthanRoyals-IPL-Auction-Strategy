#!/usr/bin/env python3
"""
rr2_model_selection.py

Train a model to predict "capped vs uncapped" from the engineered features,
then use it to estimate cap probability for today's uncapped players and
export a Top-40 list.

Inputs:
    output/player_features_with_phase.parquet
    players.csv   (for player_name)

Outputs (to ./output):
    rr2_player_cap_probs_full.csv       -- all players with probabilities
    rr2_top40_uncapped_probs.csv       -- final Top-40 uncapped list
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ----------------- 1. LOAD DATA -----------------
print("[1] Loading feature table and players...")
features_path = OUT_DIR / "player_features_with_phase.parquet"
df = pd.read_parquet(features_path)

players = pd.read_csv("players.csv")[["player_id", "player_name"]]
players["player_id"] = players["player_id"].astype(str)

# ensure player_id type
df["player_id"] = df["player_id"].astype(str)

# attach names for final outputs (NOT used in training features)
df = df.merge(players, on="player_id", how="left")

# ----------------- 2. DEFINE TARGET -----------------
target_col = "career_is_capped"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in feature table.")

y = df[target_col].astype(int)

print(f"[info] Total players: {len(df)}, capped positives: {y.sum()}, uncapped negatives: {(1-y).sum()}")

# ----------------- 3. BUILD FEATURE MATRIX X -----------------
print("[2] Building feature matrix...")

# Columns we must NOT feed into the model (identifiers / direct labels)
drop_cols = [
    "player_id",
    "player_name",
    "career_is_capped",
    "career_is_uncapped",
    # if any similar flags exist in other windows, they will be dropped via pattern:
]
drop_cols += [c for c in df.columns if "is_capped" in c.lower() or "is_uncapped" in c.lower()]

# separate categorical (object) and numeric
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ["player_name"]]
num_cols = [c for c in df.columns
            if c not in drop_cols + cat_cols
            and pd.api.types.is_numeric_dtype(df[c])]

X_num = df[num_cols].copy().fillna(0.0)

# one-hot encode categorical
if cat_cols:
    X_cat = pd.get_dummies(df[cat_cols].fillna("Unknown"), drop_first=True)
else:
    X_cat = pd.DataFrame(index=df.index)

X = pd.concat([X_num, X_cat], axis=1)

print(f"[info] Feature matrix shape: {X.shape[0]} rows × {X.shape[1]} columns")
print(f"[info] Using {len(num_cols)} numeric features and {len(X_cat.columns)} encoded categorical features")

# ----------------- 4. TRAIN/TEST SPLIT & MODEL TRAINING -----------------
print("[3] Train/test split and model training...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# eval
y_prob_test = clf.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob_test)
acc = accuracy_score(y_test, y_pred_test)

print(f"[metrics] Test AUC:  {auc:.3f}")
print(f"[metrics] Test ACC:  {acc:.3f}")
print("[metrics] Classification report:")
print(classification_report(y_test, y_pred_test))

# ----------------- 5. REFIT ON FULL DATA -----------------
print("[4] Training final model on full dataset...")
clf.fit(X, y)

# feature importance (top 20)
importances = clf.feature_importances_
fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\n[top 20 features by importance]:")
print(fi.head(20))

# ----------------- 6. PREDICT PROBABILITIES FOR ALL PLAYERS -----------------
print("[5] Predicting cap probabilities for all players...")
df["cap_prob_model"] = clf.predict_proba(X)[:, 1]

# ----------------- 7. SAVE FULL TABLE -----------------
full_cols = [
    "player_id", "player_name",
    target_col, "cap_prob_model"
]
# try to also include some key stats if present
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

df[full_cols].to_csv(OUT_DIR / "rr2_player_cap_probs_full.csv", index=False)
print(f"[saved] Full probabilities table -> {OUT_DIR / 'rr2_player_cap_probs_full.csv'}")

# ----------------- 8. BUILD TOP-40 UNCAPPED LIST -----------------
print("[6] Building Top-40 uncapped list by probability...")

uncapped_mask = df[target_col] == 0
df_uncapped = df[uncapped_mask].copy()

df_uncapped = df_uncapped.sort_values("cap_prob_model", ascending=False)

top40 = df_uncapped.head(40)

top_cols = [
    "player_id", "player_name",
    "cap_prob_model",
    "career_role_guess",
    "career_batting_runs",
    "career_bat_avg",
    "career_bat_sr",
    "career_bowling_wickets",
    "career_bow_econ",
    "venue_fit_score",
    "runs_slope",
    "wk_slope",
    "career_matches_played",
    "career_age_years",
]
top_cols = [c for c in top_cols if c in top40.columns]

top40[top_cols].to_csv(OUT_DIR / "rr2_top40_uncapped_probs.csv", index=False)
print(f"[saved] Top-40 uncapped -> {OUT_DIR / 'rr2_top40_uncapped_probs.csv'}")

print("\n[Top-20 preview]:")
print(top40[top_cols].head(20))
print("\nDone.")

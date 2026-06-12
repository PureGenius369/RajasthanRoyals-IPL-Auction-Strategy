#!/usr/bin/env python3
"""
model_selection_gb.py

Feature-pruned Gradient Boosting model to predict "capped vs uncapped"
and estimate cap probability for today's uncapped players.

Inputs:
    output/player_features_with_phase.parquet
    players.csv

Outputs:
    output/rr2_player_cap_probs_full_gb.csv
    output/rr2_top40_uncapped_probs_gb.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")

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

target_col = "career_is_capped"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found.")

y = df[target_col].astype(int)
print(f"[info] Total players: {len(df)}, capped: {y.sum()}, uncapped: {(1-y).sum()}")

# ----------------- 2. FEATURE PRUNING -----------------
print("[2] Selecting a compact, high-signal feature set...")

# helper: include column only if present
def keep_cols(candidates):
    return [c for c in candidates if c in df.columns]

numeric_candidates = [
    # Career batting
    "career_batting_runs", "career_batting_balls",
    "career_batting_innings", "career_bat_avg", "career_bat_sr",
    "career_boundary_pct",
    # Career bowling
    "career_bowling_wickets", "career_bowling_balls",
    "career_bowling_innings", "career_bow_econ", "career_bow_sr",
    # Role / usage
    "career_matches_played", "career_pct_as_wk", "career_avg_batting_order",
    "career_age_years", "career_bat_runs_per_100", "career_bow_wickets_per_100",
    "career_played_in_ipl",
    # Recent 12m
    "m12_batting_runs", "m12_batting_balls", "m12_bat_avg", "m12_bat_sr",
    "m12_bowling_wickets", "m12_bowling_balls", "m12_bow_econ",
    "m12_matches_played",
    # Recent 24m
    "m24_batting_runs", "m24_batting_balls", "m24_bat_avg", "m24_bat_sr",
    "m24_bowling_wickets", "m24_bowling_balls", "m24_bow_econ",
    "m24_matches_played",
    # Phase batting
    "runs_powerplay", "runs_middle", "runs_death",
    "sr_powerplay", "sr_middle", "sr_death",
    "pres_sr_powerplay", "pres_sr_middle", "pres_sr_death",
    "chase_sr_powerplay", "chase_sr_middle", "chase_sr_death",
    # Phase bowling
    "econ_powerplay", "econ_middle", "econ_death",
    "wk_per100_powerplay", "wk_per100_middle", "wk_per100_death",
    # Venue & trend
    "venue_fit_score", "v_mean_r100", "v_std_r100", "v_venues_count",
    "runs_slope", "wk_slope",
    # Role scarcity
    "players_available", "scarcity_weight",
]

num_cols = keep_cols(numeric_candidates)

# Small safety: if some very important basics are missing, warn
essential = ["career_batting_runs", "career_bowling_wickets", "career_matches_played"]
for e in essential:
    if e not in num_cols:
        print(f"[warn] Essential feature '{e}' not found in table.")

# compact categorical set
cat_candidates = keep_cols([
    "career_role_guess",
    "role_guess",
    "batting_hand", "bowling_hand",
    "batting_type", "bowling_type",
])
# prefer career_role_guess over role_guess
if "career_role_guess" in cat_candidates and "role_guess" in cat_candidates:
    cat_candidates.remove("role_guess")

cat_cols = cat_candidates

print(f"[info] Using {len(num_cols)} numeric features and {len(cat_cols)} categorical features")

# build X
X_num = df[num_cols].copy().astype("float64").fillna(0.0)

if cat_cols:
    X_cat = pd.get_dummies(df[cat_cols].fillna("Unknown"), drop_first=True)
else:
    X_cat = pd.DataFrame(index=df.index)

X = pd.concat([X_num, X_cat], axis=1)
print(f"[info] Final X shape: {X.shape[0]} rows × {X.shape[1]} cols")

# ----------------- 3. TRAIN/TEST SPLIT + HGB + SIGMOID CALIBRATION -----------------
print("[3] Train/test split + HistGradientBoosting with sigmoid calibration...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

base_clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

cal_clf = CalibratedClassifierCV(
    estimator=base_clf,
    cv=3,
    method="sigmoid"
)

cal_clf.fit(X_train, y_train)

y_prob_test = cal_clf.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob_test)
acc = accuracy_score(y_test, y_pred_test)

print(f"[metrics] Test AUC (GB+sigmoid): {auc:.3f}")
print(f"[metrics] Test ACC (GB+sigmoid): {acc:.3f}")
print("[metrics] Classification report (test):")
print(classification_report(y_test, y_pred_test))

# ----------------- 4. REFIT ON FULL DATA -----------------
print("[4] Fitting calibrated model on full dataset...")

base_full = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)
base_full.fit(X, y)

cal_full = CalibratedClassifierCV(
    estimator=base_full,
    cv="prefit",
    method="sigmoid"
)
cal_full.fit(X, y)

# feature importances (optional; may not be available in older sklearn)
try:
    fi = pd.Series(base_full.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[top 20 features by importance (GB)]:")
    print(fi.head(20))
except AttributeError:
    print("\n[warn] feature_importances_ not available for HistGradientBoosting in this sklearn version; skipping importance display.")


# ----------------- 5. PREDICT PROBABILITIES + AGE DECAY -----------------
print("[5] Predicting probabilities & applying age decay...")

df["cap_prob_gb"] = cal_full.predict_proba(X)[:, 1]

age_col = "career_age_years"
if age_col in df.columns:
    age = df[age_col].fillna(df[age_col].median())
else:
    age = pd.Series(28.0, index=df.index)

decay_factor = np.where(
    age <= 32, 1.0,
    np.where(age <= 35, 0.85, 0.70)
)
df["cap_prob_gb_age"] = df["cap_prob_gb"] * decay_factor

print("[info] Prob range (raw):", float(df["cap_prob_gb"].min()), "to", float(df["cap_prob_gb"].max()))
print("[info] Prob range (age-adjusted):", float(df["cap_prob_gb_age"].min()), "to", float(df["cap_prob_gb_age"].max()))

# ----------------- 6. SAVE FULL TABLE -----------------
print("[6] Saving full probability table...")

full_cols = [
    "player_id", "player_name",
    target_col,
    "cap_prob_gb", "cap_prob_gb_age",
]
for c in [
    "career_role_guess",
    "career_batting_runs", "career_bat_avg", "career_bat_sr",
    "career_bowling_wickets", "career_bow_econ",
    "career_matches_played", "career_age_years",
    "venue_fit_score", "runs_slope", "wk_slope"
]:
    if c in df.columns and c not in full_cols:
        full_cols.append(c)

full_cols = [c for c in full_cols if c in df.columns]
df[full_cols].to_csv(OUT_DIR / "rr2_player_cap_probs_full_gb.csv", index=False)
print(f"[saved] rr2_player_cap_probs_full_gb.csv")

# ----------------- 7. TOP-40 UNCAPPED -----------------
print("[7] Building Top-40 uncapped list by age-adjusted probability...")

uncapped_mask = df[target_col] == 0
df_uncapped = df[uncapped_mask].copy().sort_values("cap_prob_gb_age", ascending=False)

top40 = df_uncapped.head(40)

top_cols = [
    "player_id", "player_name",
    "cap_prob_gb", "cap_prob_gb_age",
    "career_role_guess",
    "career_batting_runs", "career_bat_avg", "career_bat_sr",
    "career_bowling_wickets", "career_bow_econ",
    "career_matches_played", "career_age_years",
    "venue_fit_score", "runs_slope", "wk_slope"
]
top_cols = [c for c in top_cols if c in top40.columns]

top40[top_cols].to_csv(OUT_DIR / "rr2_top40_uncapped_probs_gb.csv", index=False)
print(f"[saved] rr2_top40_uncapped_probs_gb.csv")

print("\n[Top-20 preview]:")
print(top40[top_cols].head(20))
print("\nDone.")

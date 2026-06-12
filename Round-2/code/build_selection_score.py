import pandas as pd
import numpy as np

# ========= 1. LOAD DATA =========
# Main feature table
df = pd.read_parquet("output/player_features_with_phase.parquet")

# Players file (for names)
players = pd.read_csv("players.csv")[["player_id", "player_name"]]
players["player_id"] = players["player_id"].astype(str)

# Make sure id is string
df["player_id"] = df["player_id"].astype(str)

# ========= 2. KEEP ONLY UNCAPPED =========
# In our pipeline this column is career_is_uncapped
uncapped_col = "career_is_uncapped"
if uncapped_col not in df.columns:
    raise ValueError(f"Column '{uncapped_col}' not found in feature table.")

df = df[df[uncapped_col] == True].copy()

# ========= 3. EXPERIENCE FILTER =========
def safe_col(name, default=0.0):
    if name in df.columns:
        return df[name].fillna(0)
    return pd.Series(default, index=df.index, dtype="float64")

bat_balls = safe_col("career_batting_balls")
bowl_balls = safe_col("career_bowling_balls")
matches_played = safe_col("career_matches_played")

mask_exp = (bat_balls >= 60) | (bowl_balls >= 60) | (matches_played >= 5)
df = df[mask_exp].copy()

# ========= 4. SAFE BAT & BOWL METRICS =========
bat_runs = safe_col("career_batting_runs")
bat_dismissals = safe_col("career_batting_dismissals")
bat_sr_raw = safe_col("career_bat_sr")

df["career_bat_avg_safe"] = bat_runs / bat_dismissals.replace(0, 1)
df["career_bat_sr_safe"] = bat_sr_raw.clip(50, 220)

bowl_wk = safe_col("career_bowling_wickets")
bowl_econ_raw = safe_col("career_bow_econ")
df["career_bow_econ_safe"] = bowl_econ_raw.clip(4, 12)

# ========= 5. ROBUST NORMALIZATION (PERCENTILES) =========
def robust_norm(s):
    s = s.astype("float64")
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi - lo == 0:
        return pd.Series(0.5, index=s.index)
    return (s.clip(lo, hi) - lo) / (hi - lo)

df["bat_runs_n"] = robust_norm(bat_runs)
df["bat_avg_n"]  = robust_norm(df["career_bat_avg_safe"])
df["bat_sr_n"]   = robust_norm(df["career_bat_sr_safe"])

df["bowl_wk_n"]  = robust_norm(bowl_wk)
df["bowl_econ_n"] = 1 - robust_norm(df["career_bow_econ_safe"])

# ========= 6. VENUE FIT (CLIPPED) =========
venue_raw = safe_col("venue_fit_score", default=1.0)
df["venue_fit_score"] = venue_raw.clip(0.5, 3.0)
df["venue_n"] = robust_norm(df["venue_fit_score"])

# ========= 7. TREND (RUNS/WK SLOPES) =========
runs_slope_raw = safe_col("runs_slope")
wk_slope_raw = safe_col("wk_slope")

df["runs_slope"] = runs_slope_raw.clip(-20, 20)
df["wk_slope"]   = wk_slope_raw.clip(-2, 2)

df["trend_n"] = 0.5 * robust_norm(df["runs_slope"]) + 0.5 * robust_norm(df["wk_slope"])

# ========= 8. ROLE WEIGHTING & RECENT FORM =========
# role_guess: prefer career_role_guess if available, else role_guess
if "career_role_guess" in df.columns:
    df["role_for_score"] = df["career_role_guess"]
else:
    df["role_for_score"] = df.get("role_guess", "unknown")

role_weight = {
    "opener": 1.0,
    "top/middle": 1.0,
    "finisher": 1.05,
    "batting allrounder": 1.08,
    "finishing allrounder": 1.08,
    "bowler": 1.10,
    "unknown": 1.0,
}
df["role_wt"] = df["role_for_score"].map(role_weight).fillna(1.0)

# recent matches in last 12 months
m12_matches = safe_col("m12_matches_played")
df["recent_matches_n"] = m12_matches.clip(0, 20) / 20.0

# ========= 9. FINAL SelectionScore =========
df["SelectionScore"] = (
    0.30 * (0.5*df["bat_runs_n"] + 0.3*df["bat_avg_n"] + 0.2*df["bat_sr_n"]) +
    0.30 * (0.6*df["bowl_wk_n"] + 0.4*df["bowl_econ_n"]) +
    0.15 * df["venue_n"] +
    0.10 * df["trend_n"] +
    0.10 * df["recent_matches_n"] +
    0.05 * df["role_wt"]
)

# ========= 10. MERGE NAMES =========
df = df.merge(players, on="player_id", how="left")

# ========= 11. SORT & EXPORT =========
df = df.sort_values("SelectionScore", ascending=False)

cols = [
    "player_id", "player_name", "role_for_score",
    "SelectionScore",
    "career_batting_runs", "career_bat_avg_safe", "career_bat_sr_safe",
    "career_bowling_wickets", "career_bow_econ_safe",
    "venue_fit_score", "runs_slope", "wk_slope",
    "career_matches_played"
]
cols = [c for c in cols if c in df.columns]

out_path = "output/top100_uncapped_selection_score_v2.csv"
df[cols].head(100).to_csv(out_path, index=False)

print(f"✅ Clean Top-100 generated: {out_path}")
print(df[cols].head(20))

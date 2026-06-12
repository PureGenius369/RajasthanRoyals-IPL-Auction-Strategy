import pandas as pd
from pathlib import Path

OUT_DIR = Path("output")

rf_path = OUT_DIR / "rr2_player_cap_probs_full_calibrated.csv"
gb_path = OUT_DIR / "rr2_player_cap_probs_full_gb.csv"

print("[1] Loading RF and GB outputs...")
rf = pd.read_csv(rf_path)
gb = pd.read_csv(gb_path)

rf["player_id"] = rf["player_id"].astype(str)
gb["player_id"] = gb["player_id"].astype(str)

# Merge
df = rf.merge(
    gb[["player_id", "cap_prob_gb_age"]],
    on="player_id",
    how="left"
)

# Safety: fill missing GB with RF (rare but safe)
df["cap_prob_gb_age"] = df["cap_prob_gb_age"].fillna(df["cap_prob_calibrated_age"])

# Final ensemble probability
df["final_cap_prob"] = (
    0.80 * df["cap_prob_calibrated_age"] +
    0.20 * df["cap_prob_gb_age"]
)

print("[info] Final prob range:",
      df["final_cap_prob"].min(),
      "to",
      df["final_cap_prob"].max())

# ------------------------------
# Build final Top-40 UNCAPPED
# ------------------------------
print("[2] Building final Top-40 uncapped list...")

uncapped = df[df["career_is_capped"] == 0].copy()
uncapped = uncapped.sort_values("final_cap_prob", ascending=False)

top40 = uncapped.head(40)

top_cols = [
    "player_id",
    "player_name",
    "final_cap_prob",
    "cap_prob_calibrated_age",
    "cap_prob_gb_age",
    "career_role_guess",
    "career_batting_runs",
    "career_bowling_wickets",
    "career_matches_played",
    "career_age_years",
    "venue_fit_score",
    "runs_slope",
    "wk_slope"
]

top_cols = [c for c in top_cols if c in top40.columns]

top40[top_cols].to_csv(
    OUT_DIR / "rr2_top40_uncapped_probs_final_ensemble.csv",
    index=False
)

print("[saved] rr2_top40_uncapped_probs_final_ensemble.csv")
print("\n[Top-20 preview]")
print(top40[top_cols].head(20))
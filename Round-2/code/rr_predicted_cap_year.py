import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("output")
inp = OUT_DIR / "rr2_top40_uncapped_probs_final_ensemble.csv"
outp = OUT_DIR / "rr2_top40_uncapped_with_predicted_year.csv"

df = pd.read_csv(inp)

current_year = datetime.now().year

# --- Normalize slopes safely ---
def norm_clip(s):
    s = s.fillna(0)
    s = np.clip(s, -100, 100)
    if s.std() == 0:
        return pd.Series(0, index=s.index)
    return (s - s.mean()) / s.std()

df["runs_slope_norm"] = norm_clip(df["runs_slope"])
df["wk_slope_norm"] = norm_clip(df["wk_slope"])

# --- Growth rate ---
df["prob_growth_per_year"] = (
    0.04
    + 0.02 * df["runs_slope_norm"]
    + 0.02 * df["wk_slope_norm"]
)

df["prob_growth_per_year"] = df["prob_growth_per_year"].clip(0.02, 0.12)

# --- Years to reach 0.80 probability ---
df["years_to_cap"] = np.ceil(
    (0.80 - df["final_cap_prob"]) / df["prob_growth_per_year"]
)

df["years_to_cap"] = df["years_to_cap"].clip(0, 5)

df["predicted_cap_year"] = current_year + df["years_to_cap"]

df[
    [
        "player_id","player_name",
        "final_cap_prob",
        "predicted_cap_year",
        "career_role_guess",
        "career_age_years",
        "career_matches_played",
        "runs_slope","wk_slope"
    ]
].to_csv(outp, index=False)

print("Saved:", outp)
print(df[["player_name","final_cap_prob","predicted_cap_year"]].head(10))

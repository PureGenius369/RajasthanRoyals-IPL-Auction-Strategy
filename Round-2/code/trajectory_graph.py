#!/usr/bin/env python3
"""
rr2_rocket_trajectory_v2.py

Rocket-style trajectory plot:

- Line: average final cap probability of *capped* players by age
- Points: Top-40 uncapped players on the same age–probability plane

Uses ensemble probability:
    final_cap_prob = 0.80 * cap_prob_calibrated_age (RF) +
                     0.20 * cap_prob_gb_age (GB)

Outputs:
    output/rr2_rocket_trajectory_v2.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("output")

# ---------------- 1. LOAD FULL TABLES & BUILD FINAL PROB ----------------
print("[1] Loading RF & GB tables...")
rf = pd.read_csv(OUT_DIR / "rr2_player_cap_probs_full_calibrated.csv")
gb = pd.read_csv(OUT_DIR / "rr2_player_cap_probs_full_gb.csv")

rf["player_id"] = rf["player_id"].astype(str)
gb["player_id"] = gb["player_id"].astype(str)

df = rf.merge(
    gb[["player_id", "cap_prob_gb_age"]],
    on="player_id",
    how="left"
)

df["cap_prob_gb_age"] = df["cap_prob_gb_age"].fillna(df["cap_prob_calibrated_age"])
df["final_cap_prob"] = (
    0.80 * df["cap_prob_calibrated_age"] +
    0.20 * df["cap_prob_gb_age"]
)

# keep realistic age band
df = df[df["career_age_years"].between(17, 40)]

# ---------------- 2. BUILD CAPPED TRAJECTORY ----------------
print("[2] Computing average capped trajectory...")

capped = df[df["career_is_capped"] == 1].copy()

# define age bins (integer ages)
age_bins = np.arange(18, 41)  # 18..40
age_centers = age_bins  # just use integer ages as x

mean_probs = []
for a in age_bins:
    subset = capped[
        (capped["career_age_years"] >= a - 0.5) &
        (capped["career_age_years"] <  a + 0.5)
    ]
    if len(subset) == 0:
        mean_probs.append(np.nan)
    else:
        mean_probs.append(subset["final_cap_prob"].mean())

mean_probs = np.array(mean_probs)

# ---------------- 3. LOAD TOP-40 UNCAPPED ----------------
print("[3] Loading Top-40 uncapped...")
top40 = pd.read_csv(OUT_DIR / "rr2_top40_uncapped_probs_final_ensemble.csv")
top40 = top40[top40["career_age_years"].between(17, 40)]

# ensure we have final_cap_prob column; if not, merge
if "final_cap_prob" not in top40.columns:
    top40["player_id"] = top40["player_id"].astype(str)
    top40 = top40.merge(
        df[["player_id", "final_cap_prob"]],
        on="player_id",
        how="left"
    )

# ---------------- 4. PLOT ROCKET TRAJECTORY ----------------
print("[4] Plotting rocket trajectory...")

plt.figure(figsize=(10, 6))

# line: historical capped trajectory
plt.plot(
    age_centers,
    mean_probs,
    label="Avg capped players (historical trajectory)"
)

# dots: Top-40 uncapped projections
plt.scatter(
    top40["career_age_years"],
    top40["final_cap_prob"],
    label="Top-40 uncapped (projected)"
)

plt.xlabel("Career Age (Years)")
plt.ylabel("Final cap probability")
plt.title("Rocket Trajectory: Capped vs Projected Top-40 Uncapped Players")
plt.legend()
plt.grid(True)

out_path = OUT_DIR / "rr2_rocket_trajectory_v2.png"
plt.tight_layout()
plt.savefig(out_path)
plt.close()

print("[saved]", out_path)

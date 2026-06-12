# rr2_prob_global_only.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("output")

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
df["final_cap_prob"] = 0.80 * df["cap_prob_calibrated_age"] + 0.20 * df["cap_prob_gb_age"]

plt.figure(figsize=(10, 6))
bins = np.linspace(0, 1.0, 21)

plt.hist(df["final_cap_prob"], bins=bins)
plt.axvline(0.60, linestyle="--")  # “elite” threshold

plt.xlabel("Final cap probability")
plt.ylabel("Number of players")
plt.title("Cap Probability Distribution: All 5,484 Players")
plt.grid(True)

out_path = OUT_DIR / "rr2_prob_distribution_global.png"
plt.tight_layout()
plt.savefig(out_path)
plt.close()

print("saved", out_path)

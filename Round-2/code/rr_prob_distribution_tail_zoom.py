# rr2_prob_tail_with_top40.py
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

top40 = pd.read_csv(OUT_DIR / "rr2_top40_uncapped_probs_final_ensemble.csv")

# --- focus only on tail where interesting stuff happens ---
tail_df = df[df["final_cap_prob"] >= 0.4]
tail_top40 = top40[top40["final_cap_prob"] >= 0.4]

plt.figure(figsize=(10, 6))

bins = np.linspace(0.4, 1.0, 13)  # coarser bins, only tail

# all players in tail
plt.hist(
    tail_df["final_cap_prob"],
    bins=bins,
    alpha=0.6,
    label="All players (prob ≥ 0.4)"
)

# Top-40 in tail
plt.hist(
    tail_top40["final_cap_prob"],
    bins=bins,
    alpha=0.8,
    label="Top-40 uncapped"
)

plt.xlabel("Final cap probability (tail zoom)")
plt.ylabel("Number of players")
plt.title("Right-Tail Zoom: All Players vs Top-40 Uncapped")
plt.legend()
plt.grid(True)

out_path = OUT_DIR / "rr2_prob_distribution_tail_zoom.png"
plt.tight_layout()
plt.savefig(out_path)
plt.close()

print("saved", out_path)

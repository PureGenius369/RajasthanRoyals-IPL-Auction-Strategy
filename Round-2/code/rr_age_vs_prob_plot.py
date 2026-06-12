import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

# --- filter to realistic ages 17–40 ---
df = df[df["career_age_years"].between(17, 40)]
top40 = top40[top40["career_age_years"].between(17, 40)]

plt.figure(figsize=(10, 6))

plt.scatter(
    df["career_age_years"],
    df["final_cap_prob"],
    alpha=0.3,
    s=12,
    label="All players"
)

plt.scatter(
    top40["career_age_years"],
    top40["final_cap_prob"],
    s=60,
    label="Top-40 uncapped"
)

plt.xlabel("Career Age (Years)")
plt.ylabel("Final cap probability")
plt.title("Age vs Cap Probability (Realistic Age Band 17–40)")
plt.legend()
plt.grid(True)

out_path = OUT_DIR / "rr2_age_vs_cap_probability_v2.png"
plt.tight_layout()
plt.savefig(out_path)
plt.close()

print("saved", out_path)

#!/usr/bin/env python3
"""
rr2_role_composition_plot_pct.py

Condensed role composition (% based):
- All players vs Top-40 uncapped

Outputs:
    output/rr2_role_composition_all_vs_top40_pct.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("output")

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
print("[1] Loading RF table and players...")

rf = pd.read_csv(OUT_DIR / "rr2_player_cap_probs_full_calibrated.csv")
players = pd.read_csv("players.csv")

rf["player_id"] = rf["player_id"].astype(str)
players["player_id"] = players["player_id"].astype(str)

df = rf.merge(
    players[["player_id", "bowling_type", "wicket_keeper"]],
    on="player_id",
    how="left"
)

top40 = pd.read_csv(OUT_DIR / "rr2_top40_uncapped_probs_final_ensemble.csv")
top40["player_id"] = top40["player_id"].astype(str)

# -------------------------------------------------------------
# 2. ROLE MAPPING
# -------------------------------------------------------------
def detect_spin_or_pace(bowling_type):
    if not isinstance(bowling_type, str):
        return None
    s = bowling_type.lower()
    spin_keywords = [
        "spin", "offbreak", "off break", "legbreak", "leg break",
        "orthodox", "slow", "chinaman"
    ]
    if any(k in s for k in spin_keywords):
        return "spin"
    if s.strip() != "":
        return "pace"
    return None

def map_condensed_role(row):
    role_raw = str(row.get("career_role_guess", "")).lower()
    bowl_style = detect_spin_or_pace(row.get("bowling_type", ""))
    is_wk = row.get("wicket_keeper", 0)

    # Wicketkeeper
    if "wk" in role_raw or "keeper" in role_raw or is_wk == 1:
        return "Wicketkeeper-Batter"

    # Allrounders
    if "allrounder" in role_raw:
        if bowl_style == "spin":
            return "Spin Allrounder"
        if bowl_style == "pace":
            return "Pace Allrounder"
        return "Pace Allrounder"

    # Specialist bowlers
    if "spinner" in role_raw or (bowl_style == "spin"):
        return "Specialist Spinner"
    if ("pacer" in role_raw or "fast" in role_raw or "seamer" in role_raw
        or (bowl_style == "pace")):
        return "Specialist Pacer"

    # Batters
    if "open" in role_raw:
        return "Opener"
    if "finish" in role_raw:
        return "Finisher"
    if "top" in role_raw or "middle" in role_raw:
        return "Middle-order"

    # Fallback
    return "Middle-order"

print("[2] Mapping condensed roles...")
df["condensed_role"] = df.apply(map_condensed_role, axis=1)

top40 = top40.merge(
    df[["player_id", "condensed_role"]],
    on="player_id",
    how="left"
)

# -------------------------------------------------------------
# 3. BUILD COUNTS & CONVERT TO PERCENTAGES
# -------------------------------------------------------------
categories = [
    "Opener",
    "Middle-order",
    "Finisher",
    "Spin Allrounder",
    "Pace Allrounder",
    "Specialist Spinner",
    "Specialist Pacer",
    "Wicketkeeper-Batter",
]

all_counts = df["condensed_role"].value_counts().reindex(categories).fillna(0)
top_counts = top40["condensed_role"].value_counts().reindex(categories).fillna(0)

all_pct = 100 * all_counts / all_counts.sum()
top_pct = 100 * top_counts / top_counts.sum()

print("\n[info] All players %:")
print(all_pct.round(2))
print("\n[info] Top-40 %:")
print(top_pct.round(2))

# -------------------------------------------------------------
# 4. PLOT PERCENTAGE BAR CHART
# -------------------------------------------------------------
x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(x - width/2, all_pct.values, width, label="All players (%)")
plt.bar(x + width/2, top_pct.values, width, label="Top-40 uncapped (%)")

plt.xticks(x, categories, rotation=30, ha="right")
plt.ylabel("Percentage of players (%)")
plt.title("Condensed Role Composition (%): All Players vs Top-40 Uncapped")
plt.legend()
plt.grid(axis="y")

out_path = OUT_DIR / "rr2_role_composition_all_vs_top40_pct.png"
plt.tight_layout()
plt.savefig(out_path)
plt.close()

print("\n[saved]", out_path)

# extract_player_features.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path("C:/Users/manns/OneDrive/Desktop/RR/New_Data/")
BB_FILE = BASE / "india_matches_ball_by_ball.csv"      # your filename
PLAYING_FILE = BASE / "playing_eleven.csv"
PLAYERS_FILE = BASE / "players.csv"
BAT_IMP = BASE / "Top_20_Batters_After_100-Ball_Filter.csv"
BOWL_IMP = BASE / "Top_20_Bowling_Impact_Scores.csv"


OUT_DIR = BASE / "features_out"
OUT_DIR.mkdir(exist_ok=True)

CHUNK = 200000   # tune smaller if memory constrained

# helper
def safe_int(x):
    try:
        return int(float(x))
    except:
        return np.nan

def phase_from_over(o):
    try:
        o = int(float(o))
    except:
        return "Unknown"
    if 1 <= o <= 6:
        return "powerplay"
    if 7 <= o <= 15:
        return "middle"
    if o >= 16:
        return "death"
    return "unknown"

# accumulator dicts for batting and bowling
bat_acc = {}   # player_id -> dict of counters
bowl_acc = {}

def ensure_bat(pid):
    if pid not in bat_acc:
        bat_acc[pid] = {
            "player_id": pid, "balls":0, "runs":0, "fours":0, "sixes":0,
            "dismissals":0, "innings_set": set(),
            "balls_powerplay":0, "runs_powerplay":0,
            "balls_middle":0, "runs_middle":0,
            "balls_death":0, "runs_death":0,
        }

def ensure_bowl(pid):
    if pid not in bowl_acc:
        bowl_acc[pid] = {
            "player_id": pid, "balls_bowled":0, "runs_conceded":0, "wickets":0,
            "overs_bowled":0.0,
            "balls_pp":0, "runs_pp":0, "wickets_pp":0,
            "balls_middle":0, "runs_middle":0, "wickets_middle":0,
            "balls_death":0, "runs_death":0, "wickets_death":0,
            "pace":0, "spin":0
        }

print("Streaming ball-by-ball and aggregating ...")
reader = pd.read_csv(BB_FILE, dtype=str, low_memory=False, chunksize=CHUNK)

for chunk in reader:
    # trim spaces in column names if necessary
    chunk.columns = [c.strip() for c in chunk.columns]
    # keep only Indian teams already filtered; otherwise filter on batting_team_name / bowling_team_name
    # compute phase
    if "over_num" in chunk.columns:
        chunk["phase"] = chunk["over_num"].apply(phase_from_over)
    elif "overs" in chunk.columns:
        # if overs decimal format exists, fallback to integer part
        chunk["phase"] = chunk["overs"].astype(float).apply(lambda x: phase_from_over(int(np.floor(x))))
    else:
        chunk["phase"] = "unknown"

    # cast numeric-ish fields
    for col in ["runs","is_four","is_six","is_wicket","ball","over_ball_num","innings_ball_num","bowl_speed","overs","over_num"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # iterate rows (vectorised groups possible but chunked safe loop is simple)
    for _, r in chunk.iterrows():
        # BATTING aggregations
        pid = r.get("striker_id")
        if pd.notna(pid):
            ensure_bat(pid)
            b = bat_acc[pid]
            balls_inc = 1
            runs_inc = r.get("runs") if pd.notna(r.get("runs")) else 0
            fours_inc = 1 if r.get("is_four") in [1, "1", "True", "true"] else 0
            sixes_inc = 1 if r.get("is_six") in [1, "1", "True", "true"] else 0
            # update
            b["balls"] += balls_inc
            b["runs"] += runs_inc
            b["fours"] += fours_inc
            b["sixes"] += sixes_inc
            # innings set
            mid = r.get("match_id")
            inn = r.get("innings")
            if pd.notna(mid) and pd.notna(inn):
                b["innings_set"].add((mid, inn))
            # phases
            ph = r.get("phase", "unknown")
            if ph == "powerplay":
                b["balls_powerplay"] += 1
                b["runs_powerplay"] += runs_inc
            elif ph == "middle":
                b["balls_middle"] += 1
                b["runs_middle"] += runs_inc
            elif ph == "death":
                b["balls_death"] += 1
                b["runs_death"] += runs_inc

            # dismissal
            if r.get("is_wicket") in ["1", 1, "True", "true"]:
                # only count if striker dismissed (dismissed_player_id == striker_id or dismissal_type indicates striker)
                # safe approach: increment striker dismissal when dismissed_player_id equals striker_id
                if "dismissed_player_id" in chunk.columns and str(r.get("dismissed_player_id")) == str(pid):
                    b["dismissals"] += 1

        # BOWLING aggregations
        bowid = r.get("bowler_id")
        if pd.notna(bowid):
            ensure_bowl(bowid)
            B = bowl_acc[bowid]
            ball = 1
            runs_c = r.get("runs") if pd.notna(r.get("runs")) else 0
            B["balls_bowled"] += 1
            B["runs_conceded"] += runs_c
            # wickets: check is_wicket and dismissed_player_id not null and dismissal credited to bowler.
            if r.get("is_wicket") in ["1", 1, "True", "true"]:
                # increase wicket count - crude but acceptable if ball-level indicates wicket
                B["wickets"] += 1
            # phase
            ph = r.get("phase","unknown")
            if ph == "powerplay":
                B["balls_pp"] += 1
                B["runs_pp"] += runs_c
                if r.get("is_wicket") in ["1",1,"True","true"]: B["wickets_pp"] += 1
            elif ph == "middle":
                B["balls_middle"] += 1
                B["runs_middle"] += runs_c
                if r.get("is_wicket") in ["1",1,"True","true"]: B["wickets_middle"] += 1
            elif ph == "death":
                B["balls_death"] += 1
                B["runs_death"] += runs_c
                if r.get("is_wicket") in ["1",1,"True","true"]: B["wickets_death"] += 1
            # pace or spin
            pos = r.get("pace_or_spin")
            if pd.notna(pos):
                if str(pos).strip().lower().startswith("pace"):
                    B["pace"] += 1
                elif str(pos).strip().lower().startswith("spin"):
                    B["spin"] += 1

# Build DataFrames
print("Converting accumulators to dataframes ...")
bat_rows = []
for pid, d in bat_acc.items():
    bat_rows.append({
        "player_id": d["player_id"],
        "balls": d["balls"],
        "runs": d["runs"],
        "fours": d["fours"],
        "sixes": d["sixes"],
        "dismissals": d["dismissals"],
        "innings_count": len(d["innings_set"]),
        "balls_powerplay": d["balls_powerplay"],
        "runs_powerplay": d["runs_powerplay"],
        "balls_middle": d["balls_middle"],
        "runs_middle": d["runs_middle"],
        "balls_death": d["balls_death"],
        "runs_death": d["runs_death"]
    })
bdf = pd.DataFrame.from_records(bat_rows)

bowl_rows = []
for pid, d in bowl_acc.items():
    balls_total = d["balls_bowled"]
    overs = balls_total // 6 + (balls_total % 6)/10.0
    bowl_rows.append({
        "player_id": d["player_id"],
        "balls_bowled": d["balls_bowled"],
        "runs_conceded": d["runs_conceded"],
        "wickets": d["wickets"],
        "overs_bowled": overs,
        "balls_pp": d["balls_pp"],
        "runs_pp": d["runs_pp"],
        "wickets_pp": d["wickets_pp"],
        "balls_middle": d["balls_middle"],
        "runs_middle": d["runs_middle"],
        "wickets_middle": d["wickets_middle"],
        "balls_death": d["balls_death"],
        "runs_death": d["runs_death"],
        "wickets_death": d["wickets_death"],
        "pace_count": d["pace"],
        "spin_count": d["spin"]
    })
bowl_df = pd.DataFrame.from_records(bowl_rows)

# compute derived metrics for batting DF
def safe_div(a,b):
    return (a/b) if b and b>0 else 0

if not bdf.empty:
    bdf["sr_overall"] = bdf.apply(lambda r: safe_div(r["runs"], r["balls"]) * 100, axis=1)
    bdf["sr_powerplay"] = bdf.apply(lambda r: safe_div(r["runs_powerplay"], r["balls_powerplay"]) * 100, axis=1)
    bdf["sr_middle"] = bdf.apply(lambda r: safe_div(r["runs_middle"], r["balls_middle"]) * 100, axis=1)
    bdf["sr_death"] = bdf.apply(lambda r: safe_div(r["runs_death"], r["balls_death"]) * 100, axis=1)
    bdf["boundary_pct_overall"] = bdf.apply(lambda r: safe_div((r["fours"]+r["sixes"]), r["balls"]), axis=1)
    bdf["not_out_rate"] = bdf.apply(lambda r: 1 - safe_div(r["dismissals"], r["innings_count"]) if r["innings_count"]>0 else 0, axis=1)

if not bowl_df.empty:
    bowl_df["eco_overall"] = bowl_df.apply(lambda r: safe_div(r["runs_conceded"], r["balls_bowled"]) * 6, axis=1)
    bowl_df["eco_pp"] = bowl_df.apply(lambda r: safe_div(r["runs_pp"], r["balls_pp"]) * 6, axis=1)
    bowl_df["eco_middle"] = bowl_df.apply(lambda r: safe_div(r["runs_middle"], r["balls_middle"]) * 6, axis=1)
    bowl_df["eco_death"] = bowl_df.apply(lambda r: safe_div(r["runs_death"], r["balls_death"]) * 6, axis=1)
    bowl_df["sr_overall"] = bowl_df.apply(lambda r: safe_div(r["balls_bowled"], r["wickets"]), axis=1)
    bowl_df["wk_rate"] = bowl_df.apply(lambda r: safe_div(r["wickets"], r["balls_bowled"]), axis=1)
    bowl_df["pct_pace"] = bowl_df.apply(lambda r: safe_div(r["pace_count"], (r["pace_count"]+r["spin_count"])), axis=1)

# Save intermediate csvs
bdf.to_csv(OUT_DIR / "bb_batting_agg.csv", index=False)
bowl_df.to_csv(OUT_DIR / "bb_bowling_agg.csv", index=False)

# Merge with existing batting/bowling impact CSVs if present
print("Merging with precomputed impact files if present ...")
bat_imp = None
bowl_imp = None
if BAT_IMP.exists():
    bat_imp = pd.read_csv(BAT_IMP, dtype=str)
    # ensure numeric
    for c in ["batting_impact_score","runs","balls"]:
        if c in bat_imp.columns:
            bat_imp[c] = pd.to_numeric(bat_imp[c], errors="coerce")
if BOWL_IMP.exists():
    bowl_imp = pd.read_csv(BOWL_IMP, dtype=str)
    for c in ["bowling_impact_score","runs_conceded","balls_bowled","wickets"]:
        if c in bowl_imp.columns:
            bowl_imp[c] = pd.to_numeric(bowl_imp[c], errors="coerce")

# prefer player_id join; if the batting impact file uses striker_name, you can join on name too (risky)
# Load player meta
players = pd.read_csv(PLAYERS_FILE, dtype=str) if PLAYERS_FILE.exists() else None

# prepare master player feature table
# join bdf and bowl_df on player_id
master = pd.merge(bdf, bowl_df, on="player_id", how="outer", suffixes=("_bat","_bowl"))
# join with player meta
if players is not None:
    master = master.merge(players[['player_id','player_name','bowling_type','batting_type','wicket_keeper','date_of_birth']], on="player_id", how="left")

# join impact scores if available using player name or player_id
if bat_imp is not None:
    # bat_imp might have striker_name; try both joins
    if 'player_id' in bat_imp.columns:
        master = master.merge(bat_imp[['player_id','batting_impact_score']], on='player_id', how='left')
    elif 'striker_name' in bat_imp.columns:
        master = master.merge(bat_imp[['striker_name','batting_impact_score']], left_on='player_name', right_on='striker_name', how='left')

if bowl_imp is not None:
    if 'player_id' in bowl_imp.columns:
        master = master.merge(bowl_imp[['player_id','bowling_impact_score']], on='player_id', how='left')
    elif 'bowler_name' in bowl_imp.columns:
        master = master.merge(bowl_imp[['bowler_name','bowling_impact_score']], left_on='player_name', right_on='bowler_name', how='left')

# Role inference (simplified rule-based)
def infer_batting_role(row):
    bo = row.get("median_batting_order", np.nan)
    # fallback to balls distribution: high % death balls -> finisher
    try:
        bo = float(bo)
    except:
        bo = np.nan
    if not np.isnan(bo):
        if bo <= 2:
            return "Opener"
        if bo <= 4:
            return "Top/Anchor"
        if bo <= 6:
            return "Middle"
        return "Finisher"
    # fallback by death ball %:
    total_balls = row.get("balls", 0)
    death_balls = row.get("balls_death", 0)
    if total_balls and death_balls/total_balls > 0.25:
        return "Finisher"
    return "Batter"

# compute median batting order if we have playing_eleven history — else approximate by average batting_order in ball-by-ball (not implemented here)
if "batting_order" in chunk.columns:
    # If playing_eleven used, you'd compute per-player median batting_order; placeholder:
    pass

# We'll infer a naive bowling role from percent of balls in phases
master["pct_death_balls_bat"] = master.apply(lambda r: r.get("balls_death",0)/r["balls"] if r.get("balls") and r.get("balls")>0 else 0, axis=1)
master["pct_death_balls_bowl"] = master.apply(lambda r: r.get("balls_death",0)/r["balls_bowled"] if r.get("balls_bowled") and r.get("balls_bowled")>0 else 0, axis=1)
def infer_bowling_role(row):
    if row.get("balls_bowled",0) == 0:
        return None
    if row.get("pct_death_balls_bowl",0) > 0.35:
        return "Death"
    if row.get("balls_pp",0)/row.get("balls_bowled",1) > 0.25:
        return "Powerplay"
    if row.get("spin_count",0) > row.get("pace_count",0):
        return "Spin"
    return "MiddleOvers"

master["batting_role"] = master.apply(infer_batting_role, axis=1)
master["bowling_role"] = master.apply(infer_bowling_role, axis=1)

# final combined impact (rule-based)
# if both impacts present and player seems all-rounder (has both sides), combine:
def combine_impact(row):
    b = row.get("batting_impact_score", np.nan)
    bo = row.get("bowling_impact_score", np.nan)
    if not np.isnan(b) and not np.isnan(bo):
        # determine primary: if batting balls >> bowling balls, weight batting more
        balls_bat = row.get("balls", 0) or 0
        balls_bowl = row.get("balls_bowled", 0) or 0
        if balls_bat > 2*balls_bowl:
            return 0.7*b + 0.3*bo
        if balls_bowl > 2*balls_bat:
            return 0.3*b + 0.7*bo
        return 0.55*b + 0.45*bo
    if not np.isnan(b):
        return b
    if not np.isnan(bo):
        return bo
    return np.nan

master["combined_impact"] = master.apply(combine_impact, axis=1)

# write outputs
master.to_csv(OUT_DIR / "player_features_master.csv", index=False)
print("Wrote player_features_master.csv to", OUT_DIR)
# optionally write top 20 batting & bowling
master_bat_top = master.sort_values("batting_impact_score", ascending=False).head(20)
master_bowl_top = master.sort_values("bowling_impact_score", ascending=False).head(20)
master_bat_top.to_csv(OUT_DIR / "top20_batting.csv", index=False)
master_bowl_top.to_csv(OUT_DIR / "top20_bowling.csv", index=False)
print("Wrote top20 files.")

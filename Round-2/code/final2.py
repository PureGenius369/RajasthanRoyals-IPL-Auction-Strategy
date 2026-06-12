#!/usr/bin/env python3
"""
rr2_ingest_features.py

Ingests the 6 CSV files from the provided ZIP (streamed), cleans them,
and creates a player master and player-level feature table suitable for Round-2 modeling.

Outputs:
 - ./output/player_master.parquet
 - ./output/player_features.parquet
 - ./output/player_master.csv
 - ./output/player_features.csv
"""
import zipfile
import pandas as pd
import numpy as np
from io import TextIOWrapper
from pathlib import Path
from datetime import datetime

ZIP_PATH = "compressed_files.zip"  # change if needed
OUT_DIR = Path("./output")
OUT_DIR.mkdir(exist_ok=True)

# --- helper utilities -------------------------------------------------------
def safe_read_csv_from_zip(z, name, nrows=None, dtype=None, parse_dates=None):
    try:
        with z.open(name) as f:
            return pd.read_csv(TextIOWrapper(f, encoding="utf-8"), nrows=nrows, dtype=dtype, parse_dates=parse_dates, low_memory=False)
    except KeyError:
        raise FileNotFoundError(f"{name} not found in zip")
    except Exception as e:
        raise

def parse_date_flexible(s):
    if pd.isna(s):
        return pd.NaT
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.NaT

# --- load data (stream from zip) -------------------------------------------
print("Opening zip and reading CSVs...")
with zipfile.ZipFile(ZIP_PATH) as z:
    bbb = safe_read_csv_from_zip(z, "ball_by_ball.csv")
    matches = safe_read_csv_from_zip(z, "matches.csv", parse_dates=['match_date'])
    ptc = safe_read_csv_from_zip(z, "player_team_competition_mapping.csv")
    pe = safe_read_csv_from_zip(z, "playing_eleven.csv")
    comps = safe_read_csv_from_zip(z, "competitions.csv")
    players = safe_read_csv_from_zip(z, "players.csv")  # nationality empty as discussed

print("Files loaded. Basic shapes:")
print("ball_by_ball:", bbb.shape)
print("matches:", matches.shape)
print("player_team_competition_mapping:", ptc.shape)
print("playing_eleven:", pe.shape)
print("competitions:", comps.shape)
print("players:", players.shape)

# --- basic cleaning & normalize column names --------------------------------
def clean_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df

bbb = clean_cols(bbb)
matches = clean_cols(matches)
ptc = clean_cols(ptc)
pe = clean_cols(pe)
comps = clean_cols(comps)
players = clean_cols(players)

# ensure key types
bbb['match_id'] = bbb['match_id'].astype(str)
matches['match_id'] = matches['match_id'].astype(str)
pe['match_id'] = pe['match_id'].astype(str)
ptc['comp_id'] = ptc['comp_id'].astype(comps['id'].dtype) if 'id' in comps else ptc['comp_id']

# parse match_date if not parsed
if 'match_date' in matches.columns and matches['match_date'].dtype == object:
    matches['match_date'] = pd.to_datetime(matches['match_date'], errors='coerce')

# --- attach comp_name & season to matches -----------------------------------
if 'comp_id' in matches.columns and 'id' in comps.columns:
    matches = matches.merge(comps.rename(columns={'id':'comp_id','name':'comp_name','season':'comp_season'}),
                            on='comp_id', how='left')

# fallback season from match_date
if 'comp_season' not in matches.columns or matches['comp_season'].isna().all():
    matches['comp_season'] = matches['match_date'].dt.year

# --- detect capped players (heuristic) --------------------------------------
# Heuristic: if a player appears in playing_eleven for a match where home_team or away_team == 'India'
# or match has India as one of the teams, mark as capped. This requires matches to have home_team/away_team text.
print("Detecting capped players via matches where India played...")
india_team_mask = matches['home_team'].fillna('').str.lower().str.contains('india') | matches['away_team'].fillna('').str.lower().str.contains('india')
india_match_ids = matches.loc[india_team_mask, 'match_id'].unique()
# who played in those matches?
pe_india = pe[pe['match_id'].isin(india_match_ids)]
capped_player_ids = set(pe_india['player_id'].astype(str).unique())
print(f"Detected {len(capped_player_ids)} players who played in India matches (treated as capped).")

# --- build per-innings batting aggregates from ball_by_ball -------------------
print("Computing per-innings batting aggregates...")
# Normalize columns used
bbb['striker_id'] = bbb['striker_id'].astype(str)
bbb['bowler_id'] = bbb['bowler_id'].astype(str)
bbb['is_wicket'] = pd.to_numeric(bbb['is_wicket'], errors='coerce').fillna(0).astype(int)
bbb['is_four'] = pd.to_numeric(bbb['is_four'], errors='coerce').fillna(0).astype(int)
bbb['is_six'] = pd.to_numeric(bbb['is_six'], errors='coerce').fillna(0).astype(int)
bbb['runs'] = pd.to_numeric(bbb['runs'], errors='coerce').fillna(0)

# per-innings batting (player as striker)
bat_innings = (bbb.groupby(['match_id','innings','striker_id'])
                .agg(runs=('runs','sum'),
                     balls=('over_ball_num','count'),
                     fours=('is_four','sum'),
                     sixes=('is_six','sum'),
                     dismissals=('is_wicket', 'sum'))  # Note: counts all wickets on balls where is_wicket==1; may include runouts etc.
                .reset_index().rename(columns={'striker_id':'player_id'}))
# mark dismissed if dismissed_player_id equals player_id in any ball of innings (more precise)
if 'dismissed_player_id' in bbb.columns:
    # create dismissed flag per innings if any dismissed_player_id equals striker
    dismissed = (bbb[~bbb['dismissed_player_id'].isna()]
                 .assign(disc_pid = bbb['dismissed_player_id'].astype(str))
                 .groupby(['match_id','innings','disc_pid']).size().reset_index().rename(columns={'disc_pid':'player_id',0:'n'}))
    dismissed['player_id'] = dismissed['player_id'].astype(str)
    dismissed = dismissed.rename(columns={'n':'dismissed_count'})
    bat_innings = bat_innings.merge(dismissed, on=['match_id','innings','player_id'], how='left')
    bat_innings['dismissed_count'] = bat_innings['dismissed_count'].fillna(0).astype(int)
else:
    bat_innings['dismissed_count'] = bat_innings['dismissals']

# per-innings bowling aggregates (player as bowler)
print("Computing per-innings bowling aggregates...")
bow_innings = (bbb.groupby(['match_id','innings','bowler_id'])
                .agg(runs_conceded=('runs','sum'),
                     balls_bowled=('over_ball_num','count'),
                     wickets=('is_wicket','sum'))
                .reset_index().rename(columns={'bowler_id':'player_id'}))

# --- merge match_date & comp info into innings --------------------------------
bat_innings = bat_innings.merge(matches[['match_id','match_date','comp_id','comp_name','comp_season','match_format','stage','venue_name','home_team','away_team']],
                                on='match_id', how='left')
bow_innings = bow_innings.merge(matches[['match_id','match_date','comp_id','comp_name','comp_season','match_format','stage','venue_name','home_team','away_team']],
                                on='match_id', how='left')

# --- player season/rolling aggregates ---------------------------------------
print("Aggregating player-level features (career + 12/24 months)...")
# unify player_id type
players['player_id'] = players['player_id'].astype(str)
bat_innings['player_id'] = bat_innings['player_id'].astype(str)
bow_innings['player_id'] = bow_innings['player_id'].astype(str)
pe['player_id'] = pe['player_id'].astype(str)
ptc['player_id'] = ptc['player_id'].astype(str)

# helper: compute features for a given cutoff date (e.g., "now" or last match date)
last_date = matches['match_date'].max()
if pd.isna(last_date):
    last_date = pd.Timestamp.now()
print("Last match date in dataset:", last_date)

def agg_features(cutoff_date=None):
    """
    Returns a DataFrame of player-level features aggregated up to cutoff_date (inclusive).
    If cutoff_date is None, uses all data (career).
    """
    if cutoff_date is None:
        bsel = bat_innings.copy()
        wsel = bow_innings.copy()
    else:
        bsel = bat_innings[bat_innings['match_date'] <= cutoff_date].copy()
        wsel = bow_innings[bow_innings['match_date'] <= cutoff_date].copy()

    # batting aggregates
    bat_agg = (bsel.groupby('player_id')
               .agg(batting_innings=('runs','count'),
                    batting_runs=('runs','sum'),
                    batting_balls=('balls','sum'),
                    batting_fours=('fours','sum'),
                    batting_sixes=('sixes','sum'),
                    batting_dismissals=('dismissed_count','sum'))
               .reset_index())
    bat_agg['bat_sr'] = np.where(bat_agg['batting_balls']>0, 100 * bat_agg['batting_runs'] / bat_agg['batting_balls'], np.nan)
    bat_agg['bat_avg'] = np.where(bat_agg['batting_dismissals']>0, bat_agg['batting_runs'] / bat_agg['batting_dismissals'], np.nan)
    bat_agg['boundary_pct'] = np.where(bat_agg['batting_balls']>0, 100*(bat_agg['batting_fours']+bat_agg['batting_sixes'])/bat_agg['batting_balls'], np.nan)
    bat_agg['dot_ball_pct'] = np.where(bat_agg['batting_balls']>0, 100*(1 - (bat_agg['batting_fours']+bat_agg['batting_sixes'])/bat_agg['batting_balls']), np.nan)

    # bowling aggregates
    bowl_agg = (wsel.groupby('player_id')
                .agg(bowling_innings=('runs_conceded','count'),
                     bowling_runs_conceded=('runs_conceded','sum'),
                     bowling_balls=('balls_bowled','sum'),
                     bowling_wickets=('wickets','sum'))
                .reset_index())
    bowl_agg['bow_econ'] = np.where(bowl_agg['bowling_balls']>0, 6 * bowl_agg['bowling_runs_conceded'] / bowl_agg['bowling_balls'], np.nan)
    bowl_agg['bow_sr'] = np.where(bowl_agg['bowling_wickets']>0, bowl_agg['bowling_balls'] / bowl_agg['bowling_wickets'], np.nan)
    bowl_agg['wickets_per_100_balls'] = np.where(bowl_agg['bowling_balls']>0, 100*bowl_agg['bowling_wickets']/bowl_agg['bowling_balls'], np.nan)

    # selection / availability features from playing_eleven
    pe_sel = pe.copy()
    if cutoff_date is not None:
        # restrict playing_eleven by match date
        pe_sel = pe_sel.merge(matches[['match_id','match_date']], on='match_id', how='left')
        pe_sel = pe_sel[pe_sel['match_date'] <= cutoff_date]

    sel_agg = (pe_sel.groupby('player_id')
               .agg(matches_played=('match_id','nunique'),
                    pct_matches_as_wk=('is_wk', 'mean'),
                    avg_batting_order=('batting_order','mean'))
               .reset_index())

    # combine everything
    df = pd.DataFrame({'player_id': pd.concat([bat_agg['player_id'], bowl_agg['player_id'], sel_agg['player_id']]).unique()})
    df = df.merge(players[['player_id','player_name','bowling_type','bowling_hand','batting_type','batting_hand','date_of_birth','wicket_keeper']], on='player_id', how='left')

    df = df.merge(bat_agg, on='player_id', how='left')
    df = df.merge(bowl_agg, on='player_id', how='left')
    df = df.merge(sel_agg, on='player_id', how='left')

    # mark capped or not (from earlier detection)
    df['is_capped'] = df['player_id'].isin(capped_player_ids)
    df['is_uncapped'] = ~df['is_capped']

    # age calculation (use date_of_birth if present)
    if 'date_of_birth' in df.columns:
        df['dob_parsed'] = df['date_of_birth'].apply(parse_date_flexible)
        df['age_days'] = (pd.to_datetime(cutoff_date if cutoff_date is not None else last_date) - df['dob_parsed']).dt.days
        df['age_years'] = (df['age_days'] / 365.25).round(2)
    else:
        df['age_years'] = np.nan

    # simple role heuristics:
    # - if avg_batting_order <=2 => opener
    # - if avg_batting_order between 3 and 5 => top/middle
    # - if avg_batting_order >5 => finisher
    # - if bowling_balls > batting_balls*0.3 => primarily bowler/allrounder
    def role_guess(row):
        try:
            abo = float(row['avg_batting_order']) if not pd.isna(row['avg_batting_order']) else np.nan
        except:
            abo = np.nan
        bat_balls = row.get('batting_balls', 0) if not pd.isna(row.get('batting_balls')) else 0
        bowl_balls = row.get('bowling_balls', 0) if not pd.isna(row.get('bowling_balls')) else 0
        if not pd.isna(abo):
            if abo <= 2.0:
                return 'opener'
            if abo <= 5.0:
                # check bowling involvement
                if bowl_balls > max(20, 0.3*bat_balls):
                    return 'batting allrounder'
                return 'top/middle'
            else:
                if bowl_balls > max(20, 0.3*bat_balls):
                    return 'finishing allrounder'
                return 'finisher'
        else:
            # no batting order -> probably bowler
            if bowl_balls > 0:
                if bowl_balls >= 200:
                    return 'bowler'
                return 'bowler/part-time'
            return 'unknown'

    df['role_guess'] = df.apply(role_guess, axis=1)

    # basic impact-like scores (normalized later by modeling step)
    # batting impact proxy: runs per 100 balls * sqrt(innings)
    df['bat_runs_per_100'] = np.where(df['batting_balls']>0, 100*df['batting_runs']/df['batting_balls'], np.nan)
    df['bat_impact_proxy'] = df['bat_runs_per_100'] * np.sqrt(df['batting_innings'].fillna(0))

    # bowling impact: wickets per 100 balls adjusted by economy
    df['bow_wickets_per_100'] = np.where(df['bowling_balls']>0, 100*df['bowling_wickets']/df['bowling_balls'], np.nan)
    df['bow_impact_proxy'] = df['bow_wickets_per_100'] / (df['bow_econ'].replace(0, np.nan))

    # visibility: played in IPL or competitions with high visibility (simple heuristic)
    high_vis = comps[comps['name'].str.lower().str.contains('ipl', na=False)]['id'].unique() if 'name' in comps.columns else []
    # build player->comp list from ptc
    p_comp = ptc.groupby('player_id')['comp_name' if 'comp_name' in ptc.columns else 'comp_id'].unique().to_dict() \
             if 'player_id' in ptc.columns else {}
    df['played_in_ipl'] = df['player_id'].apply(lambda pid: any('ipl' in str(x).lower() for x in p_comp.get(pid, [])))
    df['matches_played'] = df['matches_played'].fillna(0)

    # fill numeric NaNs with zeros where appropriate for modeling convenience
    num_cols = ['batting_innings','batting_runs','batting_balls','batting_fours','batting_sixes','batting_dismissals',
                'bat_sr','bat_avg','boundary_pct','dot_ball_pct',
                'bowling_innings','bowling_runs_conceded','bowling_balls','bowling_wickets',
                'bow_econ','bow_sr','wickets_per_100_balls','matches_played']
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return df

# create career, last12m, last24m features
player_career = agg_features(cutoff_date=None)
player_12m = agg_features(cutoff_date=last_date - pd.DateOffset(months=12))
player_24m = agg_features(cutoff_date=last_date - pd.DateOffset(months=24))

# rename columns to show windows
player_12m = player_12m.add_prefix('m12_')
player_24m = player_24m.add_prefix('m24_')
player_career = player_career.add_prefix('career_')

# unify by player_id column (strip prefixes)
player_12m = player_12m.rename(columns={'m12_player_id':'player_id'})
player_24m = player_24m.rename(columns={'m24_player_id':'player_id'})
player_career = player_career.rename(columns={'career_player_id':'player_id'})

# merge
print("Merging career + 12m + 24m features into final feature table...")
player_master = player_career.merge(player_12m, on='player_id', how='left').merge(player_24m, on='player_id', how='left')

# final cleanup, columns re-ordering
cols_order = ['player_id','player_name','is_capped','is_uncapped','role_guess','age_years',
              'career_matches_played','career_batting_runs','career_batting_innings','career_bat_sr','career_bat_avg',
              'career_bowling_wickets','career_bowling_innings','career_bow_econ','career_bow_sr',
              'm12_matches_played','m12_batting_runs','m12_batting_innings','m12_bat_sr',
              'm24_matches_played','m24_batting_runs','m24_batting_innings','m24_bat_sr',
              'played_in_ipl']
cols_existing = [c for c in cols_order if c in player_master.columns]
player_features = player_master[cols_existing].copy()

# Save outputs
print("Saving outputs to", OUT_DIR)
player_master.to_parquet(OUT_DIR / "player_master.parquet", index=False)
player_features.to_parquet(OUT_DIR / "player_features.parquet", index=False)
player_master.to_csv(OUT_DIR / "player_master.csv", index=False)
player_features.to_csv(OUT_DIR / "player_features.csv", index=False)

print("Done. Outputs:")
print(list(OUT_DIR.iterdir()))

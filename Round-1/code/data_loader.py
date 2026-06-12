import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

RR_RETAINED = [
    'yashasvi jaiswal', 'riyan parag', 'dhruv jurel', 'ravindra jadeja',
    'jofra archer', 'sam curran', 'shimron hetmyer', 'sandeep sharma',
    'wanindu hasaranga', 'nandre burger', 'shubham dubey', 'akash vashisht',
    'kuldeep sen', 'tanush kotian', 'yudhvir singh'
]

def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df

def load_all_raw():
    print("  Loading all data files...")
    players  = _load('players.csv')
    auction  = _load('auction_summary.csv')
    bids     = _load('bid_details.csv')
    comps    = _load('competitions.csv')
    matches  = _load('matches.csv')
    stats    = _load('player_match_stats.csv')
    mapping  = _load('player_competition_team_mapping.csv')
    print(f"  Stats rows loaded: {len(stats):,}")
    return players, auction, bids, comps, matches, stats, mapping

def filter_t20(comps, matches, stats):
    """Keep only T20-format matches and their player stats."""
    t20_ids   = comps[comps['match_type'].str.strip().str.upper() == 'T20']['comp_id']
    t20_match = matches[matches['comp_id'].isin(t20_ids)].copy()
    t20_match['match_date'] = pd.to_datetime(
        t20_match['match_date'], dayfirst=True, errors='coerce')

    stats_t20 = stats[stats['match_id'].isin(t20_match['match_id'])].copy()
    stats_t20['match_date'] = pd.to_datetime(
        stats_t20['match_date'], dayfirst=True, errors='coerce')

    # Coerce numeric columns
    for col in ['runs_scored','balls_faced','no_of_fours','no_of_sixes',
                'overs_bowled','runs_conceded','wicket_taken',
                'dot_balls_bowled','Dismissal_Status','batting_order']:
        stats_t20[col] = pd.to_numeric(stats_t20[col], errors='coerce').fillna(0)

    print(f"  T20 matches : {len(t20_match):,}  |  T20 stat rows: {len(stats_t20):,}")
    return t20_match, stats_t20

def _compute_age(dob_str):
    try:
        dob = pd.to_datetime(dob_str, dayfirst=True, errors='coerce')
        if pd.isna(dob): return np.nan
        return (datetime.now() - dob).days / 365.25
    except:
        return np.nan

def assign_role(row):
    """Assign primary T20 role from player profile + career stats."""
    is_wk  = int(row.get('is_wicket_keeper', 0)) == 1
    bt     = str(row.get('bowling_type',  '')).lower()
    bh     = str(row.get('bowling_hand',  '')).lower()
    avg_o  = row.get('batting_order_avg', np.nan)
    b_inn  = row.get('bowling_innings', 0)
    p_inn  = row.get('innings', 0)

    is_spin  = any(k in bt for k in ['spin','leg','off','wrist','finger','break','googly'])
    is_pace  = any(k in bt for k in ['fast','medium','pace','seam','swing'])
    is_bowl  = (is_spin or is_pace) and b_inn >= 5
    is_bat   = p_inn >= 5

    if is_wk:                      return 'WK-bat'
    if is_bat and is_bowl:         return 'Allrounder'
    if is_bowl:
        if 'leg' in bt or 'wrist' in bt or 'googly' in bt: return 'Wrist Spinner'
        if 'off'  in bt or 'finger' in bt:                 return 'Finger Spinner'
        if 'left' in bh:                                   return 'Left-arm Pacer'
        return 'Pacer'
    if not pd.isna(avg_o):
        if avg_o <= 2: return 'Top-order Batter'
        if avg_o <= 5: return 'Middle-order Batter'
        return 'Finisher'
    return 'Middle-order Batter'

def build_career_stats(stats_t20):
    """Aggregate career T20 batting + bowling stats per player."""
    # --- Batting ---
    bat = stats_t20.groupby('player_id').agg(
        innings           =('runs_scored',      'count'),
        total_runs        =('runs_scored',      'sum'),
        total_balls_faced =('balls_faced',      'sum'),
        total_fours       =('no_of_fours',      'sum'),
        total_sixes       =('no_of_sixes',      'sum'),
        dismissals        =('Dismissal_Status', lambda x: (x > 0).sum()),
        batting_order_avg =('batting_order',    'mean'),
        last_match        =('match_date',       'max'),
    ).reset_index()
    bat['t20_sr']       = bat['total_runs']  / bat['total_balls_faced'].replace(0, np.nan) * 100
    bat['t20_avg']      = bat['total_runs']  / bat['dismissals'].replace(0, np.nan)
    bat['boundary_pct'] = (bat['total_fours'] + bat['total_sixes']) / bat['total_balls_faced'].replace(0, np.nan)

    # --- Bowling ---
    bowl_df = stats_t20[stats_t20['overs_bowled'] > 0].copy()
    bowl = bowl_df.groupby('player_id').agg(
        bowling_innings     =('overs_bowled',     'count'),
        total_overs         =('overs_bowled',     'sum'),
        total_runs_conceded =('runs_conceded',    'sum'),
        total_wickets       =('wicket_taken',     'sum'),
        total_dots          =('dot_balls_bowled', 'sum'),
    ).reset_index()
    bowl['total_balls_bowled'] = (bowl['total_overs'] * 6).round().astype(int)
    bowl['t20_econ']    = bowl['total_runs_conceded'] / bowl['total_overs'].replace(0, np.nan)
    bowl['t20_bowl_sr'] = bowl['total_balls_bowled'] / bowl['total_wickets'].replace(0, np.nan)
    bowl['dot_pct']     = bowl['total_dots'] / bowl['total_balls_bowled'].replace(0, np.nan)
    return bat, bowl

def get_form_stats(stats_t20, days=180):
    """Recent form: last N days weighted stats."""
    cutoff = stats_t20['match_date'].max() - pd.Timedelta(days=days)
    recent = stats_t20[stats_t20['match_date'] >= cutoff]
    form = recent.groupby('player_id').agg(
        form_runs  =('runs_scored',   'sum'),
        form_balls =('balls_faced',   'sum'),
        form_wkts  =('wicket_taken',  'sum'),
        form_overs =('overs_bowled',  'sum'),
        form_runs_c=('runs_conceded', 'sum'),
        form_inn   =('runs_scored',   'count'),
    ).reset_index()
    form['form_sr']   = form['form_runs']   / form['form_balls'].replace(0, np.nan) * 100
    form['form_econ'] = form['form_runs_c'] / form['form_overs'].replace(0, np.nan)
    return form

def get_clutch_stats(stats_t20, t20_matches):
    """Performance in knockout/playoff stages."""
    clutch_kw = ['final','semi','qualifier','eliminator','playoff']
    ko_ids = t20_matches[
        t20_matches['stage'].str.lower().str.contains('|'.join(clutch_kw), na=False)
    ]['match_id'].tolist()
    if not ko_ids:
        return pd.DataFrame(columns=['player_id','clutch_runs','clutch_wkts','clutch_inn'])
    clutch = stats_t20[stats_t20['match_id'].isin(ko_ids)]
    return clutch.groupby('player_id').agg(
        clutch_runs=('runs_scored',  'sum'),
        clutch_wkts=('wicket_taken', 'sum'),
        clutch_inn =('runs_scored',  'count'),
    ).reset_index()

def build_master(players, auction, stats_t20, t20_matches):
    """Assemble the full master DataFrame for scoring."""
    bat, bowl   = build_career_stats(stats_t20)
    form        = get_form_stats(stats_t20)
    clutch      = get_clutch_stats(stats_t20, t20_matches)

    players = players.copy()
    players['age'] = players['date_of_birth'].apply(_compute_age)

    master = players.merge(bat,    on='player_id', how='left')
    master = master.merge(bowl,   on='player_id', how='left')
    master = master.merge(form,   on='player_id', how='left')
    master = master.merge(clutch, on='player_id', how='left')

    master['derived_role'] = master.apply(assign_role, axis=1)

    # Attach base price from most recent auction year
    base = (auction.sort_values('year')
                   .groupby('name', as_index=False)
                   .last()[['name','base_price','role','country']])
    base.columns = ['player_name','base_price_cr','auction_role','auction_country']
    # Convert to Crores: IPL base prices stored as integers (e.g. 20 = 20 Lakh → 0.20 Cr)
    base['base_price_cr'] = base['base_price_cr'] / 100

    master['player_name_lower'] = master['player_name'].str.strip().str.lower()
    base['player_name_lower']   = base['player_name'].str.strip().str.lower()
    master = master.merge(base.drop(columns='player_name'),
                          on='player_name_lower', how='left')

    # Flag overseas (not Indian)
    master['is_overseas'] = (~master['nationality'].str.lower().str.contains(
        'india', na=False)).astype(int)

    # Filter out RR retained players
    master['is_retained'] = master['player_name_lower'].isin(RR_RETAINED)

    print(f"  Master table: {len(master):,} players  "
          f"| with auction history: {master['base_price_cr'].notna().sum():,}")
    return master, bat, bowl

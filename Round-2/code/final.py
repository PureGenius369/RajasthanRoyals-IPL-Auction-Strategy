"""
Enhanced Cricket Talent Scout - Top 40 Uncapped Indian Players
================================================================
Identifies promising uncapped Indian cricket players using:
- Performance metrics (batting/bowling statistics)
- Career progression trends
- Age and potential context
- Multi-factor composite scoring
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING ====================
print("Loading datasets...")
players = pd.read_csv("players.csv")
playing11 = pd.read_csv("playing_eleven.csv")
bbb = pd.read_csv("india_matches_ball_by_ball.csv")
matches = pd.read_csv("matches.csv")
mapping = pd.read_csv("player_team_competition_mapping.csv")

print(f"✓ Loaded {len(matches)} matches | {len(bbb):,} balls | {len(players)} players\n")

# ==================== DATA PREPARATION ====================
bbb['date'] = pd.to_datetime(bbb['delivery_ist_time'], errors='coerce')
bbb['year'] = bbb['date'].dt.year
bbb['extras'] = bbb['extras_1'].fillna(0) + bbb['extras_2'].fillna(0)

# Known capped players (comprehensive list)
CAPPED_PLAYERS = {
    # Current generation stars
    "Yashasvi Jaiswal", "Shubman Gill", "Ishan Kishan", "Tilak Varma", "Rinku Singh",
    "Sai Sudharsan", "Abhishek Sharma", "Ravi Bishnoi", "Arshdeep Singh", "Umran Malik",
    "Harshit Rana", "Nitish Kumar Reddy", "Dhruv Jurel", "Washington Sundar", "Varun Chakaravarthy",
    
    # Established internationals
    "Hardik Pandya", "Axar Patel", "Shivam Dube", "Kuldeep Yadav", "Bhuvneshwar Kumar",
    "Harshal Patel", "Deepak Chahar", "Yuzvendra Chahal", "Jasprit Bumrah", "Avesh Khan",
    "Mukesh Kumar", "Mohammed Shami", "Mohammed Siraj", "Shardul Thakur", "Prasidh Krishna",
    "Ravindra Jadeja", "Ravichandran Ashwin", "KL Rahul", "Rohit Sharma", "Virat Kohli",
    "Suryakumar Yadav", "Rishabh Pant", "Sanju Samson", "Shreyas Iyer", "Deepak Hooda",
    "Krunal Pandya", "Rahul Tewatia", "Venkatesh Iyer", "Rahul Chahar", "Khaleel Ahmed",
    "Jaydev Unadkat", "T Natarajan", "Navdeep Saini", "Kamlesh Nagarkoti", "Shivam Mavi",
    "Prithvi Shaw", "Devdutt Padikkal", "Ruturaj Gaikwad", "Mayank Agarwal", "Hanuma Vihari"
}

# ==================== IDENTIFY INDIAN UNCAPPED PLAYERS ====================
print("Identifying Indian uncapped players...")
print(f"Total players in players.csv: {len(players)}")

# Get all players who actually played
players_in_bbb = (set(bbb['striker_name'].dropna().unique()) | 
                  set(bbb['bowler_name'].dropna().unique()))
print(f"Players who actually played in ball-by-ball data: {len(players_in_bbb)}")

# PRIMARY FILTER: Use nationality column (most reliable)
indian_players_set = set()

if 'nationality' in players.columns:
    # Get players with India nationality
    indian_nationality = players[players['nationality'] == 'India']['player_name'].dropna()
    print(f"Players with nationality='India' in players.csv: {len(indian_nationality)}")
    
    indian_players_set = set(indian_nationality) & players_in_bbb
    print(f"Indian players who ALSO played in ball-by-ball: {len(indian_players_set)}")
else:
    print("  WARNING: No nationality column found!")

# SECONDARY FILTER: Cross-reference with team mapping
if 'team_name' in mapping.columns:
    indian_from_mapping = set(mapping[mapping['team_name'] == 'India']['player_name'])
    print(f"Players mapped to 'India' team: {len(indian_from_mapping)}")
    
    mapped_count = len(indian_from_mapping & players_in_bbb)
    print(f"Mapped India players who played: {mapped_count}")
    
    # Only ADD players from mapping if nationality column doesn't exist
    if 'nationality' not in players.columns:
        indian_players_set = indian_from_mapping & players_in_bbb

# If we still have no Indian players, this is a data issue
if len(indian_players_set) == 0:
    print("\n⚠️  ERROR: No Indian players identified! Check your data sources.")
    print("  Possible issues:")
    print("    - 'nationality' column missing or incorrect")
    print("    - Player names don't match between CSVs")
    exit()

# Create dataframe with only confirmed Indian players
indian_players_df = players[players['player_name'].isin(indian_players_set)].copy()
indian_players_df['is_capped'] = indian_players_df['player_name'].isin(CAPPED_PLAYERS)

print(f"\nIndian players breakdown:")
print(f"  - Total Indian players: {len(indian_players_df)}")
print(f"  - Capped (in our list): {indian_players_df['is_capped'].sum()}")

# Filter to uncapped only
uncapped = indian_players_df[~indian_players_df['is_capped']].copy()
print(f"  - Uncapped: {len(uncapped)}")

# Show sample of uncapped players
if len(uncapped) > 0:
    print(f"\nSample uncapped players: {', '.join(uncapped['player_name'].head(10).tolist())}")
print()

# ==================== BATTING METRICS ====================
def calculate_batting_stats(df):
    """Calculate comprehensive batting statistics"""
    
    # Per-innings stats
    innings_stats = df.groupby(['match_id', 'innings', 'striker_name']).agg(
        runs=('runs', 'sum'),
        balls=('striker_name', 'count'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots=('runs', lambda x: (x == 0).sum())
    ).reset_index()
    
    # Calculate strike rate per innings
    innings_stats['sr'] = innings_stats['runs'] / innings_stats['balls'].replace(0, 1) * 100
    
    # Player aggregations
    player_stats = innings_stats.groupby('striker_name').agg(
        innings=('match_id', 'nunique'),
        total_runs=('runs', 'sum'),
        total_balls=('balls', 'sum'),
        total_fours=('fours', 'sum'),
        total_sixes=('sixes', 'sum'),
        total_dots=('dots', 'sum'),
        avg_sr=('sr', 'mean'),
        max_sr=('sr', 'max'),
        highest_score=('runs', 'max')
    ).reset_index()
    
    player_stats.rename(columns={'striker_name': 'player_name'}, inplace=True)
    
    # Calculate advanced metrics
    player_stats['batting_avg'] = player_stats['total_runs'] / player_stats['innings'].replace(0, 1)
    player_stats['batting_sr'] = player_stats['total_runs'] / player_stats['total_balls'].replace(0, 1) * 100
    player_stats['boundary_pct'] = (player_stats['total_fours'] + player_stats['total_sixes']) / player_stats['total_balls'].replace(0, 1) * 100
    player_stats['dot_ball_pct'] = player_stats['total_dots'] / player_stats['total_balls'].replace(0, 1) * 100
    
    return player_stats

# ==================== BOWLING METRICS ====================
def calculate_bowling_stats(df):
    """Calculate comprehensive bowling statistics"""
    
    # Wickets (excluding run-outs)
    wickets_df = df[
        (df['is_wicket'] == 1) & (df['dismissal_type'] != 'run out')
    ].groupby(['match_id', 'bowler_name']).size().reset_index(name='wickets')
    
    # Per-match bowling
    match_stats = df.groupby(['match_id', 'bowler_name']).agg(
        runs_conceded=('runs', 'sum'),
        extras_conceded=('extras', 'sum'),
        balls_bowled=('bowler_name', 'count'),
        dots=('runs', lambda x: (x == 0).sum())
    ).reset_index()
    
    match_stats['total_runs'] = match_stats['runs_conceded'] + match_stats['extras_conceded']
    match_stats = match_stats.merge(wickets_df, on=['match_id', 'bowler_name'], how='left').fillna({'wickets': 0})
    
    # Player aggregations
    player_stats = match_stats.groupby('bowler_name').agg(
        matches_bowled=('match_id', 'nunique'),
        wickets=('wickets', 'sum'),
        runs_conceded=('total_runs', 'sum'),
        balls_bowled=('balls_bowled', 'sum'),
        dots_bowled=('dots', 'sum')
    ).reset_index()
    
    player_stats.rename(columns={'bowler_name': 'player_name'}, inplace=True)
    
    # Calculate advanced metrics
    player_stats['overs'] = player_stats['balls_bowled'] / 6
    player_stats['economy'] = player_stats['runs_conceded'] / player_stats['overs'].replace(0, 1)
    player_stats['bowling_avg'] = player_stats['runs_conceded'] / player_stats['wickets'].replace(0, 999)
    player_stats['bowling_sr'] = player_stats['balls_bowled'] / player_stats['wickets'].replace(0, 999)
    player_stats['dot_ball_pct'] = player_stats['dots_bowled'] / player_stats['balls_bowled'].replace(0, 1) * 100
    
    return player_stats

print("Calculating batting statistics...")
batting_stats = calculate_batting_stats(bbb)

print("Calculating bowling statistics...")
bowling_stats = calculate_bowling_stats(bbb)

# ==================== MERGE AND FILTER ====================
print("Merging statistics with candidates...")
candidates = uncapped.merge(batting_stats, on='player_name', how='left')
candidates = candidates.merge(bowling_stats, on='player_name', how='left', suffixes=('', '_bowl'))

# Fill NaN for players who never batted/bowled
numeric_cols = candidates.select_dtypes(include=[np.number]).columns
candidates[numeric_cols] = candidates[numeric_cols].fillna(0)

print(f"Candidates before participation filter: {len(candidates)}")

# Show participation stats
print("\nParticipation summary:")
print(f"  - Players with ≥2 innings: {(candidates['innings'] >= 2).sum()}")
print(f"  - Players with ≥3 wickets: {(candidates['wickets'] >= 3).sum()}")
print(f"  - Players with either: {((candidates['innings'] >= 2) | (candidates['wickets'] >= 3)).sum()}")

# Filter: minimum participation (VERY LOW thresholds for small dataset)
MIN_INNINGS = 2  # Very low threshold - just need some participation
MIN_WICKETS = 3  # Very low threshold - just need some participation
candidates = candidates[
    (candidates['innings'] >= MIN_INNINGS) | (candidates['wickets'] >= MIN_WICKETS)
]

print(f"\n✓ {len(candidates)} candidates meet minimum participation criteria")
print(f"  (≥{MIN_INNINGS} innings OR ≥{MIN_WICKETS} wickets)")

if len(candidates) > 0:
    print(f"\nQualifying candidates:")
    for idx, row in candidates.iterrows():
        print(f"  - {row['player_name']}: {int(row['innings'])} innings, {int(row['total_runs'])} runs, {int(row['wickets'])} wickets")
print()

if len(candidates) == 0:
    print("⚠️  No candidates found. Adjust participation thresholds.")
    exit()

# ==================== PROGRESSION ANALYSIS ====================
def calculate_progression(player_name, df, metric_type='batting'):
    """Calculate career trajectory using regression slope"""
    
    if metric_type == 'batting':
        player_data = df[df['striker_name'] == player_name]
        if len(player_data) < 20:  # Need sufficient data
            return 0
        
        yearly = player_data.groupby(player_data['date'].dt.year).agg(
            runs=('runs', 'sum'),
            balls=('striker_name', 'count')
        )
        yearly['metric'] = yearly['runs'] / yearly['balls'].replace(0, 1) * 100
        
    else:  # bowling
        player_data = df[df['bowler_name'] == player_name]
        if len(player_data) < 30:
            return 0
        
        yearly = player_data.groupby(player_data['date'].dt.year).agg(
            runs=('runs', 'sum'),
            extras=('extras', 'sum'),
            balls=('bowler_name', 'count')
        )
        yearly['metric'] = (yearly['runs'] + yearly['extras']) / (yearly['balls'] / 6).replace(0, 1)
    
    if len(yearly) < 3:
        return 0
    
    # Fit linear regression
    X = np.arange(len(yearly)).reshape(-1, 1)
    y = yearly['metric'].values
    
    try:
        slope = LinearRegression().fit(X, y).coef_[0]
        # Positive slope good for batting, negative good for bowling
        return max(slope, 0) if metric_type == 'batting' else max(-slope, 0)
    except:
        return 0

print("Analyzing career progression...")
candidates['progression'] = 0.0

for idx, row in candidates.iterrows():
    player = row['player_name']
    
    if row['total_runs'] > 150:  # Significant batting
        candidates.at[idx, 'progression'] = calculate_progression(player, bbb, 'batting')
    elif row['wickets'] > 5:  # Significant bowling
        candidates.at[idx, 'progression'] = calculate_progression(player, bbb, 'bowling')

# ==================== COMPOSITE SCORING ====================
print("Computing composite scores...\n")

# 1. Performance Score
candidates['batting_impact'] = (
    candidates['total_runs'] * 
    (candidates['batting_sr'] / 100) * 
    (1 + candidates['boundary_pct'] / 100)
)

candidates['bowling_impact'] = (
    candidates['wickets'] * 
    (10 - candidates['economy'].clip(upper=12)) * 
    (1 + candidates['dot_ball_pct'] / 100)
)

candidates['total_impact'] = candidates['batting_impact'] + candidates['bowling_impact']

# 2. Age calculation and context
if 'date_of_birth' in candidates.columns:
    candidates['date_of_birth'] = pd.to_datetime(
        candidates['date_of_birth'], 
        format='%d/%m/%y', 
        errors='coerce'
    )
    candidates['age'] = (pd.Timestamp.now() - candidates['date_of_birth']).dt.days / 365.25
else:
    candidates['age'] = 25

# Youth bonus
candidates['youth_bonus'] = np.where(candidates['age'] < 23, 1.2,
                           np.where(candidates['age'] < 25, 1.1, 1.0))

# 3. Consistency metric
candidates['consistency'] = 0.0
for idx, row in candidates.iterrows():
    player = row['player_name']
    player_innings = bbb[bbb['striker_name'] == player].groupby(['match_id', 'innings'])['runs'].sum()
    
    if len(player_innings) >= 5:
        cv = player_innings.std() / player_innings.mean() if player_innings.mean() > 0 else 2
        cv_capped = min(cv, 2)  # Cap coefficient of variation at 2
        candidates.at[idx, 'consistency'] = 1 - (cv_capped / 2)

# 4. Normalize components
scaler = MinMaxScaler()

candidates['P_performance'] = scaler.fit_transform(
    (candidates['total_impact'] * candidates['youth_bonus']).values.reshape(-1, 1)
).ravel()

candidates['R_progression'] = scaler.fit_transform(
    candidates[['progression']].fillna(0)
)

candidates['C_consistency'] = scaler.fit_transform(
    candidates[['consistency']].fillna(0)
)

# 5. Final composite score
WEIGHTS = {'performance': 0.50, 'progression': 0.30, 'consistency': 0.20}

candidates['COMPOSITE_SCORE'] = (
    WEIGHTS['performance'] * candidates['P_performance'] +
    WEIGHTS['progression'] * candidates['R_progression'] +
    WEIGHTS['consistency'] * candidates['C_consistency']
)

# ==================== CAP PREDICTION MODEL ====================
print("Training cap prediction model...")

# Feature engineering
X_features = candidates[[
    'total_impact', 'batting_sr', 'economy', 'age', 
    'innings', 'wickets', 'consistency', 'progression'
]].fillna(0)

# Create synthetic target (top performers likely to be capped)
threshold = candidates['COMPOSITE_SCORE'].quantile(0.80)
y_synthetic = (candidates['COMPOSITE_SCORE'] > threshold).astype(int)

# Train model
cap_model = LogisticRegression(max_iter=1000, random_state=42)
cap_model.fit(X_features, y_synthetic)

candidates['CAP_PROBABILITY'] = cap_model.predict_proba(X_features)[:, 1]

# Predict likely cap year
candidates['PREDICTED_CAP_YEAR'] = pd.cut(
    candidates['CAP_PROBABILITY'],
    bins=[0, 0.35, 0.60, 0.80, 1.0],
    labels=['2029+', '2028', '2027', '2026']
)

# ==================== GENERATE TOP 40 ====================
print("Selecting Top 40 candidates...\n")

# Sort by composite score
top_candidates = candidates.sort_values('COMPOSITE_SCORE', ascending=False).reset_index(drop=True)

# CRITICAL: Double-check nationality to ensure only Indians
if 'nationality' in players.columns:
    # Get nationality for verification
    player_nationality = players[['player_name', 'nationality']].drop_duplicates()
    
    nationality_check = top_candidates.merge(
        player_nationality, 
        on='player_name', 
        how='left',
        suffixes=('', '_check')
    )
    
    # Check if any non-Indian players slipped through
    if 'nationality' in nationality_check.columns:
        non_indian = nationality_check[nationality_check['nationality'] != 'India']
        
        if len(non_indian) > 0:
            print(f"⚠️  Removing {len(non_indian)} non-Indian players:")
            print(f"  {', '.join(non_indian['player_name'].head(10).tolist())}")
            top_candidates = top_candidates[top_candidates['player_name'].isin(
                nationality_check[nationality_check['nationality'] == 'India']['player_name']
            )]

# Final verification: remove any capped players that slipped through
top40 = top_candidates[~top_candidates['player_name'].isin(CAPPED_PLAYERS)].head(40).reset_index(drop=True)

if len(top40) < 40:
    print(f"\n⚠️  WARNING: Only {len(top40)} qualifying candidates found (target: 40)")
    print("  Consider lowering minimum participation thresholds.")
    print(f"  Current filters: MIN_INNINGS={MIN_INNINGS}, MIN_WICKETS={MIN_WICKETS}")

top40['RANK'] = range(1, len(top40) + 1)

# ==================== EXPORT RESULTS ====================
output_columns = [
    'RANK', 'player_name', 'age', 'COMPOSITE_SCORE', 
    'total_impact', 'batting_sr', 'economy',
    'innings', 'total_runs', 'wickets', 
    'CAP_PROBABILITY', 'PREDICTED_CAP_YEAR',
    'progression', 'consistency'
]

output_file = "RR_SupeRR_Selector_Top40_Enhanced.csv"

# Try to save, handle permission errors
try:
    top40[output_columns].to_csv(output_file, index=False)
    print(f"\n✅ Full report saved: {output_file}")
except PermissionError:
    # File is open - try alternative name
    import time
    timestamp = int(time.time())
    output_file = f"RR_SupeRR_Selector_Top40_Enhanced_{timestamp}.csv"
    top40[output_columns].to_csv(output_file, index=False)
    print(f"\n✅ Full report saved: {output_file}")
    print(f"⚠️  (Original file was locked - saved with timestamp)")

# ==================== DISPLAY SUMMARY ====================
print("=" * 90)
print("🏏 RAJASTHAN ROYALS TALENT SCOUT - TOP 40 UNCAPPED INDIAN PLAYERS 🏏")
print("=" * 90)
print("\n📊 METHODOLOGY:")
print("  • Performance (50%): Batting/bowling impact with youth bonus")
print("  • Progression (30%): Career trajectory analysis")
print("  • Consistency (20%): Performance stability\n")

print("🎯 TOP 20 PROSPECTS:")
print("-" * 90)
display_cols = ['RANK', 'player_name', 'age', 'COMPOSITE_SCORE', 'PREDICTED_CAP_YEAR']
print(top40[display_cols].head(20).to_string(index=False))
print("-" * 90)

print(f"\n✅ Full report saved: {output_file}")
print(f"📈 Total candidates analyzed: {len(candidates)}")
print(f"🏆 Top 40 selected using multi-factor composite scoring\n")

# Statistics summary
print("📋 CANDIDATE BREAKDOWN:")
print(f"  • Under 23: {(top40['age'] < 23).sum()} players")
print(f"  • Predicted cap by 2026: {(top40['PREDICTED_CAP_YEAR'] == '2026').sum()} players")
print(f"  • Avg composite score: {top40['COMPOSITE_SCORE'].mean():.3f}")
print(f"  • Batters (>200 runs): {(top40['total_runs'] > 200).sum()}")
print(f"  • Bowlers (>10 wickets): {(top40['wickets'] > 10).sum()}")
print("\n" + "=" * 90)
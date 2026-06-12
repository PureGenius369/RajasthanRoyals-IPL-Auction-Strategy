import pandas as pd
import numpy as np

def compute_risk(df: pd.DataFrame, stats_t20: pd.DataFrame,
                 t20_matches: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Age risk (higher for 33+)
    age = df['age'].fillna(28)
    df['age_risk'] = np.where(age >= 34, 0.35,
                    np.where(age >= 31, 0.20,
                    np.where(age <= 24, 0.05, 0.10)))

    # 2. Injury risk: proxy via games played vs. max possible
    total_matches = len(t20_matches['match_id'].unique())
    games_played  = stats_t20.groupby('player_id')['match_id'].nunique().rename('games_played')
    df = df.merge(games_played, on='player_id', how='left')
    df['games_played'] = df['games_played'].fillna(0)

    # Availability ratio: played / possible (capped at 1.0)
    max_possible     = min(total_matches, 200)   # cap for normalisation
    avail_ratio      = (df['games_played'] / max_possible).clip(0, 1)
    df['injury_risk']= (1 - avail_ratio) * 0.30  # low games → higher injury proxy

    # 3. Availability risk: overseas players have higher schedule conflict
    df['availability_risk'] = np.where(df['is_overseas'] == 1, 0.20, 0.08)

    # 4. Volatility risk: high std in recent batting SR
    vol = (stats_t20[stats_t20['balls_faced'] > 0]
           .groupby('player_id')['strike_rate']
           .std()
           .rename('sr_std'))
    df = df.merge(vol, on='player_id', how='left')
    df['sr_std']        = df['sr_std'].fillna(0)
    df['volatility_risk']= (df['sr_std'] / 80).clip(0, 0.30)

    # Combined risk
    df['risk'] = (0.35 * df['injury_risk']
                + 0.30 * df['availability_risk']
                + 0.20 * df['volatility_risk']
                + 0.15 * df['age_risk']).clip(0, 0.40)

    df['final_score'] = (df['scarcity_imp'] * (1 - df['risk'])
                         * df['venue_fit']).fillna(0)
    return df
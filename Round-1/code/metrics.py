import pandas as pd
import numpy as np

def minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]."""
    rng = series.max() - series.min()
    return (series - series.min()) / rng if rng != 0 else pd.Series(0.0, index=series.index)

def compute_batting_impact(df: pd.DataFrame) -> pd.Series:
    sr     = minmax(df['t20_sr'].fillna(df['t20_sr'].median()))
    avg    = minmax(df['t20_avg'].fillna(df['t20_avg'].median()))
    bnd    = minmax(df['boundary_pct'].fillna(0))
    dot    = minmax(df['dot_pct_bat'].fillna(0))
    form   = minmax(df['form_score'].fillna(0))
    clutch = minmax(df['clutch_score'].fillna(0))
    return (0.30*sr + 0.15*avg + 0.20*bnd
            + 0.10*(1 - dot) + 0.15*form + 0.10*clutch)

def compute_bowling_impact(df: pd.DataFrame) -> pd.Series:
    econ     = minmax(df['t20_econ'].fillna(df['t20_econ'].median()))
    sr_bowl  = minmax(df['t20_bowl_sr'].fillna(df['t20_bowl_sr'].median()))
    dot      = minmax(df['dot_pct_bowl'].fillna(0))
    death    = minmax(df['death_skill'].fillna(0))
    form     = minmax(df['form_score'].fillna(0))
    pressure = minmax(df['pressure_score'].fillna(0))
    return (0.25*(1-econ) + 0.20*(1-sr_bowl) + 0.15*dot
            + 0.25*death + 0.10*form + 0.05*pressure)

def compute_venue_fit(df: pd.DataFrame) -> pd.Series:
    phase = minmax(df['phase_match'].fillna(0.5))
    surf  = minmax(df['surface_skill'].fillna(0.5))
    dew   = minmax(df['dew_adaptability'].fillna(0.5))
    return 0.4*phase + 0.3*surf + 0.3*dew

def compute_impact_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['batting_imp'] = compute_batting_impact(df)
    df['bowling_imp'] = compute_bowling_impact(df)
    df['venue_fit']   = compute_venue_fit(df)

    is_ar     = df['role'].str.contains('AR|allrounder', case=False, na=False)
    is_bowler = df['role'].str.contains('pacer|spinner|bowler', case=False, na=False)
    is_batter = ~is_bowler

    df['impact_score'] = (
        df['batting_imp'] * is_batter.astype(float)
        + df['bowling_imp'] * is_bowler.astype(float)
        + (df['batting_imp'] + df['bowling_imp']) * is_ar.astype(float)
        + df['venue_fit'] * 0.15
    )
    return df
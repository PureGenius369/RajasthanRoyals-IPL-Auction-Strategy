import pandas as pd

SCARCITY_WEIGHTS = {
    'indian death bowler'       : 1.40,
    'indian wrist spinner'      : 1.35,
    'indian left-arm pacer'     : 1.30,
    'indian middle-order batter': 1.20,
    'indian finisher'           : 1.20,
    'indian top-order batter'   : 1.00,
    'overseas batter'           : 1.00,
    'overseas allrounder'       : 1.00,
    'common domestic'           : 0.90,
}

INFLATION_FACTOR = 1.15

COMPETITOR_DEMAND = {
    'cameron green' : 1.40,
    'ravi bishnoi'  : 1.50,
    'mohit sharma'  : 1.20,
    'rahul tripathi': 1.30,
}

def get_scarcity_weight(row: pd.Series) -> float:
    role        = str(row.get('role', '')).lower()
    nationality = str(row.get('nationality', '')).upper()
    if nationality == 'IND':
        if 'death' in role and ('pacer' in role or 'bowler' in role):
            return SCARCITY_WEIGHTS['indian death bowler']
        if 'wrist' in role or 'leg' in role:
            return SCARCITY_WEIGHTS['indian wrist spinner']
        if 'left' in role and 'pacer' in role:
            return SCARCITY_WEIGHTS['indian left-arm pacer']
        if 'middle' in role:
            return SCARCITY_WEIGHTS['indian middle-order batter']
        if 'finish' in role:
            return SCARCITY_WEIGHTS['indian finisher']
        if 'top' in role or 'open' in role:
            return SCARCITY_WEIGHTS['indian top-order batter']
    return SCARCITY_WEIGHTS['common domestic']

def compute_expected_price(row: pd.Series) -> float:
    base = float(row.get('base_price_cr', 0.5))
    role = str(row.get('role', '')).lower()
    name = str(row.get('name', '')).lower()
    rf   = 1.3 if any(k in role for k in ['death', 'wrist', 'leg']) else 1.0
    sf   = 1.2 if float(row.get('is_star', 0)) else 1.0
    cd   = COMPETITOR_DEMAND.get(name, 1.0)
    return round(base * INFLATION_FACTOR * rf * sf * cd, 3)

def apply_scarcity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['scarcity_weight'] = df.apply(get_scarcity_weight, axis=1)
    df['scarcity_imp']    = df['impact_score'] * df['scarcity_weight']
    df['expected_price']  = df.apply(compute_expected_price, axis=1)
    df['value_index']     = df['scarcity_imp'] / df['expected_price'].replace(0, 1)
    return df
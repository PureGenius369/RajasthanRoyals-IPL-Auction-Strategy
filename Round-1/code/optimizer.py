import pandas as pd
from pulp import (LpProblem, LpVariable, LpMaximize,
                  lpSum, LpBinary, value, PULP_CBC_CMD)

PURSE        = 16.05
MAX_OVERSEAS = 1
MAX_PLAYERS  = 9

MAX_BIDS = {
    'cameron green'          : 5.50,
    'ravi bishnoi'           : 3.50,
    'rahul tripathi'         : 1.50,
    'mohit sharma'           : 1.20,
    'akash madhwal'          : 1.00,
    'rajvardhan hangargekar' : 0.95,
    'abhinav manohar'        : 0.80,
    'mahipal lomror'         : 0.80,
    'srikar bharat'          : 0.75,
}

def run_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['name_lower'] = df['name'].str.lower()
    df['max_bid']    = df['name_lower'].map(MAX_BIDS).fillna(
        df['base_price_cr'] * 1.5
    )

    prob = LpProblem("RR_Auction_Optimizer", LpMaximize)
    x    = [LpVariable(f"x_{i}", cat=LpBinary) for i in df.index]

    # Objective
    prob += lpSum(df.loc[i, 'final_score'] * x[idx]
                  for idx, i in enumerate(df.index))

    # Purse constraint
    prob += lpSum(df.loc[i, 'max_bid'] * x[idx]
                  for idx, i in enumerate(df.index)) <= PURSE

    # Overseas constraint
    os_idx = df.index[df['nationality'].str.upper() == 'OS'].tolist()
    prob  += lpSum(x[df.index.get_loc(i)] for i in os_idx) <= MAX_OVERSEAS

    # Slot constraint
    prob += lpSum(x) <= MAX_PLAYERS

    prob.solve(PULP_CBC_CMD(msg=0))

    selected = [df.index[idx] for idx, xi in enumerate(x) if value(xi) == 1]
    result   = df.loc[selected, [
        'name', 'role', 'nationality',
        'base_price_cr', 'max_bid',
        'final_score', 'scarcity_weight', 'risk'
    ]].copy()
    result = result.sort_values('final_score', ascending=False).reset_index(drop=True)
    result.index += 1
    return result
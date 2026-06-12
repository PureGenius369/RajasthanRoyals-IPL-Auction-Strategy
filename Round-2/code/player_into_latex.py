import pandas as pd

# --- Load both CSVs ---
df_year = pd.read_csv("output/rr2_top40_uncapped_with_predicted_year.csv")
df_ens  = pd.read_csv("output/rr2_top40_uncapped_probs_final_ensemble.csv")


# ---------- Helper functions ----------
def pick_col(df, possible_names, label=""):
    """
    Return the first column name from possible_names that exists in df.
    Raise a clear error if none are found.
    """
    for c in possible_names:
        if c in df.columns:
            return c
    raise ValueError(
        f"No suitable {label} column found in {df.columns.tolist()} "
        f"(tried: {possible_names})"
    )


def find_merge_key(df1, df2):
    """
    Try to find a common key column to merge on: player_id first, else player_name.
    """
    # try id-based merge
    for cand in ["player_id", "Player ID", "id"]:
        if cand in df1.columns and cand in df2.columns:
            return cand

    # else try name-based merge
    for cand in ["player_name", "Player", "name"]:
        if cand in df1.columns and cand in df2.columns:
            return cand

    raise ValueError(
        "Could not find a common key to merge on. "
        f"df_year cols: {df1.columns.tolist()}, df_ens cols: {df2.columns.tolist()}"
    )


# ---------- Choose columns from each file ----------
# From with_predicted_year file:
name_col_year  = pick_col(df_year, ["player_name", "Player", "name"], label="player name (year)")
prob_col_year  = pick_col(df_year, ["final_cap_prob"], label="probability")
year_col_year  = pick_col(df_year, ["predicted_cap_year", "cap_year"], label="cap year")
match_col_year = pick_col(df_year, ["career_matches_played", "career_matches", "matches", "matches_played"],
                          label="matches")
age_col_year   = pick_col(df_year, ["career_age_years", "age"], label="age")

# From ensemble file:
role_col_ens   = pick_col(df_ens, ["career_role_guess", "condensed_role", "role"],
                          label="role")
venue_col_ens  = pick_col(df_ens, ["venue_fit_score", "venue_fit"], label="venue fit")

# ---------- Merge the two tables ----------
merge_key = find_merge_key(df_year, df_ens)

df_merged = df_year.merge(
    df_ens[[merge_key, role_col_ens, venue_col_ens]],
    on=merge_key,
    how="left",
    suffixes=("", "_ens")
)

# ---------- Sort by probability (descending) ----------
df_merged = df_merged.sort_values(prob_col_year, ascending=False).reset_index(drop=True)

# ---------- Print LaTeX rows ----------
for i, row in df_merged.iterrows():
    rank = i + 1

    name = str(row[name_col_year]).replace("&", "\\&")
    role = str(row[role_col_ens]).replace("&", "\\&")

    prob = f"{float(row[prob_col_year]):.2f}"
    year = int(row[year_col_year])

    matches = int(row[match_col_year])
    age = f"{float(row[age_col_year]):.1f}"

    venue_val = float(row[venue_col_ens]) if not pd.isna(row[venue_col_ens]) else 0.0
    venue = f"{venue_val:.2f}"

    print(f"{rank} & {name} & {role} & {prob} & {year} & {matches} & {age} & {venue} \\\\")

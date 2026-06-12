import pandas as pd

df = pd.read_parquet("output/player_features_with_phase.parquet")
df.to_csv("output/player_features_with_phase.csv", index=False)

print("Converted parquet to CSV successfully.")

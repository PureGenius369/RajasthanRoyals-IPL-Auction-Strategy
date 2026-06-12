import pandas as pd

bbb = pd.read_csv("players.csv")
print("Ball-by-ball columns:")
print(bbb.columns.tolist())
print("\nSample data:")
print(bbb.head())
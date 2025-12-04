import pandas as pd

df = pd.read_csv('house_prices.csv')

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df) // 10

for i in range(10):
    start = i * n
    end = (i + 1) * n if i < 9 else len(df)
    subset = df.iloc[start:end]
    subset.to_csv(f'house_prices_part_{i + 1}.csv', index=False)

print("Dataset divided into 10 parts and saved as CSV files.")
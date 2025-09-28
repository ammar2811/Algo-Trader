# Algorithmic Trading Strategy Demo
This notebook demonstrates a simple algorithmic trading system with enhancements:
1. Transaction costs
2. Risk metrics (Sharpe ratio, max drawdown)
3. Parameter tuning for SMA crossovers
4. Multi-asset portfolio backtesting
5. Alternative strategies (momentum, mean reversion)
6. Visualization with trading signals

---

## Step 1: Imports and Data

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# Download 2 years of Apple stock data
data = yf.download("AAPL", start="2023-01-01", end="2025-01-01")

# Calculate moving averages
data["SMA50"] = data["Close"].rolling(window=50).mean()
data["SMA200"] = data["Close"].rolling(window=200).mean()

# Generate signals
data["Signal"] = (data["SMA50"] > data["SMA200"]).astype(int)
data["Position"] = data["Signal"].diff()
data.head()
```

---

## Step 2: Add Transaction Costs

```python
transaction_cost = 0.001  # 0.1% per trade

data["Returns"] = data["Close"].pct_change()
data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]

# Subtract cost when a trade happens
data.loc[data["Position"] != 0, "Strategy_Returns"] -= transaction_cost
```

---

## Step 3: Risk Metrics

```python
# Sharpe ratio
sharpe = (data["Strategy_Returns"].mean() / data["Strategy_Returns"].std()) * np.sqrt(252)

# Max drawdown
cumulative = (1 + data["Strategy_Returns"]).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_drawdown)
```

---

## Step 4: Parameter Tuning

```python
def backtest_sma(short, long, ticker="AAPL"):
    df = yf.download(ticker, start="2023-01-01", end="2025-01-01")
    df[f"SMA{short}"] = df["Close"].rolling(short).mean()
    df[f"SMA{long}"] = df["Close"].rolling(long).mean()
    df["Signal"] = (df[f"SMA{short}"] > df[f"SMA{long}"]).astype(int)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]
    return (1 + df["Strategy_Returns"]).cumprod()[-1]

results = {}
for s in [20, 50, 100]:
    for l in [100, 150, 200]:
        if s < l:
            results[(s, l)] = backtest_sma(s, l)

print("Best parameters:", max(results, key=results.get))
```

---

## Step 5: Multi-Asset Portfolio

```python
tickers = ["AAPL", "TSLA", "MSFT", "AMZN"]
portfolio = []

for ticker in tickers:
    df = yf.download(ticker, start="2023-01-01", end="2025-01-01")
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["Signal"] = (df["SMA50"] > df["SMA200"]).astype(int)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]
    portfolio.append(df["Strategy_Returns"])

portfolio_returns = pd.concat(portfolio, axis=1).mean(axis=1)
cumulative_portfolio = (1 + portfolio_returns).cumprod()

cumulative_portfolio.plot(title="Portfolio Performance", figsize=(10,5))
plt.show()
```

---

## Step 6: Alternative Strategies

```python
# Momentum (10-day returns positive)
data["Momentum_Signal"] = (data["Close"].pct_change(10) > 0).astype(int)

# Mean reversion (Buy when price < SMA50)
data["MeanRev_Signal"] = (data["Close"] < data["SMA50"]).astype(int)
```

---

## Step 7: Visualization with Trading Signals

```python
plt.figure(figsize=(12,6))
plt.plot(data["Close"], label="Stock Price", alpha=0.5)
plt.plot(data["SMA50"], label="SMA50")
plt.plot(data["SMA200"], label="SMA200")

plt.scatter(data.index[data["Position"] == 1], 
            data["Close"][data["Position"] == 1], 
            marker="^", color="green", label="Buy")
plt.scatter(data.index[data["Position"] == -1], 
            data["Close"][data["Position"] == -1], 
            marker="v", color="red", label="Sell")

plt.title("Trading Signals with SMA Crossover")
plt.legend()
plt.show()
```

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download 2 years of Apple stock data
data = yf.download("AAPL", start="2023-01-01", end="2025-01-01")

# Calculate moving averages
data["SMA50"] = data["Close"].rolling(window=50).mean()
data["SMA200"] = data["Close"].rolling(window=200).mean()

# Generate signals
data["Signal"] = (data["SMA50"] > data["SMA200"]).astype(int)
data["Position"] = data["Signal"].diff()

transaction_cost = 0.001  # 0.1% per trade

# Daily returns
data["Returns"] = data["Close"].pct_change()

# Apply strategy with cost
data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]

# Subtract cost whenever a trade happens
data.loc[data["Position"] != 0, "Strategy_Returns"] -= transaction_cost

# Sharpe ratio
sharpe = (data["Strategy_Returns"].mean() / data["Strategy_Returns"].std()) * np.sqrt(252)

# Max drawdown
cumulative = (1 + data["Strategy_Returns"]).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_drawdown)

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

cumulative_portfolio.plot(title="Portfolio Performance")
plt.show()

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
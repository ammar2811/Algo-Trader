# Algorithmic Trading Simulator

A compact trading research playground built with **Python, pandas, and yfinance**.  
This project demonstrates the design and evaluation of trading strategies, starting from a simple moving average crossover and extending to more advanced features like portfolio backtesting, risk metrics, and parameter optimization.  

---

## Features

- **Moving Average Crossover Strategy**  
  - Implements short-term vs long-term SMA signals  
  - Buy/Sell signals with visualization  

- **Transaction Costs**  
  - Realistic simulation with configurable per-trade costs  

- **Risk Metrics**  
  - Sharpe Ratio  
  - Maximum Drawdown  

- **Parameter Tuning**  
  - Grid search across SMA window lengths  

- **Multi-Asset Portfolio**  
  - Backtest strategies on multiple tickers (AAPL, TSLA, MSFT, AMZN)  
  - Combine into an equal-weighted portfolio  

- **Alternative Strategies**  
  - Momentum: trade based on rolling returns  
  - Mean Reversion: buy when price dips below SMA  

- **Visualization**  
  - Equity curves for buy & hold vs strategy  
  - Signal plots with buy/sell markers  

---

## Tech Stack

- **Languages & Libraries:** Python, pandas, numpy, matplotlib  
- **Data Source:** [Yahoo Finance](https://pypi.org/project/yfinance/) (`yfinance` API)  
- **Development Tools:** Jupyter Notebook, VS Code  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/algorithmic-trading-simulator.git
cd algorithmic-trading-simulator
pip install -r requirements.txt

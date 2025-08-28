# 📊 Financial Stock Auditing Dashboard

A **Flask + Plotly.js powered stock auditing dashboard** that lets you analyze, audit, and compare multiple stocks interactively.  
This project fetches live market data using **yfinance**, computes technical indicators (SMA, EMA, RSI, etc.), and provides **audit metrics** like volatility, drawdowns, gaps, and correlations.

---

## 🌟 Features

- **Multi-ticker selection** (AAPL, MSFT, TSLA, ^NSEI, NVDA, etc.)
- **Custom date range & interval selection** (`1d`, `1wk`, `1mo`)
- **Technical Indicators**  
  - SMA(20, 50)  
  - EMA(20, 50)  
  - RSI(14)  
- **Stock Audit**  
  - Volatility analysis  
  - Max drawdown  
  - ATR (Average True Range)  
  - Alerts for unusual moves & gaps  
- **Comparison Module**  
  - Normalized performance (start=100)  
  - Correlation heatmap of returns  
- **Download CSV** of processed stock data
- **Interactive Charts**  
  - Price candlesticks + overlays  
  - RSI chart  
  - Volume chart  
  - Correlation heatmap  

---

## 🛠 Tech Stack

**Backend:**  
- Python 3.9+  
- Flask (API + Templating)  
- yfinance (market data)  
- pandas, numpy (data transforms)  

**Frontend:**  
- HTML5 + Bootstrap 5  
- JavaScript (ES6)  
- Plotly.js (charts & heatmaps)  

---

## 📂 Project Structure


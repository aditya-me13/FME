# Streamlit Protective Put App

This folder contains a Streamlit implementation of your protective put project.

## What it includes

- Construct panel with the same core calculations:
  - total stock cost
  - total put cost
  - total investment
  - max loss (hedged)
  - break-even
  - protection floor
- Payoff chart with toggles for stock, put, payoff, and profit
- Hedged vs unhedged scenario comparison
- Cost-of-protection vs strike visualization
- Live market snapshot from yfinance
- Historical backtest comparison:
  - rolling protective put vs stock-only
  - equity curve and drawdown charts
  - risk metrics table

## Run locally

1. Open terminal in this folder.
2. Install dependencies:

   pip install -r requirements.txt

3. Start app:

   streamlit run app.py

## Notes

- yfinance data availability depends on ticker and market hours.
- Option chain availability can vary by ticker.
- Historical protective put backtest uses Black-Scholes estimated option values.

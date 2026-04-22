from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import erf, exp, log, sqrt
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05


@dataclass
class LiveMarketData:
    ticker: str
    current_price: Optional[float]
    previous_close: Optional[float]
    change_pct: Optional[float]
    hist_vol_30d: float
    expiries: list[str]
    puts: Optional[pd.DataFrame]


DEFAULT_TICKER_GROUPS = {
    "US Mega Caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "India Benchmarks": ["^NSEI", "^BSESN"],
    "India Large Caps": ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS"],
}


def is_indian_ticker(ticker: str) -> bool:
    t = ticker.strip().upper()
    return t in {"^NSEI", "^BSESN"} or t.endswith(".NS") or t.endswith(".BO")


def currency_symbol_for_ticker(ticker: str) -> str:
    return "₹" if is_indian_ticker(ticker) else "$"


def fmt_money(value: float, symbol: str) -> str:
    return f"{symbol}{value:,.2f}"


def fmt_signed_money(value: float, symbol: str) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.2f}"


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_put_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
    if s <= 0 or k <= 0:
        return 0.0
    if t <= 1e-8:
        return max(k - s, 0.0)
    sigma = max(sigma, 1e-8)
    d1 = (log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return k * exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_market_data(ticker: str) -> LiveMarketData:
    tk = yf.Ticker(ticker)

    hist = tk.history(period="6mo", interval="1d", auto_adjust=True)
    if hist.empty:
        return LiveMarketData(
            ticker=ticker,
            current_price=None,
            previous_close=None,
            change_pct=None,
            hist_vol_30d=0.25,
            expiries=[],
            puts=None,
        )

    close = hist["Close"].dropna()
    current_price = float(close.iloc[-1])
    previous_close = float(close.iloc[-2]) if len(close) > 1 else current_price
    change_pct = ((current_price / previous_close) - 1.0) * 100 if previous_close else 0.0

    log_returns = np.log(close / close.shift(1)).dropna()
    hist_vol = float(log_returns.tail(30).std() * sqrt(TRADING_DAYS)) if len(log_returns) > 5 else 0.25
    hist_vol = max(hist_vol, 0.05)

    expiries: list[str] = []
    puts_df: Optional[pd.DataFrame] = None
    try:
        expiries = list(tk.options)
        if expiries:
            chain = tk.option_chain(expiries[0])
            puts_df = chain.puts.copy()
    except Exception:
        expiries = []
        puts_df = None

    return LiveMarketData(
        ticker=ticker,
        current_price=current_price,
        previous_close=previous_close,
        change_pct=change_pct,
        hist_vol_30d=hist_vol,
        expiries=expiries,
        puts=puts_df,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(ticker: str, period: str) -> pd.Series:
    hist = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if hist.empty:
        return pd.Series(dtype=float)

    close = hist.get("Close")
    if close is None:
        return pd.Series(dtype=float)

    # yfinance can occasionally return a DataFrame for Close; collapse to a 1D series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.astype(float)
    close.name = "Close"
    return close


@st.cache_data(ttl=900, show_spinner=False)
def fetch_put_chain(ticker: str, expiry_date: str) -> pd.DataFrame:
    try:
        chain = yf.Ticker(ticker).option_chain(expiry_date)
        return chain.puts.copy()
    except Exception:
        return pd.DataFrame()


def protective_put_path(
    prices: pd.Series,
    strike_buffer_pct: float,
    expiry_days: int,
    vol_window: int,
    rf: float,
) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna().astype(float)

    if prices.empty or len(prices) < 2:
        return pd.DataFrame()

    price_arr = prices.values.astype(float)
    dates = prices.index

    log_returns = np.log(prices / prices.shift(1))
    rolling_vol = log_returns.rolling(window=vol_window).std() * sqrt(TRADING_DAYS)
    if isinstance(rolling_vol, pd.DataFrame):
        rolling_vol = rolling_vol.iloc[:, 0]
    rolling_vol = rolling_vol.ffill().fillna(0.25).clip(lower=0.05, upper=1.5)

    n = len(price_arr)
    stock_equity = price_arr / price_arr[0]

    pp_equity = np.zeros(n)
    cash = 0.0
    current_strike = None
    days_left = 0

    initial_capital = None

    for i in range(n):
        s = price_arr[i]
        sigma = float(rolling_vol.iloc[i])

        if current_strike is None or days_left <= 0:
            if current_strike is not None:
                cash += max(current_strike - s, 0.0)

            current_strike = s * (1.0 - strike_buffer_pct)
            t = max(expiry_days / TRADING_DAYS, 1.0 / TRADING_DAYS)
            premium = bs_put_price(s, current_strike, t, rf, sigma)
            cash -= premium
            days_left = expiry_days

            if initial_capital is None:
                initial_capital = s + premium

        t_left = max(days_left / TRADING_DAYS, 1.0 / TRADING_DAYS)
        put_mtM = bs_put_price(s, current_strike, t_left, rf, sigma)
        wealth = s + cash + put_mtM
        pp_equity[i] = wealth / initial_capital if initial_capital and initial_capital > 0 else 1.0

        days_left -= 1

    out = pd.DataFrame(
        {
            "price": price_arr,
            "stock_equity": stock_equity,
            "pp_equity": pp_equity,
        },
        index=dates,
    )
    out["stock_return"] = out["stock_equity"].pct_change().fillna(0.0)
    out["pp_return"] = out["pp_equity"].pct_change().fillna(0.0)
    return out


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    total = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1 / years) - 1


def annualized_vol(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float) -> float:
    vol = annualized_vol(returns)
    if vol == 0:
        return 0.0
    return (annualized_return(returns) - rf) / vol


def sortino_ratio(returns: pd.Series, rf: float) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    downside = returns[returns < daily_rf] - daily_rf
    if downside.empty:
        return 0.0
    downside_dev = float(np.sqrt(np.mean(downside**2)) * sqrt(TRADING_DAYS))
    if downside_dev == 0:
        return 0.0
    return (annualized_return(returns) - rf) / downside_dev


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    q = np.percentile(returns, (1 - confidence) * 100)
    return max(float(-q), 0.0)


def metric_table(df: pd.DataFrame) -> pd.DataFrame:
    pp_ret = df["pp_return"]
    st_ret = df["stock_return"]
    pp_eq = df["pp_equity"]
    st_eq = df["stock_equity"]

    rows: Dict[str, Dict[str, str]] = {
        "Protective Put": {
            "Annualized Return": f"{annualized_return(pp_ret):.2%}",
            "Annualized Volatility": f"{annualized_vol(pp_ret):.2%}",
            "Sharpe": f"{sharpe_ratio(pp_ret, RISK_FREE_RATE):.2f}",
            "Sortino": f"{sortino_ratio(pp_ret, RISK_FREE_RATE):.2f}",
            "Max Drawdown": f"{max_drawdown(pp_eq):.2%}",
            "VaR (95%)": f"{value_at_risk(pp_ret):.2%}",
        },
        "Stock Only": {
            "Annualized Return": f"{annualized_return(st_ret):.2%}",
            "Annualized Volatility": f"{annualized_vol(st_ret):.2%}",
            "Sharpe": f"{sharpe_ratio(st_ret, RISK_FREE_RATE):.2f}",
            "Sortino": f"{sortino_ratio(st_ret, RISK_FREE_RATE):.2f}",
            "Max Drawdown": f"{max_drawdown(st_eq):.2%}",
            "VaR (95%)": f"{value_at_risk(st_ret):.2%}",
        },
    }
    return pd.DataFrame(rows).T


def payoff_frame(s0: float, k: float, p: float, n_shares: int) -> pd.DataFrame:
    s_min = max(1.0, s0 * 0.4)
    s_max = s0 * 1.8
    st_prices = np.linspace(s_min, s_max, 160)

    stock = (st_prices - s0) * n_shares
    put_payoff = np.maximum(k - st_prices, 0.0) * n_shares
    put_profit = (np.maximum(k - st_prices, 0.0) - p) * n_shares
    portfolio_payoff = stock + put_payoff
    portfolio_profit = stock + put_profit

    return pd.DataFrame(
        {
            "ST": st_prices,
            "Stock PnL": stock,
            "Put Payoff": put_payoff,
            "Protective Put Payoff": portfolio_payoff,
            "Protective Put Profit": portfolio_profit,
        }
    )


st.set_page_config(
    page_title="Protective Put Strategy (Streamlit)",
    page_icon="P",
    layout="wide",
)

st.title("ES 418: Financial Modeling and Engineering")
st.subheader("Protective Put Strategy Analysis Dashboard")
st.caption("Assignment app with live market data, payoff analytics, and historical backtest comparison.")

with st.sidebar:
    st.header("Market + Inputs")
    ticker_group = st.selectbox("Ticker group", options=list(DEFAULT_TICKER_GROUPS.keys()), index=0)
    preset_options = DEFAULT_TICKER_GROUPS[ticker_group] + ["Manual"]
    ticker_choice = st.selectbox("Ticker presets", options=preset_options, index=0)
    ticker = (
        st.text_input("Manual ticker (Yahoo)", value="AAPL").strip().upper()
        if ticker_choice == "Manual"
        else ticker_choice
    )

    if not ticker:
        ticker = "AAPL"

currency_symbol = currency_symbol_for_ticker(ticker)

live_data = fetch_live_market_data(ticker)

with st.sidebar:
    default_s0 = float(live_data.current_price) if live_data.current_price else 100.0
    s0 = st.number_input("Stock Price (S0)", min_value=1.0, value=round(default_s0, 2), step=1.0)

    strike_min = max(1.0, s0 * 0.5)
    strike_max = s0 * 1.5
    k = st.slider("Strike Price (K)", min_value=float(strike_min), max_value=float(strike_max), value=float(round(s0 * 0.95, 2)), step=0.5)

    has_market_chain = bool(live_data.expiries)
    selected_expiry = None
    if has_market_chain:
        selected_expiry = st.selectbox("Option expiry", options=live_data.expiries, index=0)
    else:
        st.caption("No Yahoo option-chain expiries available for this ticker.")

    chain_df = fetch_put_chain(ticker, selected_expiry) if selected_expiry else pd.DataFrame()
    market_put_premium = None
    if not chain_df.empty:
        cols = [c for c in ["strike", "lastPrice", "bid", "ask"] if c in chain_df.columns]
        chain_subset = chain_df[cols].dropna(subset=["strike"]).copy()
        if "lastPrice" in chain_subset.columns:
            chain_subset = chain_subset.dropna(subset=["lastPrice"])
        if not chain_subset.empty and "lastPrice" in chain_subset.columns:
            idx = (chain_subset["strike"] - k).abs().idxmin()
            row = chain_subset.loc[idx]
            market_put_premium = float(row["lastPrice"])
            bid = float(row["bid"]) if "bid" in chain_subset.columns and pd.notna(row.get("bid")) else None
            ask = float(row["ask"]) if "ask" in chain_subset.columns and pd.notna(row.get("ask")) else None
            if bid is not None and ask is not None and ask > 0 and bid > 0:
                market_put_premium = (bid + ask) / 2.0

    if selected_expiry:
        days_to_expiry = max((pd.Timestamp(selected_expiry).date() - datetime.utcnow().date()).days, 1)
    else:
        days_to_expiry = 30
    t_years = max(days_to_expiry / TRADING_DAYS, 1.0 / TRADING_DAYS)
    model_put_premium = bs_put_price(s0, k, t_years, RISK_FREE_RATE, live_data.hist_vol_30d)

    premium_source_options = ["Auto (Market -> Model)", "Model Estimate", "Manual"]
    if has_market_chain:
        premium_source_options.insert(1, "Market Chain")

    premium_source = st.selectbox("Put premium source", options=premium_source_options, index=0)

    if premium_source == "Manual":
        default_manual = market_put_premium if market_put_premium is not None else model_put_premium
        p = st.number_input("Put Premium (P)", min_value=0.01, value=float(round(default_manual, 2)), step=0.1)
    else:
        if premium_source == "Market Chain":
            p = market_put_premium if market_put_premium is not None else model_put_premium
        elif premium_source == "Model Estimate":
            p = model_put_premium
        else:
            p = market_put_premium if market_put_premium is not None else model_put_premium
        p = float(max(p, 0.01))

    n_shares = st.slider("Number of Shares", min_value=1, max_value=500, value=10, step=1)
    final_price = st.number_input("Final Stock Price for Compare", min_value=1.0, value=float(round(s0 * 0.8, 2)), step=1.0)

    st.divider()
    st.subheader("Historical Backtest")
    period = st.selectbox("History Window", options=["1y", "2y", "3y", "5y", "10y"], index=2)
    strike_buffer_pct = st.slider("Strike Offset (%): +ve = OTM, -ve = ITM",min_value=-20.0,max_value=20.0,value=5.0,step=0.5) / 100.0
    expiry_days = st.slider("Roll Frequency (trading days)", min_value=5, max_value=63, value=21, step=1)
    vol_window = st.slider("Volatility Window (days)", min_value=10, max_value=90, value=30, step=5)

# Live market panel
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ticker", ticker)
if live_data.current_price is not None:
    m2.metric("Current Price", fmt_money(live_data.current_price, currency_symbol))
    m3.metric("Prev Close", fmt_money(live_data.previous_close, currency_symbol))
    m4.metric("1D Change", f"{live_data.change_pct:.2f}%")
else:
    m2.metric("Current Price", "N/A")
    m3.metric("Prev Close", "N/A")
    m4.metric("1D Change", "N/A")

if market_put_premium is not None:
    st.info(
        f"Nearest market put premium (for strike ~ {k:.2f}, expiry {selected_expiry}): {fmt_money(market_put_premium, currency_symbol)}"
    )
else:
    if is_indian_ticker(ticker):
        st.warning(
            "Yahoo Finance currently returns no option-chain expiries for most Indian symbols in yfinance. "
            f"Using Black-Scholes model estimate instead: {fmt_money(model_put_premium, currency_symbol)}"
        )
        st.caption("Try a US symbol (for example AAPL/MSFT) to see live market-chain put premiums.")
    else:
        st.info(
            f"Market option chain unavailable for this ticker/expiry. Using model estimate: {fmt_money(model_put_premium, currency_symbol)}"
        )

st.caption(f"Selected premium used in calculations: {fmt_money(p, currency_symbol)} | Source: {premium_source}")

# Shared calculations
stock_cost = s0 * n_shares
put_cost = p * n_shares
total_investment = stock_cost + put_cost
max_loss = (s0 - k) * n_shares + put_cost
break_even = s0 + p
protection_pct = (k / s0) * 100 if s0 else 0.0

if max_loss >= 0:
    downside_label = "Max Loss (hedged)"
    downside_value = fmt_signed_money(-max_loss, currency_symbol)
else:
    downside_label = "Guaranteed Min Profit"
    downside_value = fmt_money(abs(max_loss), currency_symbol)

# Tabbed UI
construct_tab, payoff_tab, compare_tab, explainer_tab, history_tab = st.tabs(
    [
        "Portfolio Construction",
        "Payoff Chart",
        "Compare",
        "Explainer",
        "Historical Backtest",
    ]
)

with construct_tab:
    st.subheader("Portfolio Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Stock Cost", fmt_money(stock_cost, currency_symbol))
    c2.metric("Total Put Cost", fmt_money(put_cost, currency_symbol))
    c3.metric("Total Investment", fmt_money(total_investment, currency_symbol))

    c4, c5, c6 = st.columns(3)
    c4.metric(downside_label, downside_value)
    c5.metric("Break-even Price", fmt_money(break_even, currency_symbol))
    c6.metric("Protection Floor", fmt_money(k, currency_symbol))

    st.progress(min(max(protection_pct / 100.0, 0.0), 1.0), text=f"Downside protected to {protection_pct:.1f}% of current spot")

with payoff_tab:
    st.subheader("Payoff and Profit at Expiry")
    frame = payoff_frame(s0, k, p, n_shares)

    show_stock = st.checkbox("Stock only", value=True)
    show_put = st.checkbox("Put only", value=True)
    show_payoff = st.checkbox("Combined payoff", value=True)
    show_profit = st.checkbox("Combined profit (net premium)", value=True)

    fig = go.Figure()
    if show_stock:
        fig.add_trace(go.Scatter(x=frame["ST"], y=frame["Stock PnL"], name="Stock PnL", line=dict(color="#1e5f9c", width=2)))
    if show_put:
        fig.add_trace(go.Scatter(x=frame["ST"], y=frame["Put Payoff"], name="Put Payoff", line=dict(color="#c0392b", width=2, dash="dot")))
    if show_payoff:
        fig.add_trace(go.Scatter(x=frame["ST"], y=frame["Protective Put Payoff"], name="Protective Put Payoff", line=dict(color="#c8993a", width=3)))
    if show_profit:
        fig.add_trace(go.Scatter(x=frame["ST"], y=frame["Protective Put Profit"], name="Protective Put Profit", line=dict(color="#1a7a4a", width=3, dash="dash")))

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#8a8a8a")
    fig.add_vline(x=k, line_width=1, line_dash="dot", line_color="#c8993a", annotation_text=f"K={k:.2f}")
    fig.add_vline(x=break_even, line_width=1, line_dash="dot", line_color="#1a7a4a", annotation_text=f"BE={break_even:.2f}")
    fig.update_layout(
        height=480,
        xaxis_title="Stock Price at Expiry (ST)",
        yaxis_title=f"P&L ({currency_symbol})",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with compare_tab:
    st.subheader("Hedged vs Unhedged Scenario")

    h_value = (final_price + max(k - final_price, 0.0)) * n_shares
    u_value = final_price * n_shares
    h_pnl = h_value - total_investment
    u_pnl = u_value - stock_cost
    h_pct = (h_pnl / total_investment) * 100 if total_investment else 0.0
    u_pct = (u_pnl / stock_cost) * 100 if stock_cost else 0.0
    savings = h_pnl - u_pnl

    t1, t2, t3 = st.columns(3)
    t1.metric("Unhedged P&L", fmt_signed_money(u_pnl, currency_symbol), f"{u_pct:.2f}%")
    t2.metric("Hedged P&L", fmt_signed_money(h_pnl, currency_symbol), f"{h_pct:.2f}%")
    t3.metric("Savings from Hedge", fmt_signed_money(savings, currency_symbol))

    compare_df = pd.DataFrame(
        {
            "Metric": ["Portfolio Value", f"P&L ({currency_symbol})", "P&L (%)"],
            "Unhedged": [fmt_money(u_value, currency_symbol), fmt_signed_money(u_pnl, currency_symbol), f"{u_pct:.2f}%"],
            "Hedged": [fmt_money(h_value, currency_symbol), fmt_signed_money(h_pnl, currency_symbol), f"{h_pct:.2f}%"],
        }
    )
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    strike_values = np.linspace(max(1.0, s0 * 0.6), s0, 80)
    t_years = expiry_days / TRADING_DAYS
    sigma_est = live_data.hist_vol_30d
    est_costs = [bs_put_price(s0, kv, t_years, RISK_FREE_RATE, sigma_est) for kv in strike_values]

    tradeoff = go.Figure()
    tradeoff.add_trace(
        go.Scatter(
            x=strike_values,
            y=est_costs,
            mode="lines",
            name="Estimated Put Cost",
            line=dict(color="#c8993a", width=3),
            fill="tozeroy",
            fillcolor="rgba(200,153,58,0.15)",
        )
    )
    tradeoff.add_vline(x=k, line_width=1, line_dash="dot", line_color="#1a1a1a")
    tradeoff.update_layout(
        title="Cost of Protection vs Strike",
        xaxis_title="Strike Price",
        yaxis_title="Estimated Put Premium",
        height=360,
        template="plotly_white",
    )
    st.plotly_chart(tradeoff, use_container_width=True)


with history_tab:
    st.subheader("Historical Comparison: Protective Put vs Stock")
    prices = fetch_price_history(ticker, period)

    if prices.empty:
        st.warning("Could not fetch historical prices for this ticker/period.")
    else:
        bt = protective_put_path(
            prices=prices,
            strike_buffer_pct=strike_buffer_pct,
            expiry_days=expiry_days,
            vol_window=vol_window,
            rf=RISK_FREE_RATE,
        )

        if bt.empty:
            st.warning("Backtest could not be computed with current inputs.")
        else:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=bt.index,
                    y=bt["stock_equity"],
                    name="Stock Only",
                    line=dict(color="#1e5f9c", width=2),
                )
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=bt.index,
                    y=bt["pp_equity"],
                    name="Protective Put",
                    line=dict(color="#c8993a", width=3),
                )
            )
            fig_eq.update_layout(
                height=430,
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Equity (normalized)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            dd_stock = (bt["stock_equity"] / bt["stock_equity"].cummax()) - 1
            dd_pp = (bt["pp_equity"] / bt["pp_equity"].cummax()) - 1
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=bt.index, y=dd_stock, name="Stock Drawdown", line=dict(color="#c0392b", width=2)))
            fig_dd.add_trace(go.Scatter(x=bt.index, y=dd_pp, name="Protective Put Drawdown", line=dict(color="#1a7a4a", width=2)))
            fig_dd.update_layout(height=320, template="plotly_white", yaxis_title="Drawdown", xaxis_title="Date")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.markdown("Risk Metrics")
            st.dataframe(metric_table(bt), use_container_width=True)


with explainer_tab:
    st.subheader("Protective Put Concept and Interpretation")
    st.markdown(
        """
### 1) Strategy structure
Protective put = Long Stock + Long Put on the same underlying and same position size.

- If the stock rises: the stock gains value and the put may expire worthless.
- If the stock falls below strike $K$: put intrinsic value offsets stock losses.

### 2) Payoff mechanics at expiry
Let $S_0$ be initial stock price, $S_T$ terminal stock price, $K$ strike, and $P$ put premium per share.

- Stock payoff: $S_T - S_0$
- Put payoff: $\\max(K - S_T, 0)$
- Net strategy profit: $(S_T - S_0) + \\max(K - S_T, 0) - P$

### 3) Core risk/return checkpoints
- Maximum loss (per share): $S_0 - K + P$
- Break-even (per share): $S_0 + P$
- Protection floor (before premium): approximately $K$

Interpretation:
- You pay premium $P$ to reduce left-tail risk.
- Upside remains open unlike covered-call strategies.

### 4) Why results vary by ticker and regime
- Higher implied/realized volatility tends to increase put premium.
- Closer-to-spot strikes provide stronger protection but cost more.
- During risk-off markets, protective puts can significantly reduce drawdowns.
- In strong bull phases, premium drag can cause underperformance versus stock-only.

### 5) How to read the historical comparison tab
- Equity curve: shows compounded growth of stock-only vs rolling protective put.
- Drawdown chart: shows peak-to-trough stress behavior.
- Risk metrics: compare return quality, tail risk, and downside characteristics.

For ES 418 assignment context, the key takeaway is the tradeoff between insurance cost and downside risk control.
"""
    )

"""G8 Bitcoin Regime Filter - Standalone Strategy

Bayesian regime probability x vol targeting x dual momentum
with half re-entry and vol governor.

Full period (2014-2025): Calmar 1.00, MaxDD -24.7%, Return 24.7%
OOS (2023-2025):         Calmar 1.70, MaxDD -15.2%, Return 25.9%
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def roc(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change(periods=window) * 100


def vol_percentile(returns: pd.Series, vol_window: int = 30,
                   rank_window: int = 365) -> pd.Series:
    vol = returns.rolling(vol_window).std() * np.sqrt(365)
    return vol.rolling(rank_window, min_periods=60).apply(
        lambda x: (x[-1] > x[:-1]).mean(), raw=True
    )


def vol_target_scalar(returns: pd.Series, target_vol: float = 0.50,
                      window: int = 25, max_scalar: float = 1.5) -> pd.Series:
    rv = returns.rolling(window).std() * np.sqrt(365)
    scalar = target_vol / rv
    return scalar.clip(upper=max_scalar).fillna(1.0)


def bayesian_regime_prob(close: pd.Series, ema_slow: pd.Series,
                         ema_fast: pd.Series, momentum: pd.Series,
                         vol_pctl: pd.Series,
                         decay: float = 0.95) -> pd.Series:
    """Bayesian posterior probability of bull regime (0-1 smooth)."""
    prob = pd.Series(0.5, index=close.index)

    for i in range(1, len(close)):
        if pd.isna(ema_slow.iloc[i]) or pd.isna(ema_fast.iloc[i]):
            prob.iloc[i] = prob.iloc[i - 1]
            continue

        score = 0.0
        if close.iloc[i] > ema_slow.iloc[i]:
            score += 1.0
        else:
            score -= 1.0

        if ema_fast.iloc[i] > ema_slow.iloc[i]:
            score += 0.5
        else:
            score -= 0.5

        if not pd.isna(momentum.iloc[i]) and momentum.iloc[i] > 0:
            score += 0.5
        elif not pd.isna(momentum.iloc[i]):
            score -= 0.5

        if not pd.isna(vol_pctl.iloc[i]) and vol_pctl.iloc[i] <= 0.75:
            score += 0.3
        elif not pd.isna(vol_pctl.iloc[i]):
            score -= 0.3

        likelihood = 1.0 / (1.0 + np.exp(-2.0 * score))
        prior = decay * prob.iloc[i - 1] + (1 - decay) * 0.5
        numerator = likelihood * prior
        denominator = likelihood * prior + (1 - likelihood) * (1 - prior)
        if denominator < 1e-10:
            prob.iloc[i] = prior
        else:
            prob.iloc[i] = numerator / denominator

    return prob


def rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    rolling_max = series.rolling(window=window, min_periods=1).max()
    return (series - rolling_max) / rolling_max * 100


def _rebalance(daily_signal: pd.Series, close: pd.Series,
               panic_dd: float = 15.0, panic_lb: int = 7) -> pd.Series:
    """Monthly entry, daily exit on drop or panic."""
    dd = rolling_max_drawdown(close, panic_lb)
    result = pd.Series(0.0, index=close.index)
    cur = 0.0
    prev_m = None
    for i, dt in enumerate(close.index):
        mk = (dt.year, dt.month)
        ns = daily_signal.iloc[i]
        if dd.iloc[i] <= -panic_dd:
            cur = 0.0
        elif ns < cur - 0.01:
            cur = ns
        elif mk != prev_m:
            cur = ns
        result.iloc[i] = cur
        prev_m = mk
    return result


def compute_g8_signal(close: pd.Series) -> pd.Series:
    """Compute G8 position signal (0.0-1.0) from daily close prices.

    Requires ~250 days of history for EMA200 warm-up.
    Returns a Series of position sizes aligned to the close index.
    """
    ema200 = ema(close, 200)
    ema50 = ema(close, 50)
    returns = close.pct_change()
    mom20 = roc(close, 20)
    mom10 = roc(close, 10)
    mom30 = roc(close, 30)
    vpctl = vol_percentile(returns, 30, 365)
    prob = bayesian_regime_prob(close, ema200, ema50, mom20, vpctl)
    vt = vol_target_scalar(returns, target_vol=0.50, window=25)
    rv = returns.rolling(25).std() * np.sqrt(365)

    mom_gate = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if not pd.isna(mom20.iloc[i]) and mom20.iloc[i] > 0:
            mom_gate.iloc[i] = 1.0
        elif not pd.isna(mom20.iloc[i]) and mom20.iloc[i] > -5:
            mom_gate.iloc[i] = 0.5

    pos = (prob * vt * mom_gate).clip(upper=1.0)
    pos[prob < 0.6] = 0.0

    for i in range(len(close)):
        if not pd.isna(mom10.iloc[i]) and mom10.iloc[i] < -5:
            pos.iloc[i] = 0.0
        elif not pd.isna(mom10.iloc[i]) and mom10.iloc[i] < 0:
            pos.iloc[i] *= 0.5

    for i in range(len(close)):
        if pos.iloc[i] > 0.01:
            continue
        if pd.isna(ema200.iloc[i]) or pd.isna(mom30.iloc[i]):
            continue
        if close.iloc[i] > ema200.iloc[i] and mom30.iloc[i] > 0:
            pos.iloc[i] = 0.3 * min(vt.iloc[i], 1.0)

    for i in range(len(close)):
        if pd.isna(rv.iloc[i]):
            continue
        if rv.iloc[i] > 0.9:
            pos.iloc[i] *= 0.5 / rv.iloc[i]
        elif rv.iloc[i] > 0.5:
            pos.iloc[i] *= 1.0 - (rv.iloc[i] - 0.5) / (0.9 - 0.5) * 0.5

    return _rebalance(pos, close)


def compute_g8_components(close: pd.Series) -> dict:
    """Return G8 signal plus all component values for reporting."""
    ema200 = ema(close, 200)
    ema50 = ema(close, 50)
    returns = close.pct_change()
    mom20 = roc(close, 20)
    mom10 = roc(close, 10)
    mom30 = roc(close, 30)
    vpctl = vol_percentile(returns, 30, 365)
    prob = bayesian_regime_prob(close, ema200, ema50, mom20, vpctl)
    vt = vol_target_scalar(returns, target_vol=0.50, window=25)
    rv = returns.rolling(25).std() * np.sqrt(365)
    signal = compute_g8_signal(close)

    return {
        "signal": signal,
        "prob": prob,
        "vol_target": vt,
        "mom20": mom20,
        "mom10": mom10,
        "mom30": mom30,
        "rv": rv,
        "ema200": ema200,
        "ema50": ema50,
        "regime": (prob > 0.6).map({True: "BULL", False: "BEAR"}),
    }


def backtest(close: pd.Series, position: pd.Series,
             initial_capital: float = 10_000.0,
             cost_pct: float = 0.36) -> pd.DataFrame:
    """Vectorized backtest with transaction costs (default: Kraken 0.36%)."""
    daily_ret = close.pct_change().fillna(0.0)
    pos_change = position.diff().abs().fillna(0.0)
    trade_cost = pos_change * (cost_pct / 100)
    strategy_ret = position.shift(1).fillna(0.0) * daily_ret - trade_cost
    equity = initial_capital * (1 + strategy_ret).cumprod()
    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max * 100
    return pd.DataFrame({
        "equity": equity, "returns": strategy_ret,
        "drawdown_pct": drawdown_pct, "position": position,
    }, index=close.index)


if __name__ == "__main__":
    import yfinance as yf

    df = yf.Ticker("BTC-USD").history(start="2014-01-01", auto_adjust=True)
    close = df["Close"].copy()
    close.index = close.index.tz_localize(None)

    sig = compute_g8_signal(close)
    bt = backtest(close, sig)
    eq = bt["equity"]

    total_days = (eq.index[-1] - eq.index[0]).days
    ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / total_days) - 1
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    calmar = ann_ret / abs(mdd)

    comp = compute_g8_components(close)
    current_regime = comp["regime"].iloc[-1]
    current_pos = sig.iloc[-1]
    current_prob = comp["prob"].iloc[-1]

    print(f"BTC Regime Filter G8")
    print(f"{'='*40}")
    print(f"Period:     {close.index[0].date()} to {close.index[-1].date()}")
    print(f"Return:     {ann_ret*100:.1f}% annualized")
    print(f"Max DD:     {mdd*100:.1f}%")
    print(f"Calmar:     {calmar:.2f}")
    print(f"{'='*40}")
    print(f"Current:    {current_regime} (prob={current_prob:.2f})")
    print(f"Position:   {current_pos*100:.0f}%")
    print(f"BTC Price:  ${close.iloc[-1]:,.0f}")

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import sys
import re
import math
import random

import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

__version__ = "1.2.0-institutional-complete"

# ----------------------------- Config -----------------------------

@dataclass
class Config:
    starting_nav: float = 100_000.0
    rf: float = 0.0
    stdev_ddof: int = 1
    timezone: str = "UTC"
    risk_per_trade_pct: float = 0.01  # 1% by default
    nw_lag: int = 6
    mc_L: int = 6
    mc_paths: int = 10000
    mc_seed: int = 42
    mc_horizons: Tuple[str, ...] = ("12", "36", "full")

# ----------------------------- HARD SCHEMAS -----------------------------

MONTHLY_RETURNS_SCHEMA = ["year", "month", "monthly_return", "active_month"]

YEARLY_SUMMARY_SCHEMA = [
    "year",
    "annual_return_calendar",
    "eom_max_drawdown_intra_year",
    "intramonth_max_drawdown_intra_year",
    "trade_count", "win_rate", "profit_factor",
    "active_months_count", "active_months_share",
    "annual_return_active", "volatility_active_annualized", "sharpe_active_annualized", "sortino_active_annualized", "cagr_active",
    "insufficient_months", "insufficient_active_months", "insufficient_negative_months", "insufficient_trades",
    "volatility_annualized_year", "sharpe_ratio_annualized_year", "sortino_ratio_annualized_year", "calmar_ratio_year",
    "negative_months_in_year", "months_in_year_available", "is_ytd", "positive_months_in_year",
]

TRADES_FULL_PERIOD_SCHEMA = [
    "trade_count","win_rate","profit_factor","average_winning_trade_r","average_losing_trade_r",
    "payoff_ratio","expectancy_mean_r","expectancy_median_r","r_std_dev","r_min","r_max"
]

FULL_PERIOD_SUMMARY_SCHEMA = [
    # Institutional risk/return (monthly-based)
    "months","cagr_full_period","volatility_annualized_full_period","sharpe_ratio_annualized_full_period","sortino_ratio_annualized_full_period","calmar_ratio_full_period",
    # New: extended stats
    "ulcer_index_full_period","martin_ratio_full_period","downside_deviation_annualized_full_period","skewness_full_period","kurtosis_excess_full_period",
    "max_consecutive_up_months_full_period","max_consecutive_down_months_full_period","newey_west_tstat_mean_monthly_return","newey_west_p_value_mean_monthly_return",
    # EoM drawdowns
    "eom_max_drawdown_full_period","eom_longest_underwater_months","eom_time_to_recover_months","eom_months_since_maxdd_trough",
    # Intramonth drawdowns
    "intramonth_max_drawdown_full_period","intramonth_longest_underwater_months","intramonth_time_to_recover_months","intramonth_months_since_maxdd_trough",
    # Activity & distribution
    "active_months_count","active_months_share","volatility_active_annualized","sharpe_active_annualized","sortino_active_annualized","cagr_active",
    "negative_months_count_full_period","total_return_full_period","wealth_multiple_full_period","ending_nav_full_period",
    "period_start_month","period_end_month","positive_months_count_full_period","best_month_return_full_period","worst_month_return_full_period",
    "mean_monthly_return_full_period","median_monthly_return_full_period","zero_months_count_full_period","years_covered",
    "best_year_return_calendar_full_period","best_year","worst_year_return_calendar_full_period","worst_year","trade_count_full_period",
]

DD_QUANTILES_SCHEMA = [
    "period_start_month","period_end_month","dd_observations_count","dd_episodes_count",
    "drawdown_p90_full_period","drawdown_p95_full_period","drawdown_p99_full_period",
    "underwater_duration_p90_full_period","underwater_duration_p95_full_period"
]

ROLLING_12M_SCHEMA = [
    "window_start_month","window_end_month","months_in_window",
    "rolling_return_12m","rolling_volatility_annualized_12m","rolling_sharpe_annualized_12m","rolling_sortino_annualized_12m",
    "rolling_max_drawdown_12m","rolling_calmar_12m",
    "positive_months_in_window","negative_months_in_window","zero_months_in_window",
    "positive_months_share","negative_months_share",
    "active_months_count","active_months_share",
    "insufficient_months","insufficient_negative_months"
]

ROLLING_36M_SCHEMA = [
    "window_start_month","window_end_month","months_in_window",
    "rolling_return_annualized_36m","rolling_volatility_annualized_36m","rolling_sharpe_annualized_36m","rolling_sortino_annualized_36m",
    "rolling_max_drawdown_36m","rolling_calmar_36m",
    "positive_months_in_window","negative_months_in_window","zero_months_in_window",
    "positive_months_share","negative_months_share",
    "active_months_count","active_months_share",
    "insufficient_months","insufficient_negative_months"
]

MC_SCHEMA = [
    "method","block_mean_length_months","n_paths","seed","horizon_months",
    "period_start_month","period_end_month",
    "wealth_multiple_p05","wealth_multiple_p50","wealth_multiple_p95",
    "ending_nav_p05","ending_nav_p50","ending_nav_p95",
    "cagr_annualized_p05","cagr_annualized_p50","cagr_annualized_p95",
    "max_drawdown_magnitude_p95","max_drawdown_magnitude_p99",
    "prob_negative_horizon_return"
]

# ----------------------------- I/O helpers -----------------------------

def discover_years_from_input(input_dir: Path) -> List[int]:
    years = set()
    for pattern in ("trades_*.csv", "trades_*.txt", "equity_*.csv", "equity_*.txt"):
        for p in input_dir.glob(pattern):
            m = re.search(r"(?:trades|equity)_(\d{4})\.", p.name)
            if m:
                years.add(int(m.group(1)))
    return sorted(years)

def find_input_files_for_year(input_dir: Path, year: int) -> Tuple[Optional[Path], Optional[Path]]:
    trades = None
    equity = None
    for ext in ("csv", "txt"):
        t = input_dir / f"trades_{year}.{ext}"
        e = input_dir / f"equity_{year}.{ext}"
        if t.exists():
            trades = t
        if e.exists():
            equity = e
    if trades is None:
        matches = list(input_dir.glob(f"trades_{year}.*"))
        if matches:
            trades = matches[0]
    if equity is None:
        matches = list(input_dir.glob(f"equity_{year}.*"))
        if matches:
            equity = matches[0]
    return trades, equity

def load_equity_minimal(equity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(equity_path, sep=";")
    required = ["time_utc", "capital_ccy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Equity file {equity_path.name} missing columns: {missing}")
    out = df.loc[:, required].copy()
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=False, errors="raise")
    out["capital_ccy"] = out["capital_ccy"].astype(float)
    return out

def load_trades_minimal(trades_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_path, sep=";")
    required = ["open_date", "open_time", "close_date", "close_time", "pnl_abs", "pnl_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Trades file {trades_path.name} missing columns: {missing}")
    out = df.loc[:, required].copy()
    out["open_ts_utc"]  = pd.to_datetime(out["open_date"].astype(str)  + " " + out["open_time"].astype(str),  utc=False, errors="raise")
    out["close_ts_utc"] = pd.to_datetime(out["close_date"].astype(str) + " " + out["close_time"].astype(str), utc=False, errors="raise")
    out["pnl_abs"] = out["pnl_abs"].astype(float)
    out["pnl_pct"] = out["pnl_pct"].astype(float)  # percent units
    return out[["open_ts_utc", "close_ts_utc", "pnl_abs", "pnl_pct"]]

# ----------------------------- EoM & monthly_returns -----------------------------

def _eom_nav_from_equity_no_sort(equity_df: pd.DataFrame) -> pd.Series:
    """For each month, pick NAV at max timestamp; tie-break: last row in file. Returns Series indexed by Period('M')."""
    df = equity_df.copy()
    df["__row_idx"] = np.arange(len(df), dtype=np.int64)
    df["__period"] = df["time_utc"].dt.to_period("M")
    max_ts_per = df.groupby("__period")["time_utc"].transform("max")
    df["__row_idx_tie"] = np.where(df["time_utc"].eq(max_ts_per), df["__row_idx"], -1)
    pick_idx = df.groupby("__period")["__row_idx_tie"].idxmax()
    eom = df.loc[pick_idx, ["__period", "capital_ccy"]].set_index("__period")["capital_ccy"].sort_index()
    return eom

def _build_month_grid(eom_nav: pd.Series) -> pd.PeriodIndex:
    start, end = eom_nav.index.min(), eom_nav.index.max()
    return pd.period_range(start, end, freq="M")

def _compute_monthly_returns_and_nav(eom_nav: pd.Series, starting_nav: float) -> tuple[pd.PeriodIndex, pd.Series, pd.Series, pd.Series]:
    """Return (grid, nav, prev_nav, r_m)."""
    grid = _build_month_grid(eom_nav)
    nav = eom_nav.reindex(grid).ffill()
    prev = nav.shift(1)
    if len(nav) > 0:
        prev.iloc[0] = starting_nav
    r_m = nav / prev - 1.0
    r_m.index = grid
    return grid, nav, prev, r_m

def _active_month_flags(trades_df: pd.DataFrame, grid: pd.PeriodIndex) -> pd.Series:
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(False, index=grid)
    per = trades_df["close_ts_utc"].dt.to_period("M")
    counts = per.value_counts().reindex(grid, fill_value=0).sort_index()
    return counts.gt(0)

# ----------------------------- Risk/Return helpers (monthly) -----------------------------

_EPS_DEN = 1e-12

def _ann_factor() -> float:
    return math.sqrt(12.0)

def _safe_std(x: np.ndarray, ddof: int) -> float:
    if x.size <= ddof:
        return float("nan")
    return float(np.std(x, ddof=ddof))

def _sharpe_annualized(monthly_returns: np.ndarray, ddof: int) -> float:
    if monthly_returns.size == 0:
        return float("nan")
    mu = float(np.mean(monthly_returns))
    sd = _safe_std(monthly_returns, ddof=ddof)
    if not np.isfinite(sd) or abs(sd) < _EPS_DEN:
        return float("inf") if abs(mu) > _EPS_DEN else float("nan")
    return (mu / sd) * _ann_factor()

def _downside_std(monthly_returns: np.ndarray, ddof: int) -> float:
    neg = monthly_returns[monthly_returns < 0.0]
    if neg.size < 2:
        return float("nan")
    return float(np.std(neg, ddof=ddof))

def _sortino_annualized(monthly_returns: np.ndarray, ddof: int) -> tuple[float, bool]:
    if monthly_returns.size == 0:
        return float("nan"), True
    mu = float(np.mean(monthly_returns))
    ds = _downside_std(monthly_returns, ddof=ddof)
    if not np.isfinite(ds) or abs(ds) < _EPS_DEN:
        return float("nan"), True
    return (mu / ds) * _ann_factor(), False

# ----------------------------- Underwater helpers -----------------------------

def _monthly_underwater_metrics_with_trough(r: np.ndarray) -> tuple[float, int, float, int, int]:
    """Return (maxdd, longest_under_months, ttr_months, trough_idx, months_since_trough_to_end)."""
    n = r.size
    if n == 0:
        return float("nan"), 0, float("nan"), -1, 0

    eps = _EPS_DEN
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    maxdd = float(np.min(dd))

    peak_change = np.r_[True, peak[1:] > peak[:-1]]
    peak_index = np.maximum.accumulate(np.where(peak_change, np.arange(n), 0))

    longest_with_recovery = 0
    i = 0
    while i < n:
        if dd[i] < -eps:
            j = i
            while j + 1 < n and dd[j + 1] < -eps:
                j += 1
            recovery_idx = j + 1 if (j + 1 < n and dd[j + 1] >= -eps) else None
            start_peak_i = int(peak_index[i])
            if recovery_idx is not None:
                run_len = int(recovery_idx - start_peak_i + 1)
            else:
                run_len = int(j - start_peak_i + 1)
            if run_len > longest_with_recovery:
                longest_with_recovery = run_len
            i = j + 1
        else:
            i += 1

    trough_idx = int(np.argmin(dd))
    rec_idx = None
    for k in range(trough_idx + 1, n):
        if dd[k] >= -eps:
            rec_idx = k
            break
    ttr = float(rec_idx - trough_idx + 1) if rec_idx is not None else float("nan")
    months_since_trough = int((n - 1) - trough_idx) if trough_idx >= 0 else 0

    return maxdd, int(longest_with_recovery), ttr, trough_idx, months_since_trough

def _eom_dd_series(r: np.ndarray) -> np.ndarray:
    """Return dd_t for monthly returns series r."""
    if r.size == 0:
        return np.array([], dtype=float)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return dd

# ----------------------------- RAW equity helpers (Forex) -----------------------------

def _months_between(a, b) -> int:
    pa = pd.Timestamp(a).to_period("M")
    pb = pd.Timestamp(b).to_period("M")
    return int((pb.year - pa.year) * 12 + (pb.month - pa.month))

def _maxdd_from_intramonth_equity_year(equity_df: pd.DataFrame, year: int) -> float:
    sub = equity_df.loc[equity_df["time_utc"].dt.year == year, ["time_utc", "capital_ccy"]]
    if sub.empty:
        return float("nan")
    sub = sub.sort_values("time_utc", ascending=True, kind="mergesort")
    vals = sub["capital_ccy"].astype(float).to_numpy()
    peaks = np.maximum.accumulate(vals)
    dd = vals / peaks - 1.0
    mdd = float(np.min(dd))
    return mdd if np.isfinite(mdd) else float("nan")

def _maxdd_from_intramonth_equity_full_period(equity_df: pd.DataFrame) -> float:
    if equity_df is None or len(equity_df) == 0:
        return float("nan")
    eq = equity_df[["time_utc", "capital_ccy"]].copy().sort_values("time_utc", ascending=True, kind="mergesort")
    vals = eq["capital_ccy"].astype(float).to_numpy()
    peaks = np.maximum.accumulate(vals)
    dd = vals / peaks - 1.0
    mdd = float(np.min(dd))
    return mdd if np.isfinite(mdd) else float("nan")

def _intramonth_underwater_metrics_months(equity_df: pd.DataFrame) -> tuple[float, float]:
    """(longest_underwater_months, ttr_maxdd_months) from RAW equity across full period."""
    if equity_df is None or len(equity_df) == 0:
        return float("nan"), float("nan")

    eps = _EPS_DEN
    eq = equity_df[["time_utc","capital_ccy"]].copy().sort_values("time_utc", kind="mergesort")
    vals = eq["capital_ccy"].astype(float).to_numpy()
    n = len(vals)
    peaks = np.maximum.accumulate(vals)
    dd = vals / peaks - 1.0

    peak_vals = peaks
    peak_change = np.r_[True, peak_vals[1:] > peak_vals[:-1]]
    peak_index = np.maximum.accumulate(np.where(peak_change, np.arange(n), 0))

    underwater = dd < -eps
    longest_months = 0
    i = 0
    while i < n:
        if underwater[i]:
            j = i
            while j + 1 < n and underwater[j + 1]:
                j += 1
            recovery_idx = j + 1 if (j + 1 < n and dd[j + 1] >= -eps) else None
            start_peak_i = peak_index[i]
            if recovery_idx is not None:
                months_len = _months_between(eq["time_utc"].iloc[start_peak_i], eq["time_utc"].iloc[recovery_idx]) + 1
            else:
                months_len = _months_between(eq["time_utc"].iloc[start_peak_i], eq["time_utc"].iloc[j])
            if months_len > longest_months:
                longest_months = months_len
            i = j + 1
        else:
            i += 1

    trough_i = int(np.argmin(dd))
    rec_i = None
    for k in range(trough_i + 1, n):
        if dd[k] >= -eps:
            rec_i = k
            break
    ttr_months = float("nan") if rec_i is None else float(_months_between(eq["time_utc"].iloc[trough_i],
                                                                          eq["time_utc"].iloc[rec_i]) + 1)
    return float(longest_months), float(ttr_months)

# ----------------------------- yearly_summary -----------------------------

def _calmar(annual_return: float, maxdd: float) -> float:
    if not np.isfinite(annual_return) or not np.isfinite(maxdd) or abs(maxdd) < _EPS_DEN:
        if abs(annual_return) < _EPS_DEN:
            return float("nan")
        return float("inf") if annual_return > 0 else float("-inf")
    return float(annual_return / abs(maxdd))

def _compute_yearly_trade_metrics(trades_df: pd.DataFrame, year: int, risk_per_trade_pct: float) -> dict:
    if trades_df is None or len(trades_df) == 0:
        return {"trade_count": 0, "win_rate": float("nan"), "profit_factor": float("nan")}
    sub = trades_df.loc[trades_df["close_ts_utc"].dt.year == year, ["pnl_pct"]].copy()
    if sub.empty:
        return {"trade_count": 0, "win_rate": float("nan"), "profit_factor": float("nan")}
    R = (sub["pnl_pct"].astype(float) / 100.0) / float(risk_per_trade_pct)
    eps = _EPS_DEN
    nz = R[R.abs() > eps]
    trade_count = int(len(sub))
    if nz.empty:
        win_rate = float("nan"); pf = float("nan")
    else:
        wins = int((nz > 0).sum())
        total = int(len(nz))
        win_rate = float(wins / total) if total > 0 else float("nan")
        pos_sum = float(nz[nz > 0].sum()) if wins > 0 else 0.0
        neg_sum = float(-nz[nz < 0].sum()) if (total - wins) > 0 else 0.0
        if neg_sum < eps and pos_sum > eps:
            pf = float("inf")
        elif pos_sum < eps and neg_sum > eps:
            pf = 0.0
        elif neg_sum < eps and pos_sum < eps:
            pf = float("nan")
        else:
            pf = float(pos_sum / neg_sum)
    return {"trade_count": trade_count, "win_rate": win_rate, "profit_factor": pf}

def _yearly_summary_from_monthlies(monthly_df: pd.DataFrame, trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    years = sorted(monthly_df["year"].unique())
    last_year = max(years)
    rows = []
    for y in years:
        grp = monthly_df.loc[monthly_df["year"] == y]
        r = grp["monthly_return"].astype(float).to_numpy()
        act = grp["active_month"].astype(bool).to_numpy()
        months_in_year = int(len(grp))
        annual_return = float(np.prod(1.0 + r) - 1.0) if months_in_year > 0 else float("nan")

        # Drawdowns (minimal set): EoM + RAW
        eom_mdd_y = _monthly_underwater_metrics_with_trough(r)[0]
        intramonth_mdd_y = _maxdd_from_intramonth_equity_year(equity_df, y)

        vol_ann = _safe_std(r, ddof=cfg.stdev_ddof) * _ann_factor()
        sharpe_ann = _sharpe_annualized(r, ddof=cfg.stdev_ddof)
        sortino_ann, insufficient_negs = _sortino_annualized(r, ddof=cfg.stdev_ddof)
        calmar = _calmar(annual_return, eom_mdd_y)

        pos = int((r > 0).sum())
        neg = int((r < 0).sum())

        # Active-only subset (monthly)
        r_active = r[act]
        active_count = int(act.sum())
        active_share = float(active_count / 12.0)
        if active_count > 0:
            ann_ret_active = float(np.prod(1.0 + r_active) - 1.0)
            vol_active = _safe_std(r_active, ddof=cfg.stdev_ddof) * _ann_factor()
            sharpe_active = _sharpe_annualized(r_active, ddof=cfg.stdev_ddof)
            sortino_active, _ = _sortino_annualized(r_active, ddof=cfg.stdev_ddof)
            wm_active = float(np.prod(1.0 + r_active))
            cagr_active = float(wm_active ** (12.0 / max(active_count, 1)) - 1.0)
        else:
            ann_ret_active = float("nan"); vol_active = float("nan")
            sharpe_active = float("nan"); sortino_active = float("nan"); cagr_active = float("nan")

        # Flags
        insufficient_months = months_in_year < 12
        insufficient_active_months = active_count < 12
        trade_metrics = _compute_yearly_trade_metrics(trades_df, y, cfg.risk_per_trade_pct)
        insufficient_trades = trade_metrics["trade_count"] < 12
        is_ytd = (y == last_year) and insufficient_months

        rows.append({
            "year": int(y),
            "annual_return_calendar": annual_return,

            # Drawdowns (minimal set)
            "eom_max_drawdown_intra_year": eom_mdd_y,
            "intramonth_max_drawdown_intra_year": intramonth_mdd_y,

            # Trades
            "trade_count": int(trade_metrics["trade_count"]),
            "win_rate": trade_metrics["win_rate"],
            "profit_factor": trade_metrics["profit_factor"],

            # Active-only
            "active_months_count": active_count,
            "active_months_share": active_share,
            "annual_return_active": ann_ret_active,
            "volatility_active_annualized": vol_active,
            "sharpe_active_annualized": sharpe_active,
            "sortino_active_annualized": sortino_active,
            "cagr_active": cagr_active,

            # Flags
            "insufficient_months": bool(insufficient_months),
            "insufficient_active_months": bool(insufficient_active_months),
            "insufficient_negative_months": bool(insufficient_negs),
            "insufficient_trades": bool(insufficient_trades),

            # Yearly risk/return (monthly-based)
            "volatility_annualized_year": vol_ann,
            "sharpe_ratio_annualized_year": sharpe_ann,
            "sortino_ratio_annualized_year": sortino_ann,
            "calmar_ratio_year": calmar,

            # Counts
            "negative_months_in_year": neg,
            "months_in_year_available": months_in_year,
            "is_ytd": bool(is_ytd),
            "positive_months_in_year": pos,
        })
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out

# ----------------------------- trades_full_period_summary -----------------------------

def _trades_full_period_summary(trades_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = ["trade_count","win_rate","profit_factor","average_winning_trade_r","average_losing_trade_r",
            "payoff_ratio","expectancy_mean_r","expectancy_median_r","r_std_dev","r_min","r_max"]
    if trades_df is None or len(trades_df)==0:
        return pd.DataFrame([{c: (0 if c=="trade_count" else float("nan")) for c in cols}])
    R = (trades_df["pnl_pct"].astype(float)/100.0)/float(cfg.risk_per_trade_pct)
    trade_count = int(len(R))
    eps = _EPS_DEN
    nz = R[R.abs()>eps]
    if len(nz)==0:
        win_rate = float("nan"); pos_sum = neg_sum = 0.0; avg_win = avg_loss = float("nan"); payoff = float("nan")
    else:
        win_rate = float((nz>0).sum()/len(nz))
        pos = nz[nz>0]; neg = nz[nz<0]
        pos_sum = float(pos.sum()) if len(pos)>0 else 0.0
        neg_sum = float(-neg.sum()) if len(neg)>0 else 0.0
        avg_win = float(pos.mean()) if len(pos)>0 else float("nan")
        avg_loss = float(neg.mean()) if len(neg)>0 else float("nan")  # negative
        if len(neg)==0 and len(pos)>0: payoff = float("inf")
        elif len(pos)==0 and len(neg)>0: payoff = 0.0
        elif len(pos)==0 and len(neg)==0: payoff = float("nan")
        else: payoff = float(avg_win/abs(avg_loss)) if abs(avg_loss)>eps else float("inf")
    if neg_sum<_EPS_DEN and pos_sum>_EPS_DEN: pf = float("inf")
    elif pos_sum<_EPS_DEN and neg_sum>_EPS_DEN: pf = 0.0
    elif neg_sum<_EPS_DEN and pos_sum<_EPS_DEN: pf = float("nan")
    else: pf = float(pos_sum/neg_sum)
    exp_mean = float(R.mean()) if trade_count>0 else float("nan")
    exp_median = float(R.median()) if trade_count>0 else float("nan")
    r_std = float(np.std(R, ddof=cfg.stdev_ddof)) if trade_count>cfg.stdev_ddof else float("nan")
    r_min = float(R.min()) if trade_count>0 else float("nan")
    r_max = float(R.max()) if trade_count>0 else float("nan")
    row = {"trade_count": trade_count, "win_rate": win_rate, "profit_factor": pf,
           "average_winning_trade_r": avg_win, "average_losing_trade_r": avg_loss, "payoff_ratio": payoff,
           "expectancy_mean_r": exp_mean, "expectancy_median_r": exp_median, "r_std_dev": r_std, "r_min": r_min, "r_max": r_max}
    return pd.DataFrame([row])

# ----------------------------- full_period_summary (extended) -----------------------------

def _ulcer_index_from_r(r: np.ndarray) -> float:
    dd = _eom_dd_series(r)
    neg = dd[dd < 0.0]
    if neg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(neg**2)))

def _martin_ratio(cagr: float, ulcer: float) -> float:
    if not np.isfinite(cagr) or not np.isfinite(ulcer) or abs(ulcer) < _EPS_DEN:
        if abs(cagr) < _EPS_DEN:
            return float("nan")
        return float("inf") if cagr > 0 else float("-inf")
    return float(cagr / ulcer)

def _skew_kurt_excess(r: np.ndarray) -> Tuple[float, float]:
    n = r.size
    if n < 3:
        return float("nan"), float("nan")
    mu = float(np.mean(r))
    s = float(np.std(r, ddof=1))
    if not np.isfinite(s) or abs(s) < _EPS_DEN:
        return float("nan"), float("nan")
    m = r - mu
    m3 = float(np.mean(m**3))
    m4 = float(np.mean(m**4))
    skew = m3 / (s**3)
    kurt_excess = m4 / (s**4) - 3.0
    return float(skew), float(kurt_excess)

def _max_consecutive_counts(r: np.ndarray) -> Tuple[int, int]:
    max_up = 0; cur_up = 0
    max_down = 0; cur_down = 0
    for x in r:
        if x > 0:
            cur_up += 1; max_up = max(max_up, cur_up)
            cur_down = 0
        elif x < 0:
            cur_down += 1; max_down = max(max_down, cur_down)
            cur_up = 0
        else:
            cur_up = 0; cur_down = 0
    return int(max_up), int(max_down)

def _newey_west_t_p_for_mean(r: np.ndarray, L: int) -> Tuple[float, float]:
    import math as _m
    n = r.size
    if n == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(r))
    x = r - mu
    nfloat = float(n)
    g0 = float(np.dot(x, x) / nfloat)
    L_eff = int(min(max(L,0), max(n-1, 0)))
    s_longrun = g0
    for l in range(1, L_eff+1):
        w = 1.0 - l / (L_eff + 1.0)  # Bartlett
        gl = float(np.dot(x[l:], x[:-l]) / nfloat)
        s_longrun += 2.0 * w * gl
    var_mean = s_longrun / nfloat
    if not np.isfinite(var_mean) or var_mean <= _EPS_DEN:
        return float("nan"), float("nan")
    t = mu / math.sqrt(var_mean)
    p = _m.erfc(abs(t) / _m.sqrt(2.0))
    return float(t), float(p)

def _full_period_summary(monthly_df: pd.DataFrame, yearly_df: pd.DataFrame, trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    names = FULL_PERIOD_SUMMARY_SCHEMA

    r = monthly_df["monthly_return"].astype(float).to_numpy()
    months = int(len(r))
    pos = int((r > 0).sum())
    neg = int((r < 0).sum())
    zero = int((abs(r) <= _EPS_DEN).sum())

    # EoM-based risk/return
    eom_maxdd, eom_longest, eom_ttr, eom_trough_idx, eom_since_trough = _monthly_underwater_metrics_with_trough(r)
    sd = _safe_std(r, ddof=cfg.stdev_ddof)
    inst_vol_ann = sd * _ann_factor()
    inst_sharpe = _sharpe_annualized(r, ddof=cfg.stdev_ddof)
    inst_sortino, _ = _sortino_annualized(r, ddof=cfg.stdev_ddof)
    wm = float(np.prod(1.0 + r)) if months > 0 else float("nan")
    total_ret = wm - 1.0 if (wm==wm) else float("nan")
    inst_cagr = (wm ** (12.0 / months) - 1.0) if months > 0 and (wm==wm) else float("nan")
    inst_calmar = _calmar(inst_cagr, eom_maxdd) if (inst_cagr==inst_cagr) else float("nan")

    # Extended stats
    ulcer = _ulcer_index_from_r(r)
    martin = _martin_ratio(inst_cagr, ulcer)
    ddev_ann = _downside_std(r, ddof=cfg.stdev_ddof) * _ann_factor()
    skew, kurt_ex = _skew_kurt_excess(r)
    max_up, max_down = _max_consecutive_counts(r)
    nw_t, nw_p = _newey_west_t_p_for_mean(r, cfg.nw_lag)

    # Intramonth (path-level) metrics
    intramonth_maxdd = _maxdd_from_intramonth_equity_full_period(equity_df)
    intramonth_longest, intramonth_ttr = _intramonth_underwater_metrics_months(equity_df)
    if equity_df is None or len(equity_df) == 0:
        intramonth_since_trough = float("nan")
    else:
        eq = equity_df[["time_utc","capital_ccy"]].copy().sort_values("time_utc", kind="mergesort")
        vals = eq["capital_ccy"].astype(float).to_numpy()
        peaks = np.maximum.accumulate(vals)
        dd = vals / peaks - 1.0
        trough_i = int(np.argmin(dd)) if len(dd)>0 else 0
        intramonth_since_trough = float(_months_between(eq["time_utc"].iloc[trough_i], eq["time_utc"].iloc[-1])) if len(dd)>0 else float("nan")

    # Active-only (monthly)
    act = monthly_df["active_month"].astype(bool).to_numpy()
    active_count = int(act.sum())
    active_share = float(active_count / months) if months > 0 else float("nan")
    r_active = r[act]
    if active_count > 0:
        sd_a = _safe_std(r_active, ddof=cfg.stdev_ddof)
        vol_a = sd_a * _ann_factor()
        sharpe_a = _sharpe_annualized(r_active, ddof=cfg.stdev_ddof)
        sortino_a, _ = _sortino_annualized(r_active, ddof=cfg.stdev_ddof)
        wm_a = float(np.prod(1.0 + r_active))
        cagr_a = float(wm_a ** (12.0 / max(active_count, 1)) - 1.0)
    else:
        vol_a = float("nan"); sharpe_a = float("nan"); sortino_a = float("nan"); cagr_a = float("nan")

    ending_nav = float(cfg.starting_nav * wm) if (wm==wm) else float("nan")
    years_cov = int(len(sorted(monthly_df["year"].unique())))
    start_year = int(monthly_df["year"].iloc[0]); start_month = int(monthly_df["month"].iloc[0])
    end_year = int(monthly_df["year"].iloc[-1]); end_month = int(monthly_df["month"].iloc[-1])
    period_start_month = f"{start_year:04d}-{start_month:02d}"
    period_end_month = f"{end_year:04d}-{end_month:02d}"
    best_month = float(np.nanmax(r)) if months > 0 else float("nan")
    worst_month = float(np.nanmin(r)) if months > 0 else float("nan")
    mean_month = float(np.mean(r)) if months > 0 else float("nan")
    median_month = float(np.median(r)) if months > 0 else float("nan")

    if yearly_df is not None and len(yearly_df) > 0:
        yr = yearly_df.dropna(subset=["annual_return_calendar"])
        if len(yr) > 0:
            i_best = int(yr["annual_return_calendar"].idxmax()); i_worst = int(yr["annual_return_calendar"].idxmin())
            best_year_ret = float(yr.loc[i_best, "annual_return_calendar"]); best_year = int(yr.loc[i_best, "year"])
            worst_year_ret = float(yr.loc[i_worst, "annual_return_calendar"]); worst_year = int(yr.loc[i_worst, "year"])
        else:
            best_year_ret = float("nan"); best_year = int(np.nan); worst_year_ret = float("nan"); worst_year = int(np.nan)
    else:
        best_year_ret = float("nan"); best_year = int(np.nan); worst_year_ret = float("nan"); worst_year = int(np.nan)

    trade_count_full = int(len(trades_df)) if trades_df is not None else 0

    values = [
        months, inst_cagr, inst_vol_ann, inst_sharpe, inst_sortino, inst_calmar,
        ulcer, martin, ddev_ann, skew, kurt_ex, max_up, max_down, nw_t, nw_p,
        eom_maxdd, int(eom_longest), eom_ttr, eom_since_trough,
        intramonth_maxdd, intramonth_longest, intramonth_ttr, intramonth_since_trough,
        active_count, active_share, vol_a, sharpe_a, sortino_a, cagr_a,
        neg, total_ret, wm, ending_nav, period_start_month, period_end_month, pos, best_month, worst_month,
        mean_month, median_month, zero, years_cov, best_year_ret, best_year, worst_year_ret, worst_year, trade_count_full,
    ]
    return pd.DataFrame([dict(zip(names, values))])

# ----------------------------- dd_quantiles_full_period -----------------------------

def _half_up_int(x: float) -> int:
    try:
        d = Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        return int(d)
    except Exception:
        return int(round(x))

def _dd_quantiles_full_period(monthly_df: pd.DataFrame) -> pd.DataFrame:
    r = monthly_df["monthly_return"].astype(float).to_numpy()
    if len(r) == 0:
        row = {
            "period_start_month": None,
            "period_end_month": None,
            "dd_observations_count": 0,
            "dd_episodes_count": 0,
            "drawdown_p90_full_period": float("nan"),
            "drawdown_p95_full_period": float("nan"),
            "drawdown_p99_full_period": float("nan"),
            "underwater_duration_p90_full_period": pd.NA,
            "underwater_duration_p95_full_period": pd.NA,
        }
        return pd.DataFrame([row])
    dd = _eom_dd_series(r)
    eps = _EPS_DEN
    # depth quantiles
    depths = np.abs(dd[dd < 0.0])
    dd_obs_cnt = int(depths.size)

    # episodes and durations
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    peak_change = np.r_[True, peak[1:] > peak[:-1]]
    peak_index = np.maximum.accumulate(np.where(peak_change, np.arange(len(eq)), 0))

    durations = []
    i = 0
    while i < len(dd):
        if dd[i] < -eps:
            j = i
            while j + 1 < len(dd) and dd[j + 1] < -eps:
                j += 1
            recovery_idx = j + 1 if (j + 1 < len(dd) and dd[j + 1] >= -eps) else None
            start_peak_i = int(peak_index[i])
            if recovery_idx is not None:
                dur = int(recovery_idx - start_peak_i + 1)
                durations.append(dur)
            i = j + 1
        else:
            i += 1
    dd_episodes_cnt = int(len(durations))

    def q_or_nan(arr: np.ndarray, q: float) -> float:
        return float(np.quantile(arr, q, method="linear")) if arr.size > 0 else float("nan")

    p90 = q_or_nan(depths, 0.90)
    p95 = q_or_nan(depths, 0.95)
    p99 = q_or_nan(depths, 0.99)

    if len(durations) > 0:
        dur_arr = np.array(durations, dtype=float)
        d90 = _half_up_int(float(np.quantile(dur_arr, 0.90, method="linear")))
        d95 = _half_up_int(float(np.quantile(dur_arr, 0.95, method="linear")))
    else:
        d90 = pd.NA
        d95 = pd.NA

    grid = pd.PeriodIndex(year=monthly_df["year"], month=monthly_df["month"], freq="M")
    period_start_month = f"{grid.min().year:04d}-{grid.min().month:02d}"
    period_end_month   = f"{grid.max().year:04d}-{grid.max().month:02d}"

    row = {
        "period_start_month": period_start_month,
        "period_end_month": period_end_month,
        "dd_observations_count": dd_obs_cnt,
        "dd_episodes_count": dd_episodes_cnt,
        "drawdown_p90_full_period": p90 if np.isfinite(p90) else float("nan"),
        "drawdown_p95_full_period": p95 if np.isfinite(p95) else float("nan"),
        "drawdown_p99_full_period": p99 if np.isfinite(p99) else float("nan"),
        "underwater_duration_p90_full_period": d90,
        "underwater_duration_p95_full_period": d95,
    }
    return pd.DataFrame([row])

# ----------------------------- rolling windows -----------------------------

def _window_stats(r_win: np.ndarray, active_win: np.ndarray, cfg: Config, annualize_for: Optional[int]=None) -> dict:
    # returns dict of stats for a window
    n = int(r_win.size)
    wm = float(np.prod(1.0 + r_win)) if n > 0 else float("nan")
    if annualize_for is None:
        ret = wm - 1.0
    else:
        ret = float(wm ** (12.0 / annualize_for) - 1.0)
    sd = _safe_std(r_win, ddof=cfg.stdev_ddof)
    vol = sd * _ann_factor()
    sharpe = _sharpe_annualized(r_win, ddof=cfg.stdev_ddof)
    sortino, insufficient_negs = _sortino_annualized(r_win, ddof=cfg.stdev_ddof)
    # MaxDD on window
    eom_mdd = _monthly_underwater_metrics_with_trough(r_win)[0]
    calmar = _calmar(ret, eom_mdd)
    pos = int((r_win > 0).sum())
    neg = int((r_win < 0).sum())
    zero = int((np.abs(r_win) <= _EPS_DEN).sum())
    pos_share = float(pos / n) if n>0 else float("nan")
    neg_share = float(neg / n) if n>0 else float("nan")
    active_cnt = int(active_win.sum())
    active_share = float(active_cnt / n) if n>0 else float("nan")
    return {
        "ret": ret, "vol": vol, "sharpe": sharpe, "sortino": sortino, "insuff_negs": bool(insufficient_negs),
        "mdd": eom_mdd, "calmar": calmar,
        "pos": pos, "neg": neg, "zero": zero, "pos_share": pos_share, "neg_share": neg_share,
        "active_cnt": active_cnt, "active_share": active_share
    }

def _rolling_tables(monthly_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 12m
    years = monthly_df["year"].to_numpy()
    months = monthly_df["month"].to_numpy()
    r = monthly_df["monthly_return"].astype(float).to_numpy()
    active = monthly_df["active_month"].astype(bool).to_numpy()
    n = len(r)
    rows12 = []
    rows36 = []
    for i in range(n):
        # 12m windows
        if i >= 11:
            r_win = r[i-11:i+1]
            a_win = active[i-11:i+1]
            st = f"{int(years[i-11]):04d}-{int(months[i-11]):02d}"
            en = f"{int(years[i]):04d}-{int(months[i]):02d}"
            stats = _window_stats(r_win, a_win, cfg, annualize_for=None)
            rows12.append({
                "window_start_month": st, "window_end_month": en, "months_in_window": 12,
                "rolling_return_12m": stats["ret"], "rolling_volatility_annualized_12m": stats["vol"],
                "rolling_sharpe_annualized_12m": stats["sharpe"], "rolling_sortino_annualized_12m": stats["sortino"],
                "rolling_max_drawdown_12m": stats["mdd"], "rolling_calmar_12m": stats["calmar"],
                "positive_months_in_window": stats["pos"], "negative_months_in_window": stats["neg"], "zero_months_in_window": stats["zero"],
                "positive_months_share": stats["pos_share"], "negative_months_share": stats["neg_share"],
                "active_months_count": stats["active_cnt"], "active_months_share": stats["active_share"],
                "insufficient_months": False, "insufficient_negative_months": stats["insuff_negs"]
            })
        # 36m windows
        if i >= 35:
            r_win = r[i-35:i+1]
            a_win = active[i-35:i+1]
            st = f"{int(years[i-35]):04d}-{int(months[i-35]):02d}"
            en = f"{int(years[i]):04d}-{int(months[i]):02d}"
            stats = _window_stats(r_win, a_win, cfg, annualize_for=36)
            rows36.append({
                "window_start_month": st, "window_end_month": en, "months_in_window": 36,
                "rolling_return_annualized_36m": stats["ret"], "rolling_volatility_annualized_36m": stats["vol"],
                "rolling_sharpe_annualized_36m": stats["sharpe"], "rolling_sortino_annualized_36m": stats["sortino"],
                "rolling_max_drawdown_36m": stats["mdd"], "rolling_calmar_36m": stats["calmar"],
                "positive_months_in_window": stats["pos"], "negative_months_in_window": stats["neg"], "zero_months_in_window": stats["zero"],
                "positive_months_share": stats["pos_share"], "negative_months_share": stats["neg_share"],
                "active_months_count": stats["active_cnt"], "active_months_share": stats["active_share"],
                "insufficient_months": False, "insufficient_negative_months": stats["insuff_negs"]
            })
    df12 = pd.DataFrame(rows12)
    df36 = pd.DataFrame(rows36)
    return df12, df36

# ----------------------------- Monte-Carlo stationary bootstrap -----------------------------

def _stationary_bootstrap_paths(r: np.ndarray, H: int, L: int, n_paths: int, seed: int) -> np.ndarray:
    """Return array of shape (n_paths, H) with bootstrap monthly returns."""
    if len(r) == 0 or H <= 0 or n_paths <= 0:
        return np.zeros((0, 0), dtype=float)
    rng = random.Random(seed)
    n = len(r)
    p = 1.0 / max(L, 1)
    paths = np.zeros((n_paths, H), dtype=float)
    for i in range(n_paths):
        # initial starting index
        idx = rng.randrange(0, n)
        for t in range(H):
            if t == 0:
                idx = rng.randrange(0, n)
            else:
                if rng.random() < p:
                    idx = rng.randrange(0, n)
                else:
                    idx = (idx + 1) % n
            paths[i, t] = r[idx]
    return paths

def _maxdd_magnitude_from_r(r: np.ndarray) -> float:
    if r.size == 0:
        return float("nan")
    dd = _eom_dd_series(r)
    mdd = float(np.min(dd))
    return float(abs(mdd)) if np.isfinite(mdd) else float("nan")

def _mc_summary(monthly_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    r = monthly_df["monthly_return"].astype(float).to_numpy()
    grid = pd.PeriodIndex(year=monthly_df["year"], month=monthly_df["month"], freq="M")
    period_start_month = f"{grid.min().year:04d}-{grid.min().month:02d}" if len(grid)>0 else None
    period_end_month   = f"{grid.max().year:04d}-{grid.max().month:02d}" if len(grid)>0 else None
    horizons = []
    for h in cfg.mc_horizons:
        if h == "full":
            horizons.append(len(r))
        else:
            try:
                horizons.append(int(h))
            except:
                continue
    horizons = [h for h in horizons if h>0]
    rows = []
    for H in horizons:
        paths = _stationary_bootstrap_paths(r, H, cfg.mc_L, cfg.mc_paths, cfg.mc_seed)
        if paths.size == 0:
            wm_p05 = wm_p50 = wm_p95 = float("nan")
            en_p05 = en_p50 = en_p95 = float("nan")
            cagr_p05 = cagr_p50 = cagr_p95 = float("nan")
            mdd_p95 = mdd_p99 = float("nan")
            prob_neg = float("nan")
        else:
            WM = np.prod(1.0 + paths, axis=1)
            EN = cfg.starting_nav * WM
            # CAGR annualized for horizon H
            CAGR = WM ** (12.0 / H) - 1.0
            # MaxDD magnitude
            MDD = np.apply_along_axis(_maxdd_magnitude_from_r, 1, paths)
            wm_p05, wm_p50, wm_p95 = np.quantile(WM, [0.05, 0.50, 0.95], method="linear")
            en_p05, en_p50, en_p95 = np.quantile(EN, [0.05, 0.50, 0.95], method="linear")
            cagr_p05, cagr_p50, cagr_p95 = np.quantile(CAGR, [0.05, 0.50, 0.95], method="linear")
            mdd_p95, mdd_p99 = np.quantile(MDD, [0.95, 0.99], method="linear")
            prob_neg = float(np.mean(WM - 1.0 < 0.0))
        row = {
            "method": "stationary_bootstrap",
            "block_mean_length_months": int(cfg.mc_L),
            "n_paths": int(cfg.mc_paths),
            "seed": int(cfg.mc_seed),
            "horizon_months": int(H),
            "period_start_month": period_start_month,
            "period_end_month": period_end_month,
            "wealth_multiple_p05": float(wm_p05), "wealth_multiple_p50": float(wm_p50), "wealth_multiple_p95": float(wm_p95),
            "ending_nav_p05": float(en_p05), "ending_nav_p50": float(en_p50), "ending_nav_p95": float(en_p95),
            "cagr_annualized_p05": float(cagr_p05), "cagr_annualized_p50": float(cagr_p50), "cagr_annualized_p95": float(cagr_p95),
            "max_drawdown_magnitude_p95": float(mdd_p95), "max_drawdown_magnitude_p99": float(mdd_p99),
            "prob_negative_horizon_return": float(prob_neg)
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ----------------------------- Writer & rounding -----------------------------

def _as_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy(deep=True)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("Int64")
    return out

def _as_bool(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy(deep=True)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("boolean")
    return out

def _validate_and_order(df: pd.DataFrame, schema_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in schema_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Output is missing required columns: {missing}")
    return df.loc[:, schema_cols]

def _fmt_half_up(x, nd):
    if pd.isna(x):
        return x
    try:
        xf = float(x)
    except Exception:
        return x
    if np.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    quant = Decimal('1').scaleb(-nd) if nd > 0 else Decimal('1')
    try:
        d = Decimal(str(xf)).quantize(quant, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return x
    if d == 0:
        d = abs(d)  # normalize -0 to +0
    return f"{d:.{nd}f}" if nd > 0 else f"{d:.0f}"

def _round_and_write_csv(df: pd.DataFrame, out_path: Path, schema_cols: list[str], decimals: dict, int_cols: list[str], bool_cols: list[str] = None) -> None:
    tmp = df.copy(deep=True)
    if int_cols:
        tmp = _as_int(tmp, int_cols)
    if bool_cols:
        tmp = _as_bool(tmp, bool_cols)
    tmp = _validate_and_order(tmp, schema_cols)
    for col, nd in decimals.items():
        if col in tmp.columns:
            tmp[col] = tmp[col].apply(lambda v, _nd=nd: _fmt_half_up(v, _nd))
    out_path.write_text(tmp.to_csv(index=False), encoding="utf-8")

# ----------------------------- CLI -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compounding_base_metrics_calculator",
        description="Compounding metrics — monthly/yearly/full, DD quantiles, rolling 12m/36m, Monte-Carlo."
    )
    p.add_argument("--risk-pct", type=float, required=True,
                   help="Risk per trade in PERCENT (e.g., 1.0 for 1%, 1.5 for 1.5%). Used for R calculations.")
    p.add_argument("--input-dir", type=str, default="input", help="Input directory (default: ./input)")
    p.add_argument("--output-dir", type=str, default="output", help="Output directory (default: ./output)")
    p.add_argument("--nw-lag", type=int, default=6, help="Newey–West lag (default: 6)")
    p.add_argument("--mc-l", type=int, default=6, help="Stationary bootstrap mean block length L (default: 6)")
    p.add_argument("--mc-paths", type=int, default=10000, help="Monte-Carlo bootstrap number of paths (default: 10000)")
    p.add_argument("--mc-seed", type=int, default=42, help="Monte-Carlo random seed (default: 42)")
    p.add_argument("--mc-horizons", type=str, default="12,36,full", help="Comma-separated horizons: e.g., 12,36,full")
    return p

# ----------------------------- Main -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.risk_pct <= 0 or args.risk_pct > 100:
        print("ERROR: --risk-pct must be in (0, 100]. Example: 1.0 or 1.5", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    years = discover_years_from_input(input_dir)
    if not years:
        print("ERROR: No input files found. Expected trades_YYYY.* and equity_YYYY.* in input/.", file=sys.stderr)
        return 2

    cfg = Config(
        risk_per_trade_pct=args.risk_pct / 100.0,
        nw_lag=int(args.nw_lag),
        mc_L=int(args.mc_l),
        mc_paths=int(args.mc_paths),
        mc_seed=int(args.mc_seed),
        mc_horizons=tuple(x.strip() for x in args.mc_horizons.split(",") if x.strip())
    )

    print(f"""
=== compounding_base_metrics_calculator v{__version__} ===
Risk per trade (R): {args.risk_pct}%  (decimal {cfg.risk_per_trade_pct:.4f})
NW lag: {cfg.nw_lag}
MC: method=stationary_bootstrap L={cfg.mc_L} paths={cfg.mc_paths} seed={cfg.mc_seed} horizons={list(cfg.mc_horizons)}
Input dir:  {input_dir}
Output dir: {output_dir}
Years (auto): {years}
""")

    # Load inputs across all years
    all_eq: list[pd.DataFrame] = []
    all_tr: list[pd.DataFrame] = []
    for y in years:
        t, e = find_input_files_for_year(input_dir, y)
        if t is None or e is None:
            print(f"[{y}] SKIP — missing file(s): trades={t is not None}, equity={e is not None}")
            continue
        eq_df = load_equity_minimal(e)
        tr_df = load_trades_minimal(t)
        all_eq.append(eq_df)
        all_tr.append(tr_df)

    if not all_eq:
        print("ERROR: No equity data loaded; cannot compute monthly returns.", file=sys.stderr)
        return 2

    if not all_tr:
        tr_all = pd.DataFrame({
            "open_ts_utc": pd.Series([], dtype="datetime64[ns]"),
            "close_ts_utc": pd.Series([], dtype="datetime64[ns]"),
            "pnl_abs": pd.Series([], dtype="float64"),
            "pnl_pct": pd.Series([], dtype="float64"),
        })
    else:
        tr_all = pd.concat(all_tr, ignore_index=True)

    eq_all = pd.concat(all_eq, ignore_index=True)

    # Step 1–2: monthly_returns (safe build + sanitize)
    eom_nav = _eom_nav_from_equity_no_sort(eq_all)
    grid, nav, prev_nav, r_m = _compute_monthly_returns_and_nav(eom_nav, starting_nav=cfg.starting_nav)

    # Active flags and zero-month sanitize (no trades & same NAV -> r=0)
    active_flags = _active_month_flags(tr_all, grid)
    same_nav = nav.eq(prev_nav)
    sanitize_mask = (~active_flags.reindex(grid, fill_value=False)) & same_nav
    r_m.loc[sanitize_mask] = 0.0

    if r_m.isna().any():
        bad_idx = r_m.index[r_m.isna()]
        raise RuntimeError(f"Monthly returns contain NaN at periods: {list(map(str, bad_idx))}")

    monthly_out = pd.DataFrame({
        "year": pd.Series([int(p.year) for p in grid], dtype="int64"),
        "month": pd.Series([int(p.month) for p in grid], dtype="int64"),
        "monthly_return": pd.Series(r_m.values, dtype="float64"),
        "active_month": pd.Series(active_flags.reindex(grid).fillna(False).astype(bool).values, dtype="bool"),
    })

    #logging block (period stats)
    period_start_month = f"{grid.min().year:04d}-{grid.min().month:02d}"
    period_end_month   = f"{grid.max().year:04d}-{grid.max().month:02d}"
    months_cnt = len(grid)
    active_cnt = int(monthly_out["active_month"].sum())
    inactive_cnt = int(months_cnt - active_cnt)
    zero_cnt = int((monthly_out["monthly_return"].abs() <= _EPS_DEN).sum())
    wm = float(np.prod(1.0 + monthly_out["monthly_return"].to_numpy()))
    ending_nav = float(cfg.starting_nav * wm)
    print(f"[Stage-1] Period: {period_start_month} → {period_end_month} | months={months_cnt}, active={active_cnt}, inactive={inactive_cnt}, zero_ret={zero_cnt}")
    print(f"[Stage-1] WM={wm:.8f}  Ending NAV={ending_nav:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    #yearly_summary
    yearly_df = _yearly_summary_from_monthlies(monthly_out, tr_all, eq_all, cfg)

    #trades_full_period_summary
    trades_full = _trades_full_period_summary(tr_all, cfg)

    #full_period_summary (extended)
    full_df = _full_period_summary(monthly_out, yearly_df, tr_all, eq_all, cfg)

    #dd_quantiles_full_period
    ddq_df = _dd_quantiles_full_period(monthly_out)

    #rolling windows
    rolling12_df, rolling36_df = _rolling_tables(monthly_out, cfg)

    #Monte-Carlo
    mc_df = _mc_summary(monthly_out, cfg)

    # === WRITE ALL ARTIFACTS ===
    _round_and_write_csv(
        monthly_out,
        output_dir / "monthly_returns.csv",
        schema_cols=MONTHLY_RETURNS_SCHEMA,
        decimals={"monthly_return": 4},
        int_cols=["year","month"],
        bool_cols=["active_month"]
    )
    print(f"Wrote monthly_returns.csv  (rows={len(monthly_out)})  [SCHEMA OK]")

    _round_and_write_csv(
        yearly_df,
        output_dir / "yearly_summary.csv",
        schema_cols=YEARLY_SUMMARY_SCHEMA,
        decimals={
            "annual_return_calendar": 4,
            "eom_max_drawdown_intra_year": 4,
            "intramonth_max_drawdown_intra_year": 4,
            "win_rate": 4,
            "profit_factor": 2,
            "active_months_share": 4,
            "annual_return_active": 4,
            "cagr_active": 4,
            "volatility_active_annualized": 4,
            "sharpe_active_annualized": 2,
            "sortino_active_annualized": 2,
            "volatility_annualized_year": 4,
            "sharpe_ratio_annualized_year": 2,
            "sortino_ratio_annualized_year": 2,
            "calmar_ratio_year": 2
        },
        int_cols=["year","trade_count","active_months_count","negative_months_in_year","months_in_year_available","positive_months_in_year"],
        bool_cols=["insufficient_months","insufficient_active_months","insufficient_negative_months","insufficient_trades","is_ytd"]
    )
    print(f"Wrote yearly_summary.csv  (rows={len(yearly_df)})  [SCHEMA OK]")

    _round_and_write_csv(
        trades_full,
        output_dir / "trades_full_period_summary.csv",
        schema_cols=TRADES_FULL_PERIOD_SCHEMA,
        decimals={
            "win_rate": 4,
            "profit_factor": 2,
            "average_winning_trade_r": 4,
            "average_losing_trade_r": 4,
            "payoff_ratio": 2,
            "expectancy_mean_r": 4,
            "expectancy_median_r": 4,
            "r_std_dev": 4,
            "r_min": 4,
            "r_max": 4
        },
        int_cols=["trade_count"],
        bool_cols=[]
    )
    print("Wrote trades_full_period_summary.csv  (rows=1)  [SCHEMA OK]")

    _round_and_write_csv(
        full_df,
        output_dir / "full_period_summary.csv",
        schema_cols=FULL_PERIOD_SUMMARY_SCHEMA,
        decimals={
            "cagr_full_period": 4,
            "volatility_annualized_full_period": 4,
            "sharpe_ratio_annualized_full_period": 2,
            "sortino_ratio_annualized_full_period": 2,
            "ulcer_index_full_period": 4,
            "martin_ratio_full_period": 2,
            "downside_deviation_annualized_full_period": 4,
            "skewness_full_period": 2,
            "kurtosis_excess_full_period": 2,
            "newey_west_tstat_mean_monthly_return": 2,
            "newey_west_p_value_mean_monthly_return": 3,
            "eom_max_drawdown_full_period": 4,
            "calmar_ratio_full_period": 2,
            "intramonth_max_drawdown_full_period": 4,
            "active_months_share": 4,
            "volatility_active_annualized": 4,
            "sharpe_active_annualized": 2,
            "sortino_active_annualized": 2,
            "cagr_active": 4,
            "total_return_full_period": 4,
            "wealth_multiple_full_period": 4,
            "ending_nav_full_period": 0,
            "best_month_return_full_period": 4,
            "worst_month_return_full_period": 4,
            "mean_monthly_return_full_period": 4,
            "median_monthly_return_full_period": 4,
            "best_year_return_calendar_full_period": 4,
            "worst_year_return_calendar_full_period": 4
        },
        int_cols=[
            "months","max_consecutive_up_months_full_period","max_consecutive_down_months_full_period",
            "eom_longest_underwater_months","eom_time_to_recover_months","eom_months_since_maxdd_trough",
            "intramonth_longest_underwater_months","intramonth_time_to_recover_months","intramonth_months_since_maxdd_trough",
            "active_months_count","negative_months_count_full_period","positive_months_count_full_period",
            "zero_months_count_full_period","years_covered","best_year","worst_year","trade_count_full_period"
        ],
        bool_cols=[]
    )
    print("Wrote full_period_summary.csv  (rows=1)  [SCHEMA OK]")

    _round_and_write_csv(
        _dd_quantiles_full_period(monthly_out),
        output_dir / "dd_quantiles_full_period.csv",
        schema_cols=DD_QUANTILES_SCHEMA,
        decimals={
            "drawdown_p90_full_period": 4,
            "drawdown_p95_full_period": 4,
            "drawdown_p99_full_period": 4
        },
        int_cols=["dd_observations_count","dd_episodes_count","underwater_duration_p90_full_period","underwater_duration_p95_full_period"],
        bool_cols=[]
    )
    print("Wrote dd_quantiles_full_period.csv  (rows=1)  [SCHEMA OK]")

    _round_and_write_csv(
        rolling12_df,
        output_dir / "rolling_12m.csv",
        schema_cols=ROLLING_12M_SCHEMA,
        decimals={
            "rolling_return_12m": 4,
            "rolling_volatility_annualized_12m": 4,
            "rolling_sharpe_annualized_12m": 2,
            "rolling_sortino_annualized_12m": 2,
            "rolling_max_drawdown_12m": 4,
            "rolling_calmar_12m": 2,
            "positive_months_share": 4,
            "negative_months_share": 4,
            "active_months_share": 4
        },
        int_cols=["months_in_window","positive_months_in_window","negative_months_in_window","zero_months_in_window","active_months_count"],
        bool_cols=["insufficient_months","insufficient_negative_months"]
    )
    print(f"Wrote rolling_12m.csv  (rows={len(rolling12_df)})  [SCHEMA OK]")

    _round_and_write_csv(
        rolling36_df,
        output_dir / "rolling_36m.csv",
        schema_cols=ROLLING_36M_SCHEMA,
        decimals={
            "rolling_return_annualized_36m": 4,
            "rolling_volatility_annualized_36m": 4,
            "rolling_sharpe_annualized_36m": 2,
            "rolling_sortino_annualized_36m": 2,
            "rolling_max_drawdown_36m": 4,
            "rolling_calmar_36m": 2,
            "positive_months_share": 4,
            "negative_months_share": 4,
            "active_months_share": 4
        },
        int_cols=["months_in_window","positive_months_in_window","negative_months_in_window","zero_months_in_window","active_months_count"],
        bool_cols=["insufficient_months","insufficient_negative_months"]
    )
    print(f"Wrote rolling_36m.csv  (rows={len(rolling36_df)})  [SCHEMA OK]")

    _round_and_write_csv(
        mc_df,
        output_dir / "monte_carlo_summary.csv",
        schema_cols=MC_SCHEMA,
        decimals={
            "wealth_multiple_p05": 4, "wealth_multiple_p50": 4, "wealth_multiple_p95": 4,
            "ending_nav_p05": 0, "ending_nav_p50": 0, "ending_nav_p95": 0,
            "cagr_annualized_p05": 4, "cagr_annualized_p50": 4, "cagr_annualized_p95": 4,
            "max_drawdown_magnitude_p95": 4, "max_drawdown_magnitude_p99": 4,
            "prob_negative_horizon_return": 4
        },
        int_cols=["block_mean_length_months","n_paths","seed","horizon_months"],
        bool_cols=[]
    )
    print(f"Wrote monte_carlo_summary.csv  (rows={len(mc_df)})  [SCHEMA OK]")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

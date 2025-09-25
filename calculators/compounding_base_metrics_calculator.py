"""
compounding_base_metrics_calculator
----------------------------------
Single-file calculator for compounding base metrics.

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import sys
import re
import math

import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    starting_nav: float = 100_000.0
    rf: float = 0.0
    stdev_ddof: int = 1
    timezone: str = "UTC"
    risk_per_trade_pct: float = 0.01  # 1% by default


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
    return out


def load_trades_minimal(trades_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_path, sep=";")
    required = ["open_date", "open_time", "close_date", "close_time", "pnl_abs", "pnl_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Trades file {trades_path.name} missing columns: {missing}")
    out = df.loc[:, required].copy()
    out["open_ts_utc"] = out["open_date"].astype(str) + " " + out["open_time"].astype(str)
    out["close_ts_utc"] = out["close_date"].astype(str) + " " + out["close_time"].astype(str)
    out["close_ts_utc"] = pd.to_datetime(out["close_ts_utc"], utc=False, errors="raise")
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


def _compute_monthly_returns(eom_nav: pd.Series, starting_nav: float) -> pd.Series:
    grid = _build_month_grid(eom_nav)
    nav = eom_nav.reindex(grid).ffill()
    prev = nav.shift(1)
    if len(nav) > 0:
        prev.iloc[0] = starting_nav
    r_m = nav / prev - 1.0
    r_m.index = grid
    return r_m


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

def _sharpe_annualized(r, ddof):
    r = np.asarray(r, float)
    mu = float(np.mean(r)) if r.size else float("nan")
    sd = float(np.std(r, ddof=ddof)) if r.size > ddof else float("nan")
    if not np.isfinite(sd):
        return float("nan")
    if sd <= _EPS_DEN:
        if abs(mu) <= _EPS_ZERO:
            return float("nan")
        return float("inf") if mu > 0 else float("-inf")
    return (mu / sd) * math.sqrt(12.0)

def _downside_std(monthly_returns: np.ndarray, ddof: int) -> float:
    neg = monthly_returns[monthly_returns < 0.0]
    if neg.size < 2:
        return float("nan")
    return float(np.std(neg, ddof=ddof))

def _sortino_annualized(r, ddof):
    r = np.asarray(r, float)
    neg = r[r < 0.0]
    if neg.size < 2:
        return float("nan"), True  # insufficient_negative_months
    mu = float(np.mean(r))
    ds = float(np.std(neg, ddof=ddof))
    if not np.isfinite(ds):
        return float("nan"), True
    if ds <= _EPS_DEN:
        if abs(mu) <= _EPS_ZERO:
            return float("nan"), False
        return (float("inf") if mu > 0 else float("-inf")), False
    return (mu / ds) * math.sqrt(12.0), False


# ----------------------------- Institutional (EoM) underwater helpers -----------------------------

def _monthly_underwater_metrics_with_trough(r: np.ndarray) -> tuple[float, int, float, int, int]:
    """Return (maxdd, longest_under_months, ttr_months, trough_idx, months_since_trough_to_end) on monthly series r.
    Institutional convention with epsilon:
      - Longest underwater: PEAK -> first RECOVERY (dd >= -eps), including the recovery month.
      - TTR for MaxDD: trough -> first dd >= -eps (includes recovery month); NaN if no recovery.
    """
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
                run_len = int(recovery_idx - start_peak_i + 1)  # PEAK..RECOVERY inclusive
            else:
                run_len = int(j - start_peak_i + 1)             # PEAK..last negative
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
    """(longest_underwater_months, ttr_maxdd_months) from RAW equity across full period.
    With epsilon guard:
      - Longest: PEAK -> first RECOVERY (dd >= -eps), including recovery month.
      - TTR: trough -> first dd >= -eps (includes recovery month); NaN if нет восстановления.
    """
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
        months_in_year_available = months_in_year
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
        active_share = float(active_count / months_in_year_available) if months_in_year_available > 0 else float("nan")
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
        insufficient_months = months_in_year_available < 12
        insufficient_active_months = active_count < 6
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
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)



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


# ----------------------------- full_period_summary -----------------------------


def _full_period_summary(monthly_df: pd.DataFrame, yearly_df: pd.DataFrame, trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    names = [
        # Institutional risk/return (monthly-based, neutral names)
        "months","cagr_full_period","volatility_annualized_full_period","sharpe_ratio_annualized_full_period","sortino_ratio_annualized_full_period","calmar_ratio_full_period",
        # Institutional EoM drawdowns (explicit eom_* prefix)
        "eom_max_drawdown_full_period","eom_longest_underwater_months","eom_time_to_recover_months","eom_months_since_maxdd_trough",
        # Intramonth (path-level) set
        "intramonth_max_drawdown_full_period","intramonth_longest_underwater_months","intramonth_time_to_recover_months","intramonth_months_since_maxdd_trough",
        # Activity & distribution (monthly-based)
        "active_months_count","active_months_share","volatility_active_annualized","sharpe_active_annualized","sortino_active_annualized","cagr_active",
        "negative_months_count_full_period","total_return_full_period","wealth_multiple_full_period","ending_nav_full_period",
        "period_start_month","period_end_month","positive_months_count_full_period","best_month_return_full_period","worst_month_return_full_period",
        "mean_monthly_return_full_period","median_monthly_return_full_period","zero_months_count_full_period","years_covered",
        "best_year_return_calendar_full_period","best_year","worst_year_return_calendar_full_period","worst_year","trade_count_full_period",
    ]

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
        eom_maxdd, int(eom_longest), eom_ttr, eom_since_trough,
        intramonth_maxdd, intramonth_longest, intramonth_ttr, intramonth_since_trough,
        active_count, active_share, vol_a, sharpe_a, sortino_a, cagr_a,
        neg, total_ret, wm, ending_nav, period_start_month, period_end_month, pos, best_month, worst_month,
        mean_month, median_month, zero, years_cov, best_year_ret, best_year, worst_year_ret, worst_year, trade_count_full,
    ]
    return pd.DataFrame([dict(zip(names, values))])

def _round_cols(df: pd.DataFrame, decimals: dict) -> pd.DataFrame:
    out = df.copy(deep=True)
    def _half_up(x, nd):
        if pd.isna(x):
            return x
        try:
            xf = float(x)
        except Exception:
            return x
        if np.isinf(xf):
            return xf
        if nd <= 0:
            quant = Decimal('1')
        else:
            quant = Decimal('1').scaleb(-nd)  # 10^-nd
        try:
            d = Decimal(str(xf)).quantize(quant, rounding=ROUND_HALF_UP)
            return float(d)
        except InvalidOperation:
            return xf
    for col, nd in decimals.items():
        if col in out.columns:
            out[col] = out[col].apply(lambda v, _nd=nd: _half_up(v, _nd))
    return out
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

def _round_and_write_csv(df: pd.DataFrame, out_path: Path, decimals: dict, int_cols: list[str], bool_cols: list[str] = None) -> None:
    tmp = df.copy(deep=True)
    # Cast integers and booleans first
    if int_cols:
        tmp = _as_int(tmp, int_cols)
    if bool_cols:
        tmp = _as_bool(tmp, bool_cols)
    # Apply per-column Decimal(ROUND_HALF_UP) formatting to TEXT to guarantee CSV rendering
    def _fmt_val(x, nd):
        if pd.isna(x):
            return x  # keep NaN -> empty cell
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
        # fixed-width string with nd decimals
        return f"{d:.{nd}f}" if nd > 0 else f"{d:.0f}"
    for col, nd in decimals.items():
        if col in tmp.columns:
            tmp[col] = tmp[col].apply(lambda v, _nd=nd: _fmt_val(v, _nd))
    out_path.write_text(tmp.to_csv(index=False, na_rep="NaN"), encoding="utf-8")


# ----------------------------- CLI -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compounding_base_metrics_calculator",
        description="Compounding metrics — monthly & yearly steps (dual EoM/intramonth drawdown blocks)."
    )
    p.add_argument("--risk-pct", type=float, required=True,
                   help="Risk per trade in PERCENT (e.g., 1.0 for 1%, 1.5 for 1.5%). Used for R calculations.")
    p.add_argument("--input-dir", type=str, default="input", help="Input directory (default: ./input)")
    p.add_argument("--output-dir", type=str, default="output", help="Output directory (default: ./output)")
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

    cfg = Config(risk_per_trade_pct=args.risk_pct / 100.0)

    print(f"""
=== compounding_base_metrics_calculator ===
Risk per trade (R): {args.risk_pct}%  (decimal {cfg.risk_per_trade_pct:.4f})
Input dir:  {input_dir}
Output dir: {output_dir}
Years (auto): {years}
""")

    # Load inputs across all years
    all_eq = []
    all_tr = []
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

    eq_all = pd.concat(all_eq, ignore_index=True)
    tr_all = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame(columns=["open_ts_utc","close_ts_utc","pnl_abs","pnl_pct"])

    # monthly_returns
    eom_nav = _eom_nav_from_equity_no_sort(eq_all)
    grid = _build_month_grid(eom_nav)
    active_flags = _active_month_flags(tr_all, grid)
    r_m = _compute_monthly_returns(eom_nav, starting_nav=cfg.starting_nav)

    monthly_out = pd.DataFrame({
        "year": pd.Series([int(p.year) for p in grid], dtype="int64"),
        "month": pd.Series([int(p.month) for p in grid], dtype="int64"),
        "monthly_return": pd.Series(r_m.values, dtype="float64"),
        "active_month": pd.Series(active_flags.reindex(grid).fillna(False).astype(bool).values, dtype="bool"),
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    # yearly_summary
    yearly_df = _yearly_summary_from_monthlies(monthly_out, tr_all, eq_all, cfg)

    # trades_full_period_summary
    trades_full = _trades_full_period_summary(tr_all, cfg)

    # full_period_summary
    full_df = _full_period_summary(monthly_out, yearly_df, tr_all, eq_all, cfg)

    # === WRITE ALL ARTIFACTS (rounding only at write-time; inputs left untouched) ===
    _round_and_write_csv(
        monthly_out,
        output_dir / "monthly_returns.csv",
        decimals={"monthly_return": 4},
        int_cols=["year","month"],
        bool_cols=["active_month"]
    )
    print("Wrote monthly_returns.csv")

    _round_and_write_csv(
        yearly_df,
        output_dir / "yearly_summary.csv",
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
        int_cols=[
            "year","trade_count","active_months_count",
            "negative_months_in_year","months_in_year_available","positive_months_in_year"
        ],
        bool_cols=["insufficient_months","insufficient_active_months","insufficient_negative_months","insufficient_trades","is_ytd"]
    )
    print("Wrote yearly_summary.csv")

    _round_and_write_csv(
        trades_full,
        output_dir / "trades_full_period_summary.csv",
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
    print("Wrote trades_full_period_summary.csv")

    _round_and_write_csv(
        full_df,
        output_dir / "full_period_summary.csv",
        decimals={
            "cagr_full_period": 4,
            "volatility_annualized_full_period": 4,
            "sharpe_ratio_annualized_full_period": 2,
            "sortino_ratio_annualized_full_period": 2,
            "eom_max_drawdown_full_period": 4,
            "calmar_ratio_full_period": 2,
            "intramonth_max_drawdown_full_period": 4,
            "active_months_share": 4,
            "volatility_active_annualized": 4,
            "sharpe_active_annualized": 2,
            "sortino_active_annualized": 2,
            "cagr_active": 4,
            "total_return_full_period": 4,
            "wealth_multiple_full_period": 2,
            "ending_nav_full_period": 0,
            "best_month_return_full_period": 4,
            "worst_month_return_full_period": 4,
            "mean_monthly_return_full_period": 4,
            "median_monthly_return_full_period": 4,
            "best_year_return_calendar_full_period": 4,
            "worst_year_return_calendar_full_period": 4
        },
        int_cols=[
            "months","eom_longest_underwater_months","eom_time_to_recover_months","eom_months_since_maxdd_trough",
            "intramonth_longest_underwater_months","intramonth_time_to_recover_months","intramonth_months_since_maxdd_trough",
            "active_months_count","negative_months_count_full_period","positive_months_count_full_period",
            "zero_months_count_full_period","years_covered","best_year","worst_year","trade_count_full_period"
        ],
        bool_cols=[]
    )
    print("Wrote full_period_summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

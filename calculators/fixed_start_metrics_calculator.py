# fixed_start_metrics_calculator.py — fixed_start_100k metrics

import sys
import re
import argparse
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# ---------------- Args / Config ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Fixed-start 100k metrics calculator")
    p.add_argument("--risk-pct", type=float, default=1.0, help="Risk per trade as PERCENT (e.g., 1.5 for 1.5%)")
    p.add_argument("--allow-reconstruct-eom-from-trades", action="store_true", default=False,
                   help="If set, reconstruct month-end NAV from trades when equity EoM is missing")
    p.add_argument("--debug-mdd-year", type=int, default=None,
                   help="If set, dump MDD debug for this year to output/mdd_debug_<year>.csv")
    return p.parse_args()

ARGS = None
INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")
ANCHOR = 100_000.0

def _self_sha256() -> str:
    try:
        here = Path(__file__)
        return hashlib.sha256(here.read_bytes()).hexdigest()
    except Exception:
        return "unknown"

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

# ---------------- IO helpers ----------------
def sniff_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
    txt = "\n".join(sample)
    cands = [";", ",", "\t", "|"]
    best = max(cands, key=lambda s: txt.count(s))
    return best if txt.count(best) > 0 else ","

def read_csv_any(path: Path) -> pd.DataFrame:
    sep = sniff_sep(path)
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)

def to_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def parse_utc_series(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x.astype(str).str.strip(), errors="coerce", utc=True)

# ---------------- Column maps ----------------
def map_equity_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    df = to_lower_cols(df.copy())
    t = next((c for c in ["time_utc","timestamp","time","datetime","date"] if c in df.columns), None)
    v = next((c for c in ["capital_ccy","equity","nav","balance","capital"] if c in df.columns), None)
    return (t, v) if t and v else None

def map_trades_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    df = to_lower_cols(df.copy())
    date = next((c for c in ["close_date","date_close","closed_date","date"] if c in df.columns), None)
    time = next((c for c in ["close_time","time_close","closed_time","time"] if c in df.columns), None)
    single = next((c for c in ["close_datetime","closed_at","close_at","timestamp","datetime"] if c in df.columns), None)
    pnl_pct = next((c for c in ["pnl_pct","pnl_percent","pnl_perc","pnl%"] if c in df.columns), None)
    return {"date": date, "time": time, "single": single, "pnl_pct": pnl_pct}

# ---------------- Trades utils ----------------
def close_time_series(trades_df: pd.DataFrame) -> Optional[pd.Series]:
    m = map_trades_columns(trades_df)
    tr = trades_df.copy()
    if m["date"] and m["time"]:
        return parse_utc_series(tr[m["date"]].astype(str).str.strip() + " " + tr[m["time"]].astype(str).str.strip())
    if m["single"]:
        return parse_utc_series(tr[m["single"]])
    return None

def derive_r_from_trades(trades_df: pd.DataFrame, risk_pct: float) -> pd.Series:
    m = map_trades_columns(trades_df)
    tr = trades_df.copy()
    if not m["pnl_pct"]:
        eprint("[ERROR] trades: required column 'pnl_pct' not found (r = pnl_pct / risk_pct).")
        sys.exit(2)
    x = pd.to_numeric(tr[m["pnl_pct"]], errors="coerce")
    if risk_pct is None or risk_pct <= 0:
        eprint("[ERROR] --risk-pct must be > 0")
        sys.exit(2)
    return x / risk_pct

# ---------------- Equity monthly EoM ----------------
def month_eom_nav_from_equity(equity_df: Optional[pd.DataFrame], year: int) -> Dict[int, float]:
    if equity_df is None or equity_df.empty:
        return {}
    mapped = map_equity_columns(equity_df)
    if not mapped:
        return {}
    tcol, vcol = mapped
    eq = equity_df.copy()
    eq["_dt"] = parse_utc_series(eq[tcol])
    eq = eq.dropna(subset=["_dt"])
    eq = eq[eq["_dt"].dt.year == year]
    if eq.empty:
        return {}
    eq = eq.sort_values("_dt", kind="mergesort")  # keep raw order for ties
    eq["month"] = eq["_dt"].dt.month
    return eq.groupby("month")[vcol].last().astype(float).to_dict()

def active_month_flags(trades_df: Optional[pd.DataFrame], year: int) -> Dict[int, bool]:
    if trades_df is None or trades_df.empty:
        return {}
    tr = trades_df.copy()
    tr["_close"] = close_time_series(tr)
    tr = tr.dropna(subset=["_close"])
    tr = tr[tr["_close"].dt.year == year]
    if tr.empty:
        return {}
    tr = tr.sort_values("_close", kind="mergesort")
    tr["month"] = tr["_close"].dt.month
    return tr.groupby("month").size().astype(bool).to_dict()

def reconstruct_month_nav_from_trades(trades_df: Optional[pd.DataFrame], year: int, nav_start: float, risk_pct: float) -> Dict[int, float]:
    if trades_df is None or trades_df.empty:
        return {}
    tr = trades_df.copy()
    tr["_close"] = close_time_series(tr)
    tr = tr.dropna(subset=["_close"])
    tr = tr[tr["_close"].dt.year == year]
    if tr.empty:
        return {}
    tr["_r"] = derive_r_from_trades(tr, risk_pct)
    tr = tr.dropna(subset=["_r"])
    if tr.empty:
        return {}
    tr = tr.sort_values("_close", kind="mergesort")
    month_nav_end: Dict[int, float] = {}
    nav = float(nav_start)
    current_month = None
    for _, row in tr.iterrows():
        mt = int(row["_close"].month)
        r = float(row["_r"])
        if current_month is None:
            current_month = mt
        if mt != current_month:
            month_nav_end[current_month] = nav
            current_month = mt
        nav *= (1.0 + r * (risk_pct / 100.0))
    if current_month is not None:
        month_nav_end[current_month] = nav
    return month_nav_end

# ---------------- Raw equity timeline & MDD ----------------
def _raw_equity_timeline_for_year(equity_df: Optional[pd.DataFrame], year: int) -> pd.DataFrame:
    """All raw equity ticks inside the calendar year, strictly from equity, chronological, no dedup/ffill/anchor."""
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=["_dt", "E"])
    mapped = map_equity_columns(equity_df)
    if not mapped:
        return pd.DataFrame(columns=["_dt", "E"])
    tcol, vcol = mapped
    eq = equity_df.copy()
    eq["_dt"] = parse_utc_series(eq[tcol])
    eq = eq.dropna(subset=["_dt"])
    eq = eq[eq["_dt"].dt.year == year]
    if eq.empty:
        return pd.DataFrame(columns=["_dt", "E"])
    eq = eq.sort_values("_dt", kind="mergesort")
    return pd.DataFrame({
        "_dt": eq["_dt"].values,
        "E": pd.to_numeric(eq[vcol], errors="coerce").astype(float).values
    })

def _mdd_from_timeline(df: pd.DataFrame):
    """Compute MDD on raw timeline; returns (mdd_float, df_with_peak_dd)."""
    if df is None or df.empty:
        return float("nan"), df
    if len(df) == 1:
        tmp = df.copy()
        tmp["P"] = df["E"].iloc[0]
        tmp["DD"] = 0.0
        return 0.0, tmp
    peak = df["E"].iloc[0]
    P = []
    DD = []
    for e in df["E"].tolist():
        if e > peak:
            peak = e
        P.append(peak)
        DD.append((e / peak - 1.0) if peak > 0 else 0.0)
    tmp = df.copy()
    tmp["P"] = P
    tmp["DD"] = DD
    mdd = float(min(DD))
    return mdd, tmp

def _mdd_on_values(values: list) -> float:
    """MDD on a list of numbers: running peak P_t, DD_t=E_t/P_t-1, return min(DD_t)."""
    if values is None or len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return 0.0
    peak = values[0]
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
            dd = 0.0
        else:
            dd = (v / peak - 1.0) if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return float(mdd)

def _eom_equity_values_for_year(equity_df: Optional[pd.DataFrame], year: int) -> list:
    """Month-end equity values inside the year (last tick in month), months w/o data skipped."""
    eom_map = month_eom_nav_from_equity(equity_df, year)
    if not eom_map:
        return []
    months = sorted(eom_map.keys())
    return [float(eom_map[m]) for m in months]

# ---------------- Formatters ----------------
def fmt4(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{Decimal(str(x)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP):.4f}"

def fmt4_strict(x):
    """HALF_UP to 4 decimals with negative-zero normalization; returns string with exactly 4 decimals."""
    if x is None:
        return "0.0000"
    try:
        if pd.isna(x):
            return "0.0000"
    except Exception:
        pass
    d = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    s = f"{d:.4f}"
    return "0.0000" if s == "-0.0000" else s

def fmt2(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"

# ---------------- Utils ----------------
def extract_year_from_name(p: Path) -> Optional[int]:
    m = re.search(r"(\d{4})", p.name)
    return int(m.group(1)) if m else None

# ---------------- Rounding router (from JSON schema) ----------------
import fnmatch, math, json

def _load_rounding_policy(path: Path = Path("/mnt/data/advanced_metrics_schema.json")) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        return js.get("rounding_policy", {})
    except Exception:
        return {}

_RP = _load_rounding_policy()

def _match_rule(col: str) -> dict:
    # Use priority_order if available
    order = _RP.get("priority_order", [])
    keys = order if order else [k for k in _RP.keys() if k != "priority_order"]
    for key in keys:
        rule = _RP.get(key, {})
        exact = rule.get("applies_to", [])
        pats = rule.get("applies_to_patterns", [])
        if col in exact:
            return rule
        for pat in pats:
            if fnmatch.fnmatch(col, pat):
                return rule
    return {}

def _format_special(x):
    try:
        if x is None:
            return "NaN"
        if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
            if math.isnan(x): return "NaN"
            return "Inf" if x > 0 else "-Inf"
        if pd.isna(x):
            return "NaN"
    except Exception:
        pass
    return None

def format_by_router(col: str, x):
    special = _format_special(x)
    if special is not None:
        return special
    rule = _match_rule(col)
    decimals = int(rule.get("decimals", 4))
    q = "0." + ("0"*decimals)
    d = Decimal(str(x)).quantize(Decimal(q), rounding=ROUND_HALF_UP)
    s = f"{d:.{decimals}f}"
    # normalize negative zero
    if re.match(r"^-0\.0+$", s):
        s = "0." + ("0"*decimals)
    return s

# ---------------- Main ----------------
def main():
    global ARGS
    ARGS = parse_args()
    print(f"[VERSION] sha256={_self_sha256()} | risk_pct={ARGS.risk_pct}% | reconstruct={ARGS.allow_reconstruct_eom_from_trades}", file=sys.stderr)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years = set()
    equities: Dict[int, pd.DataFrame] = {}
    trades: Dict[int, pd.DataFrame] = {}

    for p in INPUT_DIR.glob("equity_*.csv"):
        y = extract_year_from_name(p)
        if y is None:
            eprint(f"[WARN] Skipping file without 4-digit year in name: {p.name}")
            continue
        equities[y] = read_csv_any(p)
        years.add(y)
    for p in INPUT_DIR.glob("trades_*.csv"):
        y = extract_year_from_name(p)
        if y is None:
            eprint(f"[WARN] Skipping file without 4-digit year in name: {p.name}")
            continue
        trades[y] = read_csv_any(p)
        years.add(y)
    years = sorted(years)

    monthly_all: List[Dict] = []
    summary_rows: List[Dict] = []

    for year in years:
        eq_df = equities.get(year)
        tr_df = trades.get(year)

        eom = month_eom_nav_from_equity(eq_df, year)
        amf = active_month_flags(tr_df, year)
        recon = reconstruct_month_nav_from_trades(tr_df, year, ANCHOR, ARGS.risk_pct) if ARGS.allow_reconstruct_eom_from_trades else {}

        nav_prev = ANCHOR
        months_avail_list: List[int] = []
        neg_count = 0
        pos_count = 0

        for m in range(1, 13):
            active = bool(amf.get(m, False))
            if m in eom:
                nav_eom = float(eom[m])
                months_avail_list.append(m)
            elif active and (m in recon):
                nav_eom = float(recon[m])
                months_avail_list.append(m)
            else:
                nav_eom = nav_prev  # month not available

            mr = (nav_eom / nav_prev) - 1.0 if nav_prev > 0 else 0.0
            if mr < 0:
                neg_count += 1
            elif mr > 0:
                pos_count += 1

            monthly_all.append({
                "year": year,
                "month": m,
                "monthly_return": mr,
                "active_month": bool(active)
            })
            nav_prev = nav_eom

        # --- Yearly summary availability/YTD logic ---
        months_with_any_data = set(eom.keys()) | set(amf.keys())
        current_year = pd.Timestamp.utcnow().year
        if not months_with_any_data:
            months_in_year_available = 0
            is_ytd = (year == current_year)
        elif year == current_year:
            months_in_year_available = int(len(months_with_any_data))
            is_ytd = months_in_year_available < 12
        else:
            months_in_year_available = 12
            is_ytd = False

        # Annual return from raw monthly returns (zeros for missing months are fine: ×1)
        mr_values = [r["monthly_return"] for r in monthly_all if r["year"] == year]
        annual_return = (np.prod([1.0 + x for x in mr_values]) - 1.0) if mr_values else 0.0

        # Raw MDD on all equity ticks & EoM MDD on month-end values
        timeline = _raw_equity_timeline_for_year(eq_df, year)
        raw_mdd, dbg = _mdd_from_timeline(timeline)
        eom_vals = _eom_equity_values_for_year(eq_df, year)
        eom_mdd = _mdd_on_values(eom_vals)

        # optional debug for raw MDD
        if ARGS.debug_mdd_year and ARGS.debug_mdd_year == year:
            dfdbg = dbg.copy()
            if not dfdbg.empty:
                dfdbg["DD_fmt"] = dfdbg["DD"].apply(fmt4)
                dfdbg["P_fmt"] = dfdbg["P"].apply(lambda x: f"{x:.4f}")
                dfdbg.to_csv(OUTPUT_DIR / f"mdd_debug_{year}.csv", index=False, na_rep="NaN")

        # Trades metrics
        if tr_df is None or tr_df.empty:
            trade_count, win_rate, pf = 0, None, None
        else:
            tr = tr_df.copy()
            tr["_close"] = close_time_series(tr)
            tr = tr.dropna(subset=["_close"])
            tr = tr[tr["_close"].dt.year == year]
            if tr.empty:
                trade_count, win_rate, pf = 0, None, None
            else:
                r = derive_r_from_trades(tr, ARGS.risk_pct)
                tr = tr.assign(_r=pd.to_numeric(r, errors="coerce")).dropna(subset=["_r"])
                trade_count = int(len(tr))
                nonzero = tr.loc[tr["_r"] != 0, "_r"]
                if len(nonzero) == 0:
                    win_rate = None
                    pf = None
                else:
                    wins = int((nonzero > 0).sum())
                    win_rate = float(wins) / float(len(nonzero))
                    pos_sum = tr.loc[tr["_r"] > 0, "_r"].sum()
                    neg_sum = tr.loc[tr["_r"] < 0, "_r"].sum()
                    eps_zero = 1e-12
                    pos_sum = float(pos_sum)
                    neg_sum = float(neg_sum)
                    if abs(neg_sum) < eps_zero and pos_sum > eps_zero:
                        pf = float('inf')
                    elif abs(neg_sum) >= eps_zero and pos_sum < eps_zero:
                        pf = 0.0
                    elif abs(neg_sum) < eps_zero and pos_sum < eps_zero:
                        pf = float('nan')
                    else:
                        pf = float(pos_sum / abs(neg_sum))

        active_months_count = int(sum(1 for m in range(1, 13) if bool(amf.get(m, False))))
        active_months_share = float(active_months_count / 12.0)

        summary_rows.append({
            "year": year,
            "annual_return_calendar": annual_return,
            "intramonth_max_drawdown_intra_year": raw_mdd,
            "eom_max_drawdown_intra_year": eom_mdd,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "profit_factor": pf,
            "active_months_count": active_months_count,
            "active_months_share": active_months_share,
            "insufficient_months": months_in_year_available < 12,
            "insufficient_active_months": active_months_count < 6,
            "insufficient_negative_months": neg_count < 2,
            "insufficient_trades": trade_count < 12,
            "negative_months_in_year": neg_count,
            "months_in_year_available": months_in_year_available,
            "is_ytd": is_ytd,
            "positive_months_in_year": pos_count
        })

    # Write yearly summary with schema, formatting
    ys = pd.DataFrame(summary_rows).sort_values(["year"]).reset_index(drop=True)
    cols = [
        "year",
        "annual_return_calendar",
        "eom_max_drawdown_intra_year",
        "intramonth_max_drawdown_intra_year",
        "trade_count",
        "win_rate",
        "profit_factor",
        "active_months_count",
        "active_months_share",
        "insufficient_months",
        "insufficient_active_months",
        "insufficient_negative_months",
        "insufficient_trades",
        "negative_months_in_year",
        "months_in_year_available",
        "is_ytd",
        "positive_months_in_year"
    ]
    for c in cols:
        if c not in ys.columns:
            ys[c] = None
    ys = ys[cols]

    # Apply HALF_UP formatting
    for c in ["annual_return_calendar", "intramonth_max_drawdown_intra_year", "eom_max_drawdown_intra_year", "active_months_share", "win_rate"]:
        ys[c] = ys[c].apply(lambda v, col=c: format_by_router(col, v))
    ys["profit_factor"] = ys["profit_factor"].apply(lambda v: format_by_router("profit_factor", v))

    for c in ["insufficient_months","insufficient_active_months","insufficient_negative_months","insufficient_trades","is_ytd"]:
        ys[c] = ys[c].astype(bool)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ys.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False, na_rep="NaN")

    # ---- Write monthly_returns.csv once (overwrite, no duplicates) ----
    monthly_df = pd.DataFrame(monthly_all)
    if not monthly_df.empty:
        monthly_df = monthly_df[["year","month","monthly_return","active_month"]].astype({"active_month": bool}).sort_values(["year","month"]).reset_index(drop=True)
        monthly_df["monthly_return"] = monthly_df["monthly_return"].apply(lambda v: format_by_router("monthly_return", v))
        monthly_df.to_csv(OUTPUT_DIR / "monthly_returns.csv", index=False, na_rep="NaN")

if __name__ == "__main__":
    main()

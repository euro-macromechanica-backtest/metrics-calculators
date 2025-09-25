
# Advanced Metrics (Institutional Extended Set)

**Schema ID:** `metrics_schema`  
**Version:** `metrics_schema-1.0`  

> This schema is a full metrics schema set intended for advanced institutional metrics (Advanced Metrics - Institutional Extended Set). It includes the base metrics core specification and expanded with additional analytics under the same UTC, ddof=1, NaN/Inf, and sample policies. Disambiguation conventions remain: 'Annualized, Full Period' != 'Calendar Year, Annualized'. Schema ID: metrics_schema.

_Rendered from:_ `metrics_schema.updated.json`
## Table of Contents
- [CSV Schemas](#csv-schemas)
- [Definitions](#definitions)
- [Methodology & Policies](#methodology--policies)
- [Epsilons](#epsilons)
- [Minimum Sample Policy](#minimum-sample-policy)
- [Monthly Returns Source](#monthly-returns-source)
- [Naming Conventions](#naming-conventions)
- [NaN/Inf Display & Policy](#naninf-display--policy)
- [No OOS Policy](#no-oos-policy)
- [Rebasing](#rebasing)
- [Risk Parametrization](#risk-parametrization)
- [Sharpe Risk-Free Policy](#sharpe-risk-free-policy)
- [Sortino Target Policy](#sortino-target-policy)
- [Supplementary Active-Only Metrics](#supplementary-active-only-metrics)
- [Variance](#variance)
- [Yearly Annualization Policy](#yearly-annualization-policy)
- [Yearly Metrics Policy](#yearly-metrics-policy)
- [Parameters](#parameters)
- [Profiles](#profiles)
- [Rounding Policy](#rounding-policy)
- [Runtime Configuration](#runtime-configuration)
- [Units](#units)
- [Metadata](#metadata)

## CSV Schemas


### Profile: compounding_eoy_soy_base_100k


#### File: dd_quantiles_full_period.csv

*Full-period drawdown depth quantiles computed on the EoM (month-end) underwater series dd_t = eq_t/cummax(eq_t) - 1, using only months with dd_t<0. Magnitudes are reported as negative decimals (dd_t). Durations are computed on CLOSED episodes only and reported in integer months.*
| name | type | label | units | desc |
|---|---|---|---|---|
| period_start_month | date | Period Start (Month, UTC) | YYYY-MM (UTC) | First month in dataset (YYYY-MM-01 UTC) used for the quantile computation. |
| period_end_month | date | Period End (Month, UTC) | YYYY-MM (UTC) | Last month in dataset (YYYY-MM-01 UTC) used for the quantile computation. |
| dd_observations_count | int | Drawdown Months — Count | count | Number of months with dd_t<0 included in the quantile computation. |
| dd_episodes_count | int | Drawdown Episodes — Count | count | Number of closed underwater episodes (from new high to recovery). |
| drawdown_p90_full_period | float | Drawdown P90 (EoM, Full Period) | decimal | 90th percentile of drawdown depth dd_t over dd_t<0; negative decimal. |
| drawdown_p95_full_period | float | Drawdown P95 (EoM, Full Period) | decimal | 95th percentile of drawdown depth dd_t over dd_t<0; negative decimal. |
| drawdown_p99_full_period | float | Drawdown P99 (EoM, Full Period) | decimal | 99th percentile of drawdown depth dd_t over dd_t<0; negative decimal. |
| underwater_duration_p90_full_period | int | Underwater Duration P90 (EoM, Months, Full Period) | months | 90th percentile of closed underwater episode durations, rounded half-up to months. |
| underwater_duration_p95_full_period | int | Underwater Duration P95 (EoM, Months, Full Period) | months | 95th percentile of closed underwater episode durations, rounded half-up to months. |
| drawdown_p90_full_period_xrisk | float | Drawdown P90 (EoM, Full Period, ×Risk) | R-multiple | 90th percentile of drawdown magnitudes in R-multiples: drawdown_p90_full_period / risk_per_trade_pct. |
| drawdown_p95_full_period_xrisk | float | Drawdown P95 (EoM, Full Period, ×Risk) | R-multiple | 95th percentile of drawdown magnitudes in R-multiples: drawdown_p95_full_period / risk_per_trade_pct. |
| drawdown_p99_full_period_xrisk | float | Drawdown P99 (EoM, Full Period, ×Risk) | R-multiple | 99th percentile of drawdown magnitudes in R-multiples: drawdown_p99_full_period / risk_per_trade_pct. |


#### File: full_period_summary.csv

| name | type | label | units | desc |
|---|---|---|---|---|
| months | int |  | months | Number of months in series (full period length). |
| cagr_full_period | float | CAGR (Full Period) | decimal | Full-period CAGR. |
| volatility_annualized_full_period | float | Volatility (Annualized, Full Period) | decimal | Annualized volatility from monthly returns. |
| sharpe_ratio_annualized_full_period | float | Sharpe (Annualized, Full Period) |  | Annualized Sharpe; rf=0 unless specified. |
| sortino_ratio_annualized_full_period | float | Sortino (Annualized, Full Period) |  | Sortino Ratio (annualized): (mean(r_m − t_m) / downside_std_m) * √12; default t_m=0. If an annual target t_ann is supplied (MAR or rf), use t_m=(1+t_ann)^(1/12)−1; undefined (NaN) if <2 negative months. |
| calmar_ratio_full_period | float | Calmar Ratio (EoM, Full Period) |  | cagr_full_period / |eom_max_drawdown_full_period| over the full period. |
| eom_max_drawdown_full_period | float | Max Drawdown (EoM, Full Period) | decimal (0.012 = 1.2%) | Maximum drawdown on end-of-month (EoM) monthly equity: min(dd_t) where dd_t = eq_t / cummax(eq_t) - 1 ≤ 0; reported as a negative decimal. |
| eom_longest_underwater_months | int | Longest Underwater (EoM, months) | months | Longest underwater spell on the EoM monthly equity measured from the last peak (dd=0) to the first recovery (dd>=0), inclusive of the recovery month; if no recovery occurs, measure to the last negative month. Reported as integer months. |
| eom_time_to_recover_months | int | Time To Recover MaxDD (EoM, months) | months | Time to recover for the MaxDD on the EoM monthly equity: number of months from the trough to the first dd>=0, inclusive of the recovery month; if never recovered by period end, report NaN. |
| eom_months_since_maxdd_trough | int | Months Since MaxDD Trough (EoM) | months | Months since the MaxDD trough on the EoM monthly equity: exclude the trough month and include the end month (last available month). |
| intramonth_max_drawdown_full_period | float | Max Drawdown (Intramonth, Full Period) | decimal (0.012 = 1.2%) | Maximum drawdown on the Intramonth (path-level) equity: min(dd_t) along the path (UTC ordering); reported as a negative decimal. |
| intramonth_longest_underwater_months | int | Longest Underwater (Intramonth, months) | months | Longest underwater spell on the Intramonth (path-level) equity measured from the last peak (dd=0) to the first recovery (dd>=0), inclusive of the recovery month; if no recovery, measure to the last negative observation. Duration in months = months_between(peak_ts, recovery_ts) + 1 when recovered; otherwise months_between(peak_ts, last_negative_ts). |
| intramonth_time_to_recover_months | int | Time To Recover MaxDD (Intramonth, months) | months | Time to recover for the MaxDD on the Intramonth (path-level) equity: months_between(trough_ts, recovery_ts) + 1; if never recovered by period end, report NaN. |
| intramonth_months_since_maxdd_trough | int | Months Since MaxDD Trough (Intramonth) | months | Months since the MaxDD trough on the Intramonth (path-level) equity: months_between(trough_ts, last_ts). The trough month is excluded; the end month is included. |
| underwater_months_share_full_period | float | Underwater Months — Share (Full Period) | decimal | Share of months with dd_t<0 over the full period = dd_observations_count / months. |
| ulcer_index_full_period | float | Ulcer Index (EoM, Full Period) | decimal | Root-mean-square of dd_t over months with dd_t<0 on the EoM monthly grid; reported as a positive decimal. |
| martin_ratio_full_period | float | Martin Ratio (EoM, Full Period) |  | CAGR (Full Period) / Ulcer Index (Full Period); NaN/Inf policy applies. |
| pain_index_full_period | float | Pain Index (EoM, Full Period) | decimal | Pain Index on EoM grid: mean(|dd_t|) over months with dd_t<0; positive decimal. |
| pain_ratio_full_period | float | Pain Ratio (EoM, Full Period) |  | Pain Ratio = CAGR (Full Period) / Pain Index (EoM, Full Period); NaN/Inf policy applies. |
| skewness_full_period | float | Skewness (Full Period) | decimal (0.012 = 1.2%) | Skewness of monthly returns over the full period (sample stdev with ddof=1 in standardization). |
| kurtosis_excess_full_period | float | Excess Kurtosis (Full Period) | decimal (0.012 = 1.2%) | Excess kurtosis (kurtosis − 3) of monthly returns over the full period (sample stdev, ddof=1). |
| eom_max_drawdown_full_period_xrisk | float | Max Drawdown (EoM, Full Period, ×Risk) | R-multiple | Maximum drawdown on the EoM monthly equity expressed in R-multiples: |eom_max_drawdown_full_period| / risk_per_trade_pct; positive magnitude. |
| intramonth_max_drawdown_full_period_xrisk | float | Max Drawdown (Intramonth, Full Period, ×Risk) | R-multiple | Maximum drawdown on the Intramonth (path-level) equity expressed in R-multiples: |intramonth_max_drawdown_full_period| / risk_per_trade_pct; positive magnitude. |
| ulcer_index_full_period_xrisk | float | Ulcer Index (EoM, Full Period, ×Risk) | R-multiple | Ulcer Index on the EoM grid expressed in R-multiples: ulcer_index_full_period / risk_per_trade_pct. |
| monthly_var_95_full_period_xrisk | float | Monthly VaR 95% (Full Period, ×Risk) | R-multiple | Monthly historical VaR 95% over the full period expressed in R-multiples: |monthly_var_95_full_period| / risk_per_trade_pct. |
| monthly_es_95_full_period_xrisk | float | Monthly ES 95% (Full Period, ×Risk) | R-multiple | Monthly historical ES 95% over the full period expressed in R-multiples: |monthly_es_95_full_period| / risk_per_trade_pct. |
| monthly_var_99_full_period_xrisk | float | Monthly VaR 99% (Full Period, ×Risk) | R-multiple | Monthly historical VaR 99% over the full period expressed in R-multiples: |monthly_var_99_full_period| / risk_per_trade_pct. |
| monthly_es_99_full_period_xrisk | float | Monthly ES 99% (Full Period, ×Risk) | R-multiple | Monthly historical ES 99% over the full period expressed in R-multiples: |monthly_es_99_full_period| / risk_per_trade_pct. |
| worst_month_return_full_period_xrisk | float | Worst Month (Full Period, ×Risk) | R-multiple | Worst single monthly return over the full period expressed in R-multiples: |worst_month_return_full_period| / risk_per_trade_pct. |
| best_month_return_full_period_xrisk | float | Best Month (Full Period, ×Risk) | R-multiple | Best single monthly return over the full period expressed in R-multiples: |best_month_return_full_period| / risk_per_trade_pct. |
| active_months_count | int | Active Months — Count | months | Number of active months over the full period. |
| active_months_share | float | Active Months — Share | decimal | Active months share = active_months_count / months. |
| volatility_active_annualized | float | Volatility (Annualized, Active Months — Supp.) | decimal (0.012 = 1.2%) | Annualized vol on active months (ddof=1)*sqrt(12). |
| sharpe_active_annualized | float | Sharpe (Annualized, Active Months — Supp.) |  | Sharpe on active months; sqrt(12). |
| sortino_active_annualized | float | Sortino (Annualized, Active Months — Supp.) |  | Sortino on active months; default target 0; sqrt(12). |
| cagr_active | float | CAGR (Active Months — Supp.) | decimal (0.012 = 1.2%) | Active-only CAGR: (Π(1+r_active))^(12/N_active)-1. |
| negative_months_count_full_period | int | Negative Months — Count (Full Period) | months | Number of months with monthly_return<0 over the full period. |
| total_return_full_period | float | Total Return (Full Period) | decimal (0.012 = 1.2%) | Π(1 + monthly_return) − 1 over the full compounding series. |
| wealth_multiple_full_period | float | Wealth Multiple (Full Period) |  | 1 + total_return_full_period. |
| ending_nav_full_period | float | Ending NAV (Full Period) | USD | NAV0 * Π(1 + monthly_return) with NAV0=100000. |
| period_start_month | date | Period Start (Month, UTC) | YYYY-MM (UTC) | First month in the dataset (YYYY-MM UTC). |
| period_end_month | date | Period End (Month, UTC) | YYYY-MM (UTC) | Last month in the dataset (YYYY-MM UTC). |
| positive_months_count_full_period | int | Positive Months — Count (Full Period) | months | Months with monthly_return>0. |
| best_month_return_full_period | float | Best Month Return (Full Period) | decimal (0.012 = 1.2%) |  |
| worst_month_return_full_period | float | Worst Month Return (Full Period) | decimal (0.012 = 1.2%) |  |
| mean_monthly_return_full_period | float | Mean Monthly Return (Full Period) | decimal (0.012 = 1.2%) |  |
| median_monthly_return_full_period | float | Median Monthly Return (Full Period) | decimal (0.012 = 1.2%) |  |
| zero_months_count_full_period | int | Zero Months — Count (Full Period) | months |  |
| years_covered | int | Years — Covered (Full Period) | years |  |
| best_year_return_calendar_full_period | float | Best Calendar-Year Return (Full Period) | decimal (0.012 = 1.2%) |  |
| best_year | int | Best Year (YYYY) | decimal (0.012 = 1.2%) |  |
| worst_year_return_calendar_full_period | float | Worst Calendar-Year Return (Full Period) | decimal (0.012 = 1.2%) |  |
| worst_year | int | Worst Year (YYYY) |  |  |
| downside_deviation_annualized_full_period | float | Downside Deviation (Annualized, Full Period) | decimal (0.012 = 1.2%) | Annualized downside deviation using monthly returns and target=0% annual converted to monthly; ddof=1; ×√12. |
| newey_west_tstat_mean_monthly_return | float | Newey–West t-stat (Mean Monthly Return) | decimal (0.012 = 1.2%) | HAC-robust t-statistic for mean monthly return using Newey–West standard error with Bartlett kernel and lag q (runtime parameter). |
| newey_west_p_value_mean_monthly_return | float | p-value (Newey–West, Two-sided) | decimal (0.012 = 1.2%) | Two-sided p-value for the HAC-robust t-statistic under asymptotic normal approximation. |
| max_consecutive_up_months_full_period | int | Max Consecutive Up Months (Full Period) | months | Longest run of months with monthly_return>0. Months with monthly_return==0 break runs (do not count). |
| max_consecutive_down_months_full_period | int | Max Consecutive Down Months (Full Period) | months | Longest run of months with monthly_return<0. Months with monthly_return==0 break runs (do not count). |
| omega_ratio_full_period_target_0m | float | Omega (τ=0, Full Period) |  | Omega ratio at monthly threshold τ=0 over the full period. |
| monthly_var_95_full_period | float | Monthly VaR 95% (Full Period) | decimal | Historical monthly VaR at 95% over the full period (5th percentile of r). Negative decimal return. |
| monthly_es_95_full_period | float | Monthly ES 95% (Full Period) | decimal | Historical monthly ES at 95% over the full period (mean of r ≤ VaR95). |
| monthly_var_99_full_period | float | Monthly VaR 99% (Full Period) | decimal | Historical monthly VaR at 99% over the full period (1st percentile of r). |
| monthly_es_99_full_period | float | Monthly ES 99% (Full Period) | decimal | Historical monthly ES at 99% over the full period (mean of r ≤ VaR99). |
| gain_to_pain_ratio_monthly_full_period | float | Gain-to-Pain (Monthly, Full Period) |  | Gain-to-Pain on monthly returns: sum(r_m>0) / |sum(r_m<0)| over the full period; zeros excluded from both sums. NaN/Inf policy per generic_ratio. |
| tail_ratio_p95_p5_full_period | float | Tail Ratio P95/|P5| (Full Period) |  | Tail Ratio = Q95(r) / |Q5(r)| over the full period. |
| trade_count_full_period | int |  | count | Total number of trades over the full period. |


#### File: monte_carlo_summary.csv

*Monte Carlo bootstrap results on the full-period monthly return series. Each row is one configuration (method, block_mean_length_months L, horizon). Monthly resampling (stationary_bootstrap by default). CAGR annualization via (1+ret)^(12/H)−1. MaxDD magnitudes computed on the EoM grid. Includes risk-normalized (×Risk) outputs for MaxDD percentiles, time-to-breach, complement probabilities, and conditional depths.*
| name | type | label | units | desc |
|---|---|---|---|---|
| period_start_month | date | Period Start (Month, UTC) | YYYY-MM (UTC) | First month used for sampling (full period start). |
| period_end_month | date | Period End (Month, UTC) | YYYY-MM (UTC) | Last month used for sampling (full period end; YTD inclusive). |
| method | string | Bootstrap Method |  | stationary_bootstrap (Politis–Romano) or moving_block_bootstrap (MBB). Default: stationary_bootstrap. |
| block_mean_length_months | int | Block Mean Length (Months) | months | Expected block length (L) in months; for stationary bootstrap L=1/p. Institutional default: 6. |
| n_paths | int | Paths |  | Number of bootstrap paths (e.g., 10000). |
| horizon_months | int | Horizon (Months) | months | Simulation horizon in months (e.g., 12, 36, full_period). In output, 'full_period' is resolved to the dataset length in months. |
| seed | int | Seed |  | Random seed for reproducibility (optional). |
| cagr_annualized_p05 | float | CAGR (Annualized) — P05 | decimal (0.012 = 1.2%) | 5th percentile of annualized CAGR across paths over the horizon. |
| cagr_annualized_p50 | float | CAGR (Annualized) — Median | decimal (0.012 = 1.2%) | 50th percentile (median) of annualized CAGR across paths over the horizon. |
| cagr_annualized_p95 | float | CAGR (Annualized) — P95 | decimal (0.012 = 1.2%) | 95th percentile of annualized CAGR across paths over the horizon. |
| max_drawdown_magnitude_p50 | float | Max Drawdown (EoM) — Magnitude P50 | decimal (0.012 = 1.2%) | 50th percentile of maximum drawdown magnitude (|dd|) on the EoM monthly grid across simulated paths; positive decimal. |
| max_drawdown_magnitude_p95 | float | Max Drawdown (EoM) — Magnitude P95 | decimal (0.012 = 1.2%) | 95th percentile of maximum drawdown magnitude (|dd|) computed on the EoM monthly grid across simulated paths; reported as a positive decimal. |
| max_drawdown_magnitude_p99 | float | Max Drawdown (EoM) — Magnitude P99 | decimal (0.012 = 1.2%) | 99th percentile of maximum drawdown magnitude (|dd|) computed on the EoM monthly grid across simulated paths; reported as a positive decimal. |
| prob_negative_horizon_return | float | Pr[Horizon Return < 0] | decimal | Probability (share of paths) that cumulative return over the horizon is < 0 (decimal). |
| wealth_multiple_p05 | float | Wealth Multiple — P05 |  | 5th percentile of wealth multiple over the horizon (Π(1+r)). |
| wealth_multiple_p50 | float | Wealth Multiple — Median |  | Median of wealth multiple over the horizon (Π(1+r)). |
| wealth_multiple_p95 | float | Wealth Multiple — P95 |  | 95th percentile of wealth multiple over the horizon (Π(1+r)). |
| ending_nav_p05 | float | Ending NAV — P05 (USD) | USD | Percentile of Ending NAV over the horizon given NAV0=100000 (USD). |
| ending_nav_p50 | float | Ending NAV — Median (USD) | USD | Percentile of Ending NAV over the horizon given NAV0=100000 (USD). |
| ending_nav_p95 | float | Ending NAV — P95 (USD) | USD | Percentile of Ending NAV over the horizon given NAV0=100000 (USD). |
| risk_per_trade_pct | float | Risk per Trade (Decimal) |  | Risk per trade used in the simulation (decimal, e.g., 0.01 or 0.015). |
| prob_maxdd_ge_5x_risk_eom | float | Pr[EoM MaxDD ≥ 5×Risk] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 5×risk_per_trade_pct over the horizon. |
| prob_maxdd_ge_7x_risk_eom | float | Pr[EoM MaxDD ≥ 7×Risk] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 7×risk_per_trade_pct over the horizon. |
| prob_maxdd_ge_10x_risk_eom | float | Pr[EoM MaxDD ≥ 10×Risk] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 10×risk_per_trade_pct over the horizon. |
| prob_maxdd_ge_5pc_eom | float | Pr[EoM MaxDD ≥ 5%] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 5% over the horizon. |
| prob_maxdd_ge_7pc_eom | float | Pr[EoM MaxDD ≥ 7%] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 7% over the horizon. |
| prob_maxdd_ge_10pc_eom | float | Pr[EoM MaxDD ≥ 10%] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 10% over the horizon. |
| mc_maxdd_xrisk_p50 | float | Max Drawdown (EoM) — ×Risk P50 | R-multiple | Median of EoM MaxDD magnitude across paths expressed in R-multiples: max_drawdown_magnitude_p50 / risk_per_trade_pct. |
| mc_maxdd_xrisk_p95 | float | Max Drawdown (EoM) — ×Risk P95 | R-multiple | 95th percentile of EoM MaxDD magnitude across paths expressed in R-multiples: max_drawdown_magnitude_p95 / risk_per_trade_pct. |
| mc_maxdd_xrisk_p99 | float | Max Drawdown (EoM) — ×Risk P99 | R-multiple | 99th percentile of EoM MaxDD magnitude across paths expressed in R-multiples: max_drawdown_magnitude_p99 / risk_per_trade_pct. |
| mc_ttb_ge_5x_risk_p50 | int | Time to Breach ≥5×Risk — Median (Months) |  | Median months to first breach of |dd| ≥ 5×risk among paths where breach occurs; end-anchored months. |
| mc_ttb_ge_7x_risk_p50 | int | Time to Breach ≥7×Risk — Median (Months) |  | Median months to first breach of |dd| ≥ 7×risk among paths where breach occurs; end-anchored months. |
| mc_ttb_ge_10x_risk_p50 | int | Time to Breach ≥10×Risk — Median (Months) |  | Median months to first breach of |dd| ≥ 10×risk among paths where breach occurs; end-anchored months. |
| prob_no_breach_ge_5x_risk_eom | float | Pr[No Breach ≥5×Risk] | decimal | Complement probability: 1 − Pr[EoM MaxDD ≥ 5×risk] over the horizon. |
| prob_no_breach_ge_7x_risk_eom | float | Pr[No Breach ≥7×Risk] | decimal | Complement probability: 1 − Pr[EoM MaxDD ≥ 7×risk] over the horizon. |
| prob_no_breach_ge_10x_risk_eom | float | Pr[No Breach ≥10×Risk] | decimal | Complement probability: 1 − Pr[EoM MaxDD ≥ 10×risk] over the horizon. |
| cond_es_maxdd_ge_5x_risk | float | Cond. Depth | Breach ≥5×Risk (×Risk) | R-multiple | Conditional expected MaxDD depth in R-multiples given breach ≥ 5×risk: E[ MaxDD_xRisk | breach ≥ 5×risk ]. |
| cond_es_maxdd_ge_7x_risk | float | Cond. Depth | Breach ≥7×Risk (×Risk) | R-multiple | Conditional expected MaxDD depth in R-multiples given breach ≥ 7×risk: E[ MaxDD_xRisk | breach ≥ 7×risk ]. |
| cond_es_maxdd_ge_10x_risk | float | Cond. Depth | Breach ≥10×Risk (×Risk) | R-multiple | Conditional expected MaxDD depth in R-multiples given breach ≥ 10×risk: E[ MaxDD_xRisk | breach ≥ 10×risk ]. |
| prob_maxdd_ge_20pc_eom | float | Pr[EoM MaxDD ≥ 20%] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 20% over the horizon. |
| prob_maxdd_ge_30pc_eom | float | Pr[EoM MaxDD ≥ 30%] | decimal | Share of simulated paths with EoM MaxDD magnitude ≥ 30% over the horizon. |


#### File: monthly_returns.csv

| name | type | label | units | desc |
|---|---|---|---|---|
| year | int |  | YYYY | Calendar year (YYYY). |
| month | int |  | 1–12 | 1–12. |
| monthly_return | float | Monthly Return | decimal | Monthly return from EoM NAV; includes months with 0. (Compounding/realized-PnL: include all months; if there were no trades and NAV did not change ⇒ 0; if there were trades but net monthly PnL is exactly 0 (NAV unchanged) ⇒ 0.) |
| active_month | bool | Active Month |  | True if month has at least one trade (trade_count>0). |


#### File: rolling_12m.csv

*Rolling 12-month window metrics (end-anchored by 'month'; UTC month anchors). Values are computed only when at least 12 months are available; otherwise set to NaN and flag insufficient_months.*
| name | type | label | units | desc |
|---|---|---|---|---|
| month | date | Month (Window End, UTC) | 1–12 | Month-anchored date YYYY-MM UTC for the end of the 12M window. |
| window_months | int | Window — Months | months | Size of the rolling window (12). |
| window_start_month | date | Window Start (UTC) |  | Start month of the 12M window (inclusive). |
| window_end_month | date | Window End (UTC) |  | End month of the 12M window (inclusive); equals 'month'. |
| rolling_return_12m | float | Rolling 12M Return | decimal | Product over last 12 months: Π(1+r_m) − 1. NaN if insufficient months. |
| rolling_volatility_annualized_12m | float | Volatility (Annualized, Rolling 12M) | decimal (0.012 = 1.2%) | Annualized volatility on last 12 months: stdev(r_m, ddof=1)*√12. NaN if insufficient months. |
| rolling_sharpe_annualized_12m | float | Sharpe (Annualized, Rolling 12M) |  | Annualized Sharpe on last 12 months: mean(r_m)/stdev(r_m, ddof=1)*√12 (rf=0 by default). NaN if insufficient months. |
| insufficient_months | bool | Insufficient Months |  | True if fewer than 12 months available as of the window end. |
| rolling_max_drawdown_12m | float | Rolling 12M Max Drawdown (EoM) | decimal (0.012 = 1.2%) | Maximum drawdown on the 12-month EoM sub-curve: min(dd_t); reported as a negative return. |
| rolling_calmar_12m | float | Rolling 12M Calmar (EoM) |  | Rolling 12M Calmar: rolling_return_12m (as annualized equals same for 12M) divided by |rolling_max_drawdown_12m|; NaN/Inf policy applies. |
| rolling_sortino_annualized_12m | float | Sortino (Annualized, Rolling 12M) |  | Annualized Sortino on last 12 months; downside via min(r_m − t_m, 0) with institutional target policy; ddof=1; ×√12. |
| active_months_count_12m | int | Active Months — Count (12M) | months | Number of months with active_month=True inside the 12M window. |
| active_months_share_12m | float | Active Months — Share (12M) | decimal | active_months_count_12m / 12; set to null when insufficient_months=true. |
| positive_months_count_12m | int | Positive Months — Count (12M) | months | Number of months with monthly_return>0 inside the 12M window. |
| negative_months_count_12m | int | Negative Months — Count (12M) | months | Number of months with monthly_return<0 inside the 12M window. |
| insufficient_negative_months | bool | Insufficient Negatives (12M) |  | True if fewer than 2 months with r_m below target within the 12M window (Sortino undefined). |
| positive_months_share_12m | float | Positive Months — Share (12M) | decimal | positive_months_count_12m / 12; set to null when insufficient_months=true. |
| negative_months_share_12m | float | Negative Months — Share (12M) | decimal | negative_months_count_12m / 12; set to null when insufficient_months=true. |
| rolling_omega_12m_target_0m | float | Rolling Omega (τ=0, 12M) |  | Omega(τ=0) on the last 12 months. |
| rolling_var_95_12m | float | Rolling Monthly VaR 95% (12M) | decimal | Historical monthly VaR95 within the 12M window. |
| rolling_es_95_12m | float | Rolling Monthly ES 95% (12M) | decimal | Historical monthly ES95 within the 12M window. |
| rolling_tail_ratio_p95_p5_12m | float | Rolling Tail Ratio P95/|P5| (12M) |  | Tail Ratio = Q95/|Q5| within the 12M window. |
| rolling_max_drawdown_12m_xrisk | float | Rolling 12M Max Drawdown (×Risk) | R-multiple | Rolling 12M maximum drawdown expressed in R-multiples: |rolling_max_drawdown_12m| / risk_per_trade_pct. |
| rolling_var_95_12m_xrisk | float | Rolling Monthly VaR 95% (12M, ×Risk) | R-multiple | Rolling monthly VaR 95% expressed in R-multiples: |rolling_var_95_12m| / risk_per_trade_pct. |
| rolling_es_95_12m_xrisk | float | Rolling Monthly ES 95% (12M, ×Risk) | R-multiple | Rolling monthly ES 95% expressed in R-multiples: |rolling_es_95_12m| / risk_per_trade_pct. |


#### File: rolling_36m.csv

*Rolling 36-month window metrics (end-anchored by 'month'; UTC month anchors). Values are computed only when at least 36 months are available; otherwise set to NaN and flag insufficient_months.*
| name | type | label | units | desc |
|---|---|---|---|---|
| month | date | Month (Window End, UTC) | 1–12 | Month-anchored date YYYY-MM UTC for the end of the 36M window. |
| window_months | int | Window — Months | months | Size of the rolling window (36). |
| window_start_month | date | Window Start (UTC) |  | Start month of the 36M window (inclusive). |
| window_end_month | date | Window End (UTC) |  | End month of the 36M window (inclusive); equals 'month'. |
| rolling_return_annualized_36m | float | Rolling 36M Return (Annualized) | decimal | Annualized return over last 36 months: (Π(1+r_m))^(12/N) − 1 with N=36. NaN if insufficient months. |
| rolling_max_drawdown_36m | float | Rolling 36M Max Drawdown (EoM) | decimal (0.012 = 1.2%) | Maximum drawdown on the 36-month EoM sub-curve: min(dd_t); reported as a negative return. |
| rolling_calmar_36m | float | Rolling 36M Calmar (EoM) |  | Rolling 36M Calmar: rolling_return_annualized_36m / |rolling_max_drawdown_36m| with NaN/Inf policy. |
| insufficient_months | bool | Insufficient Months |  | True if fewer than 36 months available as of the window end. |
| rolling_volatility_annualized_36m | float | Volatility (Annualized, Rolling 36M) | decimal (0.012 = 1.2%) | Annualized volatility on last 36 months: stdev(r_m, ddof=1)*√12; NaN if insufficient months. |
| rolling_sharpe_annualized_36m | float | Sharpe (Annualized, Rolling 36M) |  | Annualized Sharpe on last 36 months: mean(r_m)/stdev(r_m, ddof=1)*√12 (rf=0 by default). NaN if insufficient months. |
| rolling_sortino_annualized_36m | float | Sortino (Annualized, Rolling 36M) |  | Annualized Sortino on last 36 months; downside via min(r_m − t_m, 0) with institutional target policy; ddof=1; ×√12. |
| active_months_count_36m | int | Active Months — Count (36M) | months | Number of months with active_month=True inside the 36M window. |
| active_months_share_36m | float | Active Months — Share (36M) | decimal | active_months_count_36m / 36; set to null when insufficient_months=true. |
| positive_months_count_36m | int | Positive Months — Count (36M) | months | Number of months with monthly_return>0 inside the 36M window. |
| negative_months_count_36m | int | Negative Months — Count (36M) | months | Number of months with monthly_return<0 inside the 36M window. |
| insufficient_negative_months | bool | Insufficient Negatives (36M) |  | True if fewer than 2 months with r_m below target within the 36M window (Sortino undefined). |
| positive_months_share_36m | float | Positive Months — Share (36M) | decimal | positive_months_count_36m / 36; set to null when insufficient_months=true. |
| negative_months_share_36m | float | Negative Months — Share (36M) | decimal | negative_months_count_36m / 36; set to null when insufficient_months=true. |
| rolling_omega_36m_target_0m | float | Rolling Omega (τ=0, 36M) |  | Omega(τ=0) on the last 36 months. |
| rolling_var_95_36m | float | Rolling Monthly VaR 95% (36M) | decimal | Historical monthly VaR95 within the 36M window. |
| rolling_es_95_36m | float | Rolling Monthly ES 95% (36M) | decimal | Historical monthly ES95 within the 36M window. |
| rolling_tail_ratio_p95_p5_36m | float | Rolling Tail Ratio P95/|P5| (36M) |  | Tail Ratio = Q95/|Q5| within the 36M window. |
| rolling_max_drawdown_36m_xrisk | float | Rolling 36M Max Drawdown (×Risk) | R-multiple | Rolling 36M maximum drawdown expressed in R-multiples: |rolling_max_drawdown_36m| / risk_per_trade_pct. |
| rolling_var_95_36m_xrisk | float | Rolling Monthly VaR 95% (36M, ×Risk) | R-multiple | Rolling monthly VaR 95% expressed in R-multiples: |rolling_var_95_36m| / risk_per_trade_pct. |
| rolling_es_95_36m_xrisk | float | Rolling Monthly ES 95% (36M, ×Risk) | R-multiple | Rolling monthly ES 95% expressed in R-multiples: |rolling_es_95_36m| / risk_per_trade_pct. |


#### File: confidence_intervals.csv

*Confidence intervals for metrics computed on 'full_period' or 'calendar_year' scopes using bootstrap only (no Monte Carlo, no trade-bootstrap).*
| name | type | label | units | desc |
|---|---|---|---|---|
| period_start_month | date | Period Start (Month, UTC) | YYYY-MM (UTC) | First month of the data in scope for this CI row (YYYY-MM UTC). |
| period_end_month | date | Period End (Month, UTC) | YYYY-MM (UTC) | Last month of the data in scope for this CI row (YYYY-MM UTC). |
| scope | string | Scope |  | One of: 'full_period' | 'calendar_year'. |
| year | int | Year (if scope=calendar_year) | YYYY | Calendar year (YYYY) if scope='calendar_year'; else null. |
| metric_key | string | Metric Key |  | Key of the metric this CI pertains to (see definitions.ci_supported_metrics). |
| metric_label | string | Metric Label |  | Human-readable label of the metric at export time. |
| metric_basis | string | Basis |  | Computation basis: 'monthly' | 'eom' | 'intramonth'. |
| units | string | Units |  | Units consistent with the base metric (e.g., 'decimal', 'R-multiple', 'dimensionless ratio'). |
| rounding_family | string | Rounding Family |  | Formatting hint: 'percent_like' | 'xrisk_like' | 'sharpe_sortino_like' | 'ratio_like'. |
| method | string | CI Method |  | Method per row: 'bootstrap_percentile' | 'bootstrap_bca'. |
| ci_level_pct | float | CI Level (%) |  | Confidence level, e.g., 90.0. |
| estimate | float | Point Estimate | decimal (0.012 = 1.2%) | Point estimate in the metric’s units and sign conventions. |
| ci_low | float | CI Low |  | Lower bound of the CI in the metric’s units and sign conventions. |
| ci_high | float | CI High |  | Upper bound of the CI in the metric’s units and sign conventions. |
| bootstrap_type | string | Bootstrap Type |  | For metric CIs: e.g., 'stationary_bootstrap' or 'moving_block_bootstrap'. |
| block_mean_length_months | int | Block Mean Length (Months) | months | L for monthly/EoM bases. |
| block_mean_length_days | int | Block Mean Length (Days) |  | L for intramonth basis in calendar days. |
| n_boot | int | Bootstrap Replicates |  | Number of bootstrap replicates for metric CIs. |
| seed | int | Seed |  | Random seed for reproducibility (optional). |


#### File: trades_full_period_summary.csv

| name | type | label | units | desc |
|---|---|---|---|---|
| trade_count | int |  | count | Total trade_count (full period). |
| win_rate | float |  | decimal | Share of R>0 among trades with R≠0; denominator excludes zero-R trades. |
| profit_factor | float |  |  | PF on sums; zeros (R=0) are excluded from both sums. |
| average_winning_trade_r | float |  | R | mean(R | R > +eps). |
| average_losing_trade_r | float |  | R | mean(R | R < -eps). |
| payoff_ratio | float |  |  | average_winning_trade_r / |average_losing_trade_r|. |
| expectancy_mean_r | float |  | R | Mean R. |
| expectancy_median_r | float |  | R | Median R. |
| r_std_dev | float |  |  | Stdev R, ddof=1. |
| r_min | float |  |  | Min R. |
| r_max | float |  |  | Max R. |
| worst_5_trade_run_r | float |  | R-multiple | Worst k-trade run by R: minimum rolling sum of R over window k=5; REPORTED AS NEGATIVE (drawdown sign). If trade_count<5 → NaN. |
| worst_10_trade_run_r | float |  | R-multiple | Worst k-trade run by R over window k=10; REPORTED AS NEGATIVE (drawdown sign). If trade_count<10 → NaN. |
| worst_20_trade_run_r | float |  | R-multiple | Worst k-trade run by R over window k=20; REPORTED AS NEGATIVE (drawdown sign). If trade_count<20 → NaN. |
| edr_100_trades_p50_r | float | EDR100 — Median (×Risk) | R-multiple | EDR100 (Median): median MaxDD of cumulative R over a 100-trade horizon from stationary bootstrap on trades; REPORTED AS NEGATIVE. |
| edr_100_trades_p95_r | float | EDR100 — P95 (×Risk) | R-multiple | EDR100 (P95): 95th percentile (more conservative) of MaxDD of cumulative R over a 100-trade horizon; REPORTED AS NEGATIVE. |
| prob_maxdd_100trades_le_5r | float | Pr[MaxDD ≤ −5R] (100 trades) | decimal |  |
| prob_maxdd_100trades_le_7r | float | Pr[MaxDD ≤ −7R] (100 trades) | decimal |  |
| prob_maxdd_100trades_le_10r | float | Pr[MaxDD ≤ −10R] (100 trades) | decimal |  |
| losing_streak_max_p50_100trades | int | Max Losing Streak — P50 (Trades) | decimal (0.012 = 1.2%) | Median of the maximum consecutive losing-trade streak over a 100-trade horizon from stationary bootstrap on the R sequence. |
| losing_streak_max_p95_100trades | int | Max Losing Streak — P95 (Trades) | decimal (0.012 = 1.2%) | 95th percentile of the maximum consecutive losing-trade streak over a 100-trade horizon from stationary bootstrap on the R sequence. |
| prob_losing_streak_ge_7_100trades | float | Pr[Losing Streak ≥ 7] (100 trades) | decimal | Probability (share of paths) from stationary bootstrap on the trade-level R sequence. |
| prob_losing_streak_ge_10_100trades | float | Pr[Losing Streak ≥ 10] (100 trades) | decimal | Probability (share of paths) from stationary bootstrap on the trade-level R sequence. |
| holding_period_mean_minutes | float | Avg Holding Period (Minutes) | decimal (0.012 = 1.2%) | Average duration of closed trades in calendar minutes: mean((exit_ts - entry_ts)/60). Closed trades only; UTC. |
| holding_period_median_minutes | float | Median Holding Period (Minutes) | decimal (0.012 = 1.2%) | Median duration of closed trades (minutes, UTC). |
| holding_period_p95_minutes | float | Holding Period — P95 (Minutes) | decimal (0.012 = 1.2%) | 95th percentile of closed-trade durations (minutes, UTC). |
| holding_period_mean_minutes_wins | float | Avg Holding (Minutes, Wins) | decimal (0.012 = 1.2%) | Average duration of winning trades (R > +eps), in minutes (UTC). |
| holding_period_mean_minutes_losses | float | Avg Holding (Minutes, Losses) | decimal (0.012 = 1.2%) | Average duration of losing trades (R < -eps), in minutes (UTC). |


#### File: yearly_summary.csv

| name | type | label | units | desc |
|---|---|---|---|---|
| year | int |  | YYYY | Calendar year (YYYY). |
| annual_return_calendar | float | Return (Calendar Year / YTD) | decimal | Π(1+ret_m_year) - 1. |
| eom_max_drawdown_intra_year | float | Max Drawdown (EoM, Year) | decimal | Maximum drawdown on end-of-month (EoM) monthly equity within the calendar year (peak resets on Jan 1, UTC): min(dd_t); reported as a negative decimal. |
| intramonth_max_drawdown_intra_year | float | Max Drawdown (Intramonth, Year) | decimal | Maximum drawdown on the Intramonth (path-level) equity within the calendar year (peak resets on Jan 1, UTC): min(dd_t); reported as a negative decimal. |
| trade_count | int | Trades — Count | count | Trades in the year. |
| win_rate | float | Win Rate | decimal | Share of R>0 among trades with R≠0; denominator excludes zero-R trades. |
| profit_factor | float | Profit Factor |  | PF on sums within the year; zeros (R=0) are excluded from both sums. |
| active_months_count | int | Active Months — Count | months | Number of active months in the year (trade_count>0). |
| active_months_share | float | Active Months — Share | decimal | Active months share in the year = active_months_count/12. |
| annual_return_active | float | Return (Active Months — Supp.) | decimal | Return computed on active months only (supplementary). |
| volatility_active_annualized | float | Volatility (Annualized, Active Months — Supp.) | decimal (0.012 = 1.2%) | Annualized vol on active months (ddof=1)*sqrt(12). |
| sharpe_active_annualized | float | Sharpe (Annualized, Active Months — Supp.) |  | Sharpe on active months; rf_m if supplied; sqrt(12). |
| sortino_active_annualized | float | Sortino (Annualized, Active Months — Supp.) |  | Sortino on active months; default target 0; sqrt(12). |
| cagr_active | float | CAGR (Active Months — Supp.) | decimal (0.012 = 1.2%) | Active-only CAGR: (Π(1+r_active))^(12/N_active)-1. |
| insufficient_months | bool |  |  | True if months<12. |
| insufficient_active_months | bool |  |  | True if active_months_count < methodology.supplementary_active_only_metrics.min_active_months_for_metrics. |
| insufficient_negative_months | bool |  |  | True if negative months < 2 (affects Sortino). |
| insufficient_trades | bool |  |  | True if trade_count < methodology.minimum_sample_policy.min_trades_per_year_warning. |
| volatility_annualized_year | float | Volatility (Calendar Year, Annualized) | decimal (0.012 = 1.2%) | Annualized volatility within the calendar year (ddof=1)*sqrt(12). |
| sharpe_ratio_annualized_year | float | Sharpe (Calendar Year, Annualized) |  | Sharpe Ratio within the calendar year; rf_m if supplied; sqrt(12). |
| sortino_ratio_annualized_year | float | Sortino (Calendar Year, Annualized) |  | Sortino within the calendar year; default target 0; sqrt(12). |
| calmar_ratio_year | float | Calmar Ratio (EoM, Calendar Year) |  | annual_return_calendar / |eom_max_drawdown_intra_year|; Calmar-style ratio for the calendar year. |
| negative_months_in_year | int | Negative Months — Count (Calendar Year) | months | Number of months with monthly_return<0 in the calendar year. |
| months_in_year_available | int | Months in Year — Available | months | Number of monthly observations available in the calendar year (1..12). |
| is_ytd | bool | YTD (Partial Year) |  | True if months_in_year_available < 12 (partial year / YTD). |
| positive_months_in_year | int | Positive Months — Count (Calendar Year) | months | Number of months with monthly_return>0 in the calendar year. |
| omega_ratio_year_target_0m | float | Omega (τ=0, Year) |  | Omega(τ=0) within the calendar year. |
| monthly_var_95_year | float | Monthly VaR 95% (Year) | decimal | Historical monthly VaR95 computed on the year’s monthly returns. |
| monthly_es_95_year | float | Monthly ES 95% (Year) | decimal | Historical monthly ES95 computed on the year’s monthly returns. |


### Profile: fixed_start_100k


#### File: monthly_returns.csv

| name | type | label | units | desc |
|---|---|---|---|---|
| year | int |  | YYYY | Calendar year (YYYY). |
| month | int |  | 1–12 | 1–12. |
| monthly_return | float | Monthly Return | decimal | Monthly return inside the year; anchor 100k at year start.Months with no trades and no NAV change ⇒ 0; trade months with zero net monthly PnL (NAV unchanged) ⇒ 0. |
| active_month | bool | Active Month |  | True if month has at least one trade (trade_count>0). |


#### File: yearly_summary.csv

| name | type | label | units | desc |
|---|---|---|---|---|
| year | int |  | YYYY | Calendar year (YYYY). |
| annual_return_calendar | float | Return (Calendar Year / YTD) | decimal | Within-year Π(1+monthly_return) - 1. |
| eom_max_drawdown_intra_year | float | Max Drawdown (EoM, Year) | decimal | Maximum drawdown on end-of-month (EoM) monthly equity within the calendar year (peak resets on Jan 1, UTC): min(dd_t); reported as a negative decimal. |
| intramonth_max_drawdown_intra_year | float | Max Drawdown (Intramonth, Year) | decimal | Maximum drawdown on the Intramonth (path-level) equity within the calendar year (peak resets on Jan 1, UTC): min(dd_t); reported as a negative decimal. |
| trade_count | int | Trades — Count | count | Trades in the year. |
| win_rate | float | Win Rate | decimal | Share of R>0 among trades with R≠0; denominator excludes zero-R trades. |
| profit_factor | float | Profit Factor |  | PF on sums within the year; zeros (R=0) are excluded from both sums. |
| active_months_count | int | Active Months — Count | months | Number of active months in the year (trade_count>0). |
| active_months_share | float | Active Months — Share | decimal | Active months share in the year = active_months_count/12. |
| insufficient_months | bool |  |  | True if months<12. |
| insufficient_active_months | bool |  |  | True if active_months_count < methodology.supplementary_active_only_metrics.min_active_months_for_metrics. |
| insufficient_negative_months | bool |  |  | True if negative months < 2 (affects Sortino). |
| insufficient_trades | bool |  |  | True if trade_count < methodology.minimum_sample_policy.min_trades_per_year_warning. |
| negative_months_in_year | int | Negative Months — Count (Calendar Year) | months | Number of months with monthly_return<0 in the calendar year. |
| months_in_year_available | int | Months in Year — Available | months | Number of monthly observations available in the calendar year (1..12). |
| is_ytd | bool | YTD (Partial Year) |  | True if months_in_year_available < 12 (partial year / YTD). |
| positive_months_in_year | int | Positive Months — Count (Calendar Year) | months | Number of months with monthly_return>0 in the calendar year. |


## Definitions

- **active_month** — Boolean flag per month: True if at least one trade occurred in the calendar month (trade_count>0).
- **active_months_count** — Number of months with active_month=True within the aggregation scope (year or full period).
- **active_months_share** — Share of active months within the aggregation scope = active_months_count / total months in scope.
- **annual_return_active** — Annual Return (Active Months): product over active months only minus 1; supplementary.
- **annual_return_calendar** — Within-year return from monthly series: Π(1+ret_m_year) - 1.
- **average_losing_trade_r** — mean(R | R < -eps).
- **average_winning_trade_r** — mean(R | R > +eps).
- **best_month_return_full_period** — Best single monthly return over the full compounding period.
- **best_year** — Calendar year (YYYY) corresponding to best_year_return_calendar_full_period.
- **best_year_return_calendar_full_period** — Maximum calendar-year return observed over the full period (based on annual_return_calendar).
- **block_mean_length_months** — Expected block length L (months). In runtime this may be a single integer (e.g., 6) or a list for sensitivity (e.g., [3,6,9]). In CSV output each row carries a scalar L in 'block_mean_length_months'. For stationary bootstrap the restart probability is p=1/L.
- **cagr_active** — Active-only CAGR: (Π(1+r_m_active))^(12/N_active) - 1; N_active is number of active months.
- **cagr_annualized_p05** — 5th percentile of annualized CAGR across simulated paths over the given horizon.
- **cagr_annualized_p50** — 50th percentile (median) of annualized CAGR across simulated paths over the given horizon.
- **cagr_annualized_p95** — 95th percentile of annualized CAGR across simulated paths over the given horizon.
- **cagr_annualized_pXX** — Percentile of annualized CAGR across bootstrap paths: ((Π(1+r))^(12/H) − 1).
- **cagr_full_period** — (Π(1+monthly_return))^(12/N) - 1, where N is the number of months.
- **calmar_active** — Not defined. The Calmar ratio is reported only on the calendar series (Full Period and Calendar Year) using EoM drawdowns; do not compute for active months.
- **calmar_ratio_full_period** — Calmar Ratio (EoM, Full Period): cagr_full_period / |eom_max_drawdown_full_period| computed on the EoM monthly equity.
- **calmar_ratio_year** — Calmar Ratio (EoM, Calendar Year): annual_return_calendar / |eom_max_drawdown_intra_year|. If |eom_max_drawdown_intra_year| < eps_den: +Inf if annual_return_calendar > 0; −Inf if < 0; NaN if |annual_return_calendar| < eps_zero.
- **dd_episodes_count** — Count of CLOSED underwater episodes (from equity peak to recovery to new high) on the EoM monthly grid during the full period.
- **dd_observations_count** — Number of months with dd_t<0 on the EoM monthly grid used to compute drawdown depth quantiles over the full period.
- **downside_deviation_annualized_full_period** — Annualized downside deviation of monthly returns with target=0% annual converted to monthly; ddof=1; multiplied by sqrt(12).
- **drawdown_curve_definition** — Define monthly equity curve eq_t = Π_{k≤t}(1+r_k). Monthly drawdown dd_t = eq_t / cummax(eq_t) - 1 (≤0). Underwater periods are contiguous spans with dd_t<0; a period ends when a new high is made.
- **drawdown_p90_full_period** — 90th percentile of monthly drawdown depths dd_t for months with dd_t<0 on the EoM monthly grid; negative decimal.
- **drawdown_p95_full_period** — 95th percentile of monthly drawdown depths dd_t for months with dd_t<0 on the EoM monthly grid; negative decimal.
- **drawdown_p99_full_period** — 99th percentile of monthly drawdown depths dd_t for months with dd_t<0 on the EoM monthly grid; negative decimal.
- **ending_nav_full_period** — Ending NAV over the full period on the compounding track: NAV0 * product(1 + monthly_return). With NAV0=100000 (anchor).
- **ending_nav_p05** — 5th percentile of Ending NAV over the horizon given NAV0=100000 (USD).
- **ending_nav_p50** — Median of Ending NAV over the horizon given NAV0=100000 (USD).
- **ending_nav_p95** — 95th percentile of Ending NAV over the horizon given NAV0=100000 (USD).
- **expectancy_mean_r** — mean(R).
- **expectancy_median_r** — median(R).
- **gain_to_pain_ratio_monthly_full_period** — Sum of positive monthly returns divided by absolute sum of negative monthly returns; zeros excluded; apply generic_ratio.
- **horizon_months** — Simulation horizon in months. Runtime accepts integers (e.g., 12, 36) and the alias 'full_period' (resolved to N months in the dataset). In CSV output 'horizon_months' is always the resolved integer.
- **insufficient_active_months** — Flag: True if active months are below methodology.supplementary_active_only_metrics.min_active_months_for_metrics for the scope.
- **insufficient_months** — Flag: True if number of months in scope is below the minimum for stable annualization (see methodology.minimum_sample_policy).
- **insufficient_negative_months** — Flag: True if number of negative-return months in scope is < 2 (Sortino undefined).
- **insufficient_trades** — Flag: True if trade_count in the calendar year is below methodology.minimum_sample_policy.min_trades_per_year_warning.
- **ytd_flag_policy** — Set is_ytd=true only for the latest calendar year present in the dataset if months_in_year_available < 12; for all prior years is_ytd=false regardless of completeness. period_end_month is the last available month in that year. CI rows for that year use the same YTD range.
- **intramonth_equity_path** — Path-level equity built from all available equity observations sorted by timestamp in UTC (stable, input order on ties). No resampling to month-end.
- **intramonth_drawdown_curve_definition** — Along the Intramonth (path-level) equity, define dd_t = equity_t / cummax(equity_t) - 1 with cummax taken over time (UTC).
- **intramonth_max_drawdown_full_period** — Minimum dd_t on the Intramonth path over the full period. Reported as a negative decimal.
- **intramonth_max_drawdown_intra_year** — Minimum dd_t on the Intramonth path within the calendar year; the peak resets on January 1 (UTC). Reported as a negative decimal.
- **intramonth_longest_underwater_months** — Longest underwater spell on the Intramonth path measured from the last peak (dd = 0) to the first recovery (dd >= 0), inclusive of the recovery. Duration in months = months_between(peak_ts, recovery_ts) + 1; if no recovery, measure to the last negative observation without +1.
- **intramonth_time_to_recover_months** — Time to recover for the MaxDD on the Intramonth path: months_between(trough_ts, recovery_ts) + 1. If recovery never occurs by period end, report NaN.
- **intramonth_months_since_maxdd_trough** — Full months from the MaxDD trough to the last available timestamp on the Intramonth path: months_between(trough_ts, last_ts). The trough month is excluded; the end month is included.
- **months_between** — Calendar month difference used for Intramonth durations: months_between(a, b) = (year_b - year_a) * 12 + (month_b - month_a). No +1 at the boundary unless explicitly stated by the metric.
- **peak_anchor_policy** — Underwater spells start at the most recent peak (dd = 0) immediately preceding the first dd < 0 within the spell.
- **trough_tie_breaker_policy** — If the drawdown minimum occurs multiple times, select the earliest occurrence (first-in-time trough).
- **yearly_peak_reset_policy** — For all *_intra_year metrics (EoM and Intramonth), reset the peak at 00:00:00 UTC on January 1 of each calendar year.
- **kurtosis_excess_full_period** — Excess kurtosis (kurtosis minus 3) of monthly returns (ddof=1 for standardization).
- **martin_ratio_full_period** — Martin Ratio (EoM, Full Period): cagr_full_period / ulcer_index_full_period; follow NaN/Inf denominator policy.
- **max_consecutive_down_months_full_period** — Maximum number of consecutive months with monthly_return<0 in the full period; zeros break streaks.
- **max_consecutive_up_months_full_period** — Maximum number of consecutive months with monthly_return>0 in the full period; zeros break streaks.
- **eom_max_drawdown_full_period** — Maximum drawdown on end-of-month (EoM) monthly equity: min(dd_t) where dd_t = eq_t / cummax(eq_t) - 1 ≤ 0; reported as a negative decimal.
- **eom_max_drawdown_intra_year** — Maximum drawdown on end-of-month (EoM) monthly equity within the calendar year (peak resets on Jan 1, UTC): min(dd_t); reported as a negative decimal.
- **eom_time_to_recover_months** — Time to recover for the maximum drawdown on the end-of-month (EoM) monthly equity: number of months from the trough month to the first month with dd_t ≥ 0, inclusive of the recovery month; if recovery has not occurred by period_end_month, report NaN.
- **eom_longest_underwater_months** — Longest underwater spell on the EoM monthly equity measured from the most recent peak (dd=0) preceding the spell to the first recovery (dd_t ≥ 0), inclusive of the recovery month; if no recovery occurs, measure to the last month with dd_t < 0. Reported as integer months.
- **eom_months_since_maxdd_trough** — Months since the maximum drawdown trough on the EoM monthly equity: exclude the trough month and include the period end month (last available month). Reported as integer months.
- **max_drawdown_magnitude_p95** — 95th percentile of maximum drawdown magnitude on the EoM monthly grid across simulated paths; positive decimal.
- **max_drawdown_magnitude_p99** — 99th percentile of maximum drawdown magnitude on the EoM monthly grid across simulated paths; positive decimal.
- **max_drawdown_magnitude_pXX** — Percentile of maximum drawdown magnitude on the EoM monthly grid across simulated paths; reported as positive decimal.
- **mean_monthly_return_full_period** — Arithmetic mean of monthly returns over the full compounding period.
- **median_monthly_return_full_period** — Median of monthly returns over the full compounding period.
- **method** — Bootstrap method used to generate simulated monthly return paths (e.g., stationary_bootstrap or moving_block_bootstrap).
- **month** — Month key used in rolling CSVs as a month-anchored date (YYYY-MM-01 UTC). In monthly_returns.csv, the 'month' column is an integer 1–12 used together with 'year'.
- **monthly_return** — Monthly return from EoM NAV; decimal units; Months with no trade_count: monthly_return=0. If there are no trades and NAV is unchanged ⇒ monthly_return=0; if trades occurred but net monthly PnL is exactly 0 (NAV unchanged) ⇒ monthly_return=0.
- **months** — Number of monthly observations over the full period.
- **months_in_year_available** — Number of monthly observations available in the calendar year (1..12).
- **moving_block_bootstrap** — Moving block bootstrap: resample fixed-length overlapping blocks of returns (length L months) with replacement; preserves short-run dependence.
- **n_paths** — Number of simulated bootstrap paths.
- **negative_months_count_12m** — Count of months with r_m<0 in the last 12 months (rolling window).
- **negative_months_count_36m** — Count of months with r_m<0 in the last 36 months (rolling window).
- **negative_months_count_full_period** — Count of months with monthly_return < 0 over the full period.
- **negative_months_in_year** — Count of months with monthly_return < 0 within the calendar year.
- **negative_months_share_12m** — Share of negative months in the 12M window = negative_months_count_12m/12.
- **negative_months_share_36m** — Share of negative months in the 36M window = negative_months_count_36m/36.
- **newey_west_p_value_mean_monthly_return** — Two-sided p-value for the Newey–West t-statistic under asymptotic normal approximation.
- **newey_west_tstat_mean_monthly_return** — HAC-robust t-statistic for the mean monthly return using Newey–West standard error with Bartlett kernel and lag q.
- **payoff_ratio** — average_winning_trade_r / |average_losing_trade_r|.
- **period_end_month** — Last month in the dataset (month-anchored date, UTC).
- **period_start_month** — First month in the dataset (month-anchored date, UTC).
- **positive_months_count_12m** — Count of months with r_m>0 in the last 12 months (rolling window).
- **positive_months_count_36m** — Count of months with r_m>0 in the last 36 months (rolling window).
- **positive_months_count_full_period** — Count of months with monthly_return > 0 over the full period.
- **positive_months_in_year** — Count of months with monthly_return > 0 within the calendar year.
- **positive_months_share_12m** — Share of positive months in the 12M window = positive_months_count_12m/12.
- **positive_months_share_36m** — Share of positive months in the 36M window = positive_months_count_36m/36.
- **prob_negative_horizon_return** — Share of simulated paths with cumulative horizon return below 0 (decimal).
- **profit_factor** — PF = sum(R>+eps) / |sum(R<-eps)|; trades with |R| ≤ eps are excluded from both sums.
- **r** — R = pnl_pct / risk_per_trade_pct (risk_per_trade_pct from runtime/parameters).
- **r_max** — max(R).
- **r_min** — min(R).
- **r_std_dev** — stdev(R, ddof=1).
- **rolling_calmar_12m** — Rolling 12M Calmar: rolling_return_12m / |rolling_max_drawdown_12m|; NaN/Inf policy applies.
- **rolling_calmar_36m** — Rolling Calmar: rolling_return_annualized_36m / |rolling_max_drawdown_36m|; NaN/Inf policy applies.
- **rolling_max_drawdown_12m** — Maximum drawdown on the 12-month EoM sub-curve; reported as a negative return.
- **rolling_max_drawdown_36m** — Maximum drawdown on the 36-month EoM sub-curve; reported as a negative return.
- **rolling_return_12m** — Rolling cumulative return over the last 12 months: product(1+r_m) - 1.
- **rolling_return_annualized_36m** — Annualized return over the last 36 months: (product(1+r_m))^(12/N)-1, N=36.
- **rolling_sharpe_annualized_12m** — Annualized Sharpe using last 12 months (rf=0 by default): mean(r_m)/stdev(r_m, ddof=1)*sqrt(12).
- **rolling_sharpe_annualized_36m** — Annualized Sharpe on the last 36 months (rf=0 by default): mean(r_m)/stdev(r_m, ddof=1)*sqrt(12).
- **rolling_sortino_annualized_12m** — Annualized Sortino on the last 12 months: mean excess over target divided by downside stdev computed from min(r_m−t_m,0); ddof=1; ×sqrt(12).
- **rolling_sortino_annualized_36m** — Annualized Sortino on the last 36 months: downside via min(r_m−t_m,0); ddof=1; ×sqrt(12).
- **rolling_volatility_annualized_12m** — Annualized volatility using last 12 months: stdev(r_m, ddof=1)*sqrt(12).
- **rolling_volatility_annualized_36m** — Annualized volatility on the last 36 months: stdev(r_m, ddof=1)*sqrt(12).
- **seed** — Random seed used for reproducible simulation runs (optional).
- **sharpe_active_annualized** — Sharpe on active months only; (mean(r_m - rf_m)/stdev(r_m - rf_m, ddof=1))*sqrt(12).
- **sharpe_ratio_annualized_full_period** — (mean(r_m - rf_m) / stdev(r_m - rf_m, ddof=1)) * sqrt(12); rf=0 by default.
- **sharpe_ratio_annualized_year** — Sharpe Ratio (annualized) within a calendar year: (mean(r_m - rf_m) / stdev(r_m - rf_m, ddof=1)) * sqrt(12), where r_m are monthly returns of that year.
- **skewness_full_period** — Skewness of monthly returns (use sample standard deviation ddof=1 in standardization).
- **sortino_active_annualized** — Sortino on active months only; downside via min(r_m - t_m,0) with ddof=1; sqrt(12) annualization.
- **sortino_ratio_annualized_full_period** — Sortino Ratio (Annualized): Let monthly returns be r_m and target t_m. Numerator = mean(r_m - t_m). Denominator (downside deviation) = sqrt(sample_mean( min(r_m - t_m, 0)^2 )), ddof=1 on negatives. Annualization via sqrt(12): Sortino_ann = (mean(r_m - t_m) / downside_std_m) * sqrt(12). Default target t_m = 0. If an annual target t_ann is supplied (e.g., MAR or rf), use t_m = (1 + t_ann)^(1/12) - 1. Reporting policy: downside_std_m requires at least two negative monthly observations (ddof=1); if fewer than two negatives, Sortino is undefined and must be reported as NaN.
- **sortino_ratio_annualized_year** — Sortino Ratio (annualized) within a calendar year: target t_m per sortino_target_policy (default 0); downside uses min(r_m - t_m, 0) with ddof=1 on negatives; sqrt(12) annualization.
- **stationary_bootstrap** — Stationary bootstrap (Politis–Romano): resample returns by stitching blocks whose lengths are geometrically distributed with mean L months. Preserves serial dependence in expectation.
- **total_return_full_period** — Total return over the full period: product(1 + monthly_return) - 1, computed on the compounding monthly series.
- **trade_count** — Total number of executed trades within the aggregation scope of the file (calendar year or full period).
- **trade_count_full_period** — Total number of executed trades over the full period (all months).
- **ulcer_index_full_period** — Ulcer Index: sqrt(mean(dd_t^2)) computed over months with dd_t<0 on the EoM monthly grid; positive decimal.
- **underwater_duration_p90_full_period** — 90th percentile of closed underwater episode durations in months (rounded half-up).
- **underwater_duration_p95_full_period** — 95th percentile of closed underwater episode durations in months (rounded half-up).
- **volatility_active_annualized** — Annualized volatility computed only on active months; stdev(active r_m, ddof=1)*sqrt(12).
- **volatility_annualized_full_period** — stdev(monthly_return, ddof=1) * sqrt(12).
- **volatility_annualized_year** — Annualized volatility computed within a calendar year: stdev(monthly_return_in_year, ddof=1) * sqrt(12).
- **wealth_multiple_full_period** — Wealth multiple over the full compounding period: 1 + total_return_full_period.
- **wealth_multiple_p05** — 5th percentile of the wealth multiple over the horizon (Π(1+r)).
- **wealth_multiple_p50** — Median of the wealth multiple over the horizon (Π(1+r)).
- **wealth_multiple_p95** — 95th percentile of the wealth multiple over the horizon (Π(1+r)).
- **win_rate** — win_rate = count(R>+eps) / count(|R|>eps); trades with |R| ≤ eps are excluded.
- **window_end_month** — Last month included in the rolling window (inclusive; YYYY-MM-01 UTC).
- **window_months** — Size of the rolling window in months.
- **window_start_month** — First month included in the rolling window (inclusive; YYYY-MM-01 UTC).
- **worst_month_return_full_period** — Worst single monthly return over the full compounding period.
- **worst_year** — Calendar year (YYYY) corresponding to worst_year_return_calendar_full_period.
- **worst_year_return_calendar_full_period** — Minimum calendar-year return observed over the full period (based on annual_return_calendar).
- **year** — Calendar year (YYYY) for yearly rows (UTC-based).
- **years_covered** — Number of calendar years represented in the dataset (count of distinct years with at least one month).
- **zero_months_count_full_period** — Number of months with monthly_return == 0 over the full period.
- **omega_ratio** — For monthly returns r_m and a monthly threshold τ: Omega(τ) = mean(max(r_m − τ, 0)) / mean(max(τ − r_m, 0)). Shows the balance of gains vs losses around τ. Apply methodology.nan_inf_policy.generic_ratio to the denominator.
- **tail_ratio_p95_p5** — Tail Ratio = Q95(r) / |Q5(r)| on the monthly return distribution. >1 indicates a heavier upper tail in scale than the lower tail.
- **monthly_var_95** — Historical (empirical) Value-at-Risk at 95% on monthly returns: the 5th percentile of r; typically a negative decimal return.
- **monthly_es_95** — Historical (empirical) Expected Shortfall at 95% on monthly returns: the mean of r among observations r ≤ VaR95.
- **monthly_var_99** — Historical Value-at-Risk at 99% on monthly returns: the 1st percentile of r.
- **monthly_es_99** — Historical Expected Shortfall at 99% on monthly returns: the mean of r among observations r ≤ VaR99.
- **dd_thresholds_eom** — List of absolute EoM drawdown thresholds θ (decimals) used to compute probabilities Pr[MaxDD_eom ≥ θ]. Example: [0.05, 0.07, 0.10].
- **dd_thresholds_eom_normalized_multipliers** — Optional list of multipliers k for risk-normalized thresholds: θ = k × runtime.risk_per_trade_pct. Example: [5, 7, 10].
- **dd_thresholds_eom_mode** — Enum switch for Monte Carlo exceedance probabilities of EoM MaxDD: 'normalized' (θ = k × risk_per_trade_pct), 'absolute' (θ are fixed decimals), or 'both' (compute and publish both families).
- **risk_per_trade_pct** — Per-trade risk budget (decimal of equity) effective at runtime for the simulation/output row.
- **prob_maxdd_ge_5x_risk_eom** — Probability that EoM maximum drawdown magnitude over the horizon is ≥ 5×runtime.risk_per_trade_pct.
- **prob_maxdd_ge_7x_risk_eom** — Probability that EoM maximum drawdown magnitude over the horizon is ≥ 7×runtime.risk_per_trade_pct.
- **prob_maxdd_ge_10x_risk_eom** — Probability that EoM maximum drawdown magnitude over the horizon is ≥ 10×runtime.risk_per_trade_pct.
- **prob_maxdd_ge_5pc_eom** — Probability (share of bootstrap paths) that EoM maximum drawdown magnitude over the horizon is ≥ 5%.
- **prob_maxdd_ge_7pc_eom** — Probability (share of bootstrap paths) that EoM maximum drawdown magnitude over the horizon is ≥ 7%.
- **prob_maxdd_ge_10pc_eom** — Probability (share of bootstrap paths) that EoM maximum drawdown magnitude over the horizon is ≥ 10%.
- **xrisk_normalization** — Risk-normalized (×Risk) metric: value_xrisk = magnitude(value) / risk_per_trade_pct for losses/drawdowns/tails, reported as positive R-multiples unless explicitly signed. For Ulcer and quantiles, divide the (already positive) magnitude by risk_per_trade_pct.
- **mc_ttb_ge_kx_risk_p50** — Median number of months to the first EoM drawdown breach of |dd| ≥ k×risk among simulated paths where such a breach occurs; computed per horizon and block-length L.
- **prob_no_breach_ge_kx_risk_eom** — Complement probability to breach at threshold k×risk over the horizon: 1 − prob_maxdd_ge_kx_risk_eom.
- **cond_es_maxdd_ge_kx_risk** — Conditional expected depth of EoM MaxDD in R-multiples given that the breach threshold k×risk has been exceeded at least once within the horizon.
- **omega_ratio_full_period_target_0m** — Omega(τ=0) over the full compounding period: Omega(0) = mean(max(r_m − 0, 0)) / mean(max(0 − r_m, 0)) computed on monthly returns. Require at least one observation above and below τ; otherwise report NaN or ±Inf per methodology.nan_inf_policy.generic_ratio.
- **omega_ratio_year_target_0m** — Omega(τ=0) computed within the calendar year on that year's monthly returns. Require at least one observation above and below τ; otherwise apply methodology.nan_inf_policy.generic_ratio.
- **tail_ratio_p95_p5_full_period** — Tail Ratio over the full compounding period: Q95(r_m) / |Q5(r_m)| on monthly returns. >1 implies heavier (in scale) upper tail relative to the lower tail. Apply methodology.nan_inf_policy.generic_ratio when the denominator is near zero.
- **worst_k_trade_run_r** — Worst k-trade run by R: min over overlapping windows of Σ_{i=j..j+k-1} R_i. REPORTED AS NEGATIVE. R=0 are included. If trade_count<k → NaN.
- **worst_5_trade_run_r** — Worst k-trade run with k=5 per worst_k_trade_run_r; reported as negative.
- **worst_10_trade_run_r** — Worst k-trade run with k=10 per worst_k_trade_run_r; reported as negative.
- **worst_20_trade_run_r** — Worst k-trade run with k=20 per worst_k_trade_run_r; reported as negative.
- **edr_100_trades_p50_r** — EDR100 (Median): median MaxDD of cumulative R over 100 trades from a stationary bootstrap on the R sequence; reported as negative.
- **edr_100_trades_p95_r** — EDR100 (P95): 95th percentile of MaxDD of cumulative R over 100 trades from a stationary bootstrap; reported as negative.
- **prob_maxdd_100trades_le_7r** — Probability that trade-equity MaxDD over 100 trades is ≤ −7R (share of bootstrap paths; decimal).
- **prob_maxdd_100trades_le_10r** — Probability that trade-equity MaxDD over 100 trades is ≤ −10R (share of bootstrap paths; decimal).
- **losing_streak_max_p50_100trades** — Median (P50) of the maximum consecutive losing-trade streak over a 100-trade horizon (in trades).
- **losing_streak_max_p95_100trades** — P95 of the maximum consecutive losing-trade streak over a 100-trade horizon (in trades).
- **prob_losing_streak_ge_7_100trades** — Probability that the maximum losing streak over 100 trades is ≥ 7 (share of bootstrap paths; decimal).
- **prob_maxdd_100trades_le_5r** — Probability that trade-equity MaxDD over 100 trades is ≤ −5R (share of bootstrap paths; decimal).
- **ci_supported_metrics** — ['cagr_full_period', 'volatility_annualized_full_period', 'eom_max_drawdown_full_period', 'ulcer_index_full_period', 'monthly_var_95_full_period', 'monthly_es_95_full_period', 'monthly_var_99_full_period', 'monthly_es_99_full_period', 'intramonth_max_drawdown_full_period', 'eom_max_drawdown_intra_year', 'intramonth_max_drawdown_intra_year']
- **ci_level_pct** — Confidence level in percent (e.g., 90).
- **ci_low** — Lower bound of the confidence interval in the metric’s native units and sign conventions.
- **ci_high** — Upper bound of the confidence interval in the metric’s native units and sign conventions.
- **metric_basis** — Computation basis of the metric for CI: 'monthly' | 'eom' | 'intramonth'.
- **rounding_family** — Formatting group used to mirror the rounding of the underlying metric.
- **estimate** — Point estimate of the metric for the CI row, computed on the original sample (not the bootstrap mean). Reported in the same units and sign conventions as the base metric.
- **monthly_var_95_full_period** — Historical monthly VaR at 95% on monthly returns using quantile_method='linear' (HF type 7); reported as a negative decimal.
- **monthly_es_95_full_period** — Historical monthly ES at 95%: mean(r <= VaR95) with the same quantile method; negative decimal.
- **monthly_var_99_full_period** — Historical (empirical) monthly VaR at 99% on full-period monthly returns: the 1st percentile of r computed with quantile_method='linear'. Reported as a negative decimal monthly return (≤0).
- **monthly_es_99_full_period** — Historical (empirical) monthly ES at 99% on full-period monthly returns: the mean of r where r ≤ VaR99 (tail mean), using the same quantile_method='linear'. Reported as a negative decimal monthly return (≤0).
- **monthly_var_95_full_period_xrisk** — Risk-normalized magnitude: |monthly_var_95_full_period| / risk_per_trade_pct; reported as an R-multiple (positive).
- **monthly_es_95_full_period_xrisk** — Risk-normalized magnitude: |monthly_es_95_full_period| / risk_per_trade_pct; reported as an R-multiple (positive).
- **monthly_var_99_full_period_xrisk** — Risk-normalized magnitude: |monthly_var_99_full_period| / risk_per_trade_pct; reported as an R-multiple (positive).
- **monthly_es_99_full_period_xrisk** — Risk-normalized magnitude: |monthly_es_99_full_period| / risk_per_trade_pct; reported as an R-multiple (positive).
- **ci_method** — Confidence-interval method used in confidence_intervals.csv.method: 'bootstrap_percentile' | 'bootstrap_bca'.
- **mc_method** — Bootstrap method used in monte_carlo_summary.csv.method: 'stationary_bootstrap' | 'moving_block_bootstrap'.
- **r_multiple_units** — R-multiple (value divided by risk_per_trade_pct; dimensionless).
- **pain_index_full_period** — Mean magnitude of EoM drawdowns: mean(|dd_t|) over months with dd_t<0 on the EoM grid; positive decimal.
- **pain_ratio_full_period** — CAGR (Full Period) divided by pain_index_full_period. Apply methodology.nan_inf_policy.generic_ratio.
- **underwater_months_share_full_period** — Fraction of months with dd_t<0: dd_observations_count / months (decimal).
- **prob_maxdd_ge_20pc_eom** — Probability that EoM MaxDD over the horizon is ≥ 20%.
- **prob_maxdd_ge_30pc_eom** — Probability that EoM MaxDD over the horizon is ≥ 30%.
- **holding_period_mean_minutes** — Average duration of closed trades: mean((exit_ts - entry_ts)/60) in calendar minutes (UTC).
- **holding_period_median_minutes** — Median duration of closed trades in calendar minutes (UTC).
- **holding_period_p95_minutes** — 95th percentile of closed-trade durations in calendar minutes (UTC).
- **holding_period_mean_minutes_wins** — Average duration of winning trades (R > +eps) in minutes (UTC).
- **holding_period_mean_minutes_losses** — Average duration of losing trades (R < -eps) in minutes (UTC).

## Methodology & Policies


### Active Month Definition

Active month = a calendar month with at least one trade (trade_count>0). This is the only criterion; do not infer it from monthly_return.

### Drawdown Frequency Policy

EoM metrics are computed strictly on the month-end grid (UTC). Intramonth metrics are computed on the full equity path (UTC); durations are then converted to integer months via months_between. Max drawdown values and dd quantiles are reported as negative decimals; DD-based magnitudes (e.g. Ulcer Index) are reported as positive decimals; durations are integer months. Operationally, maximum drawdown is computed as min(dd_t) on the chosen grid/path; we report the dd value (negative), while DD magnitudes are reported as positive decimals.

### Calmar Policy

Calmar is NOT defined on the active-months subset. Do not compute or publish a Calmar ratio for active-only metrics. Rationale: Calmar pairs a return with a drawdown measured on the same continuous equity basis. The active-month subset is discontinuous and has no well-defined drawdown curve; mixing bases (e.g., CAGR_active with EoM MaxDD) is not permitted.

### Ytd Flag Policy

is_ytd = true only for the latest calendar year present if months_in_year_available < 12; false for prior years.

### Edge Cases

- **sortino_ratio_annualized**: Downside deviation (ddof=1) needs ≥2 negative months. If count_neg<2 → Sortino = NaN (undefined).

### Distribution Shape Policy

- **kurtosis_excess**: Compute excess kurtosis (kurtosis-3) using sample standard deviation (ddof=1).
- **skewness**: Compute skewness on monthly returns using sample standard deviation (ddof=1).

### Drawdown Curve

Dual basis for drawdowns: (1) EoM (monthly grid) — build eq_t = Π(1 + r_m) with dd_t = eq_t / cummax(eq_t) - 1; (2) Intramonth (path-level) — sort equity points by UTC time, compute dd_t = equity_t / cummax(equity_t) - 1 along the path; convert durations to months via months_between.

### Drawdown Quantiles Policy

- **duration**: Durations are computed for CLOSED underwater episodes only (from new high to recovery). Ongoing episode at period end is excluded from duration quantiles.
- **grid**: Monthly end-of-month equity; dd_t = eq_t/cummax(eq_t) - 1; use only months with dd_t<0.
- **magnitude**: drawdown_pNN_full_period are computed on dd_t and reported as negative decimals.
- **period**: Full available dataset (e.g., 2008–2025) unless otherwise specified in the file's start/end fields.
- **quantile_method**: linear

### Trade Duration Policy

- **basis**: Calendar minutes between entry_ts and exit_ts (UTC): duration_minutes = (exit_ts - entry_ts)/60.
- **scope**: Closed trades only; positions open at period end are excluded.
- **fills_aggregation**: Duration is computed at the position level: from the first opening fill to the final closing fill; intermediate adds/partials within one position do not create separate trades for duration.
- **same_bar_trades**: Trades opened and closed within the same 5-minute bar can yield small durations (including <5 minutes) based on actual execution timestamps.
- **bar_only_fallback**: If only bar timestamps are available (no execution timestamps), use an approximation: duration_minutes ≈ bars_held * bar_length_minutes (e.g., 5).
- **rounding**: Store as float minutes in CSV; presentation rounding is controlled by rounding_policy.

### Trades Bootstrap Policy

- **method**: Stationary bootstrap (Politis–Romano) on the trade-level R sequence.
- **block_mean_length_trades**: Expected block length L in trades; institutional default L=20 (sensitivity 10–30 OK).
- **paths**: 10,000 bootstrap paths by default.
- **horizons_trades**: List of trade horizons; for EDR use 100 trades.
- **maxdd_measure**: Maximum drawdown (most negative excursion) of cumulative R over the horizon.
- **outputs**: Publish quantiles p50 and p95 (optionally mean).
- **seed**: Reproducibility via runtime/parameters seed; otherwise system entropy.
- **notes**: Preserves short-run dependence through blocks; R=0 trades are kept.

### Xrisk Policy

- **definition**: Express risk-bearing quantities in R-multiples by dividing by runtime.risk_per_trade_pct.
- **magnitudes_positive**: drawdown_pNN_full_period_xrisk reported as negative decimals, other - positive
- **labels**: Append '×Risk' to labels to disambiguate from percent/decimal units.
- **applicability**: Applies to EoM and Intramonth drawdowns, Ulcer Index, monthly VaR/ES, rolling VaR/ES and drawdown, and Monte Carlo MaxDD-related outputs.

### Trade Run Policy

- **basis**: Chronological trade sequence (UTC); use raw R per trade, not monthly aggregation.
- **window**: Rolling sum over window size k; windows are overlapping; R=0 are included.
- **sign_convention**: All worst-run and EDR figures are reported with a negative sign (drawdown).
- **insufficient_sample**: If trade_count < k, return NaN.

### Omega Policy

- **definition**: Omega(τ) = mean(max(r_m − τ, 0)) / mean(max(τ − r_m, 0)) computed on monthly returns.
- **thresholds**: Default τ = 0 (monthly). Optionally reuse sortino_target_policy to set τ from an annual target (rf/MAR) via τ = (1 + t_ann)^(1/12) − 1.
- **sample_requirements**: Require at least one observation above and below τ; otherwise Omega is NaN or ±Inf per nan_inf_policy.generic_ratio.

### Var Es Policy

- **method**: Historical (empirical) VaR/ES on monthly returns; no parametric assumption.
- **levels**:
  - 95
  - 99
- **quantile_method**: linear
- **quantile_hf_type**: 7
- **es_tail_rule**: mean of observations r <= VaR_alpha using the same quantile method; no fractional weighting
- **notes**: For regulatory comparability, quantile_method='nearest_rank' (HF type 1). Use the same method consistently in bootstrap CIs.
- **reporting**: Report as decimal monthly returns (typically negative).
- **rolling**: For rolling windows, compute within-window; for calendar years, compute on that year's monthly returns.

## Epsilons

| Key | Value |
|---|---|
| eps_den | 1e-12 |
| eps_zero | 1e-12 |


## Minimum Sample Policy

| Key | Value |
|---|---|
| cagr_min_months | 1 |
| min_trades_per_year_warning | 12 |
| preferred_min_months_for_annualization | 12 |
| publish_under_min_months | Compute if formula-defined and return result; also emit a warning flag in metadata.sample_quality. |
| sharpe_ratio_annualized_min_months | 2 |
| sortino_ratio_annualized_min_negative_months | 2 |
| volatility_annualized_min_months | 2 |


## Naming Conventions


### Families


#### Active Only Supplementary

- **description**: Supplementary metrics computed on months with trade activity only; not a replacement for calendar metrics.
- **display_tag**: Active Months (Supp.)

#### Calendar Year From Monthlies

- **description**: Metrics computed strictly within each calendar year using that year's monthly returns; annualization via sqrt(12) where applicable.
- **display_tag**: Calendar Year

#### Full Period From Monthlies

- **description**: Metrics computed over the entire available series of monthly returns; annualization via sqrt(12) where applicable.
- **display_tag**: Full Period

### Note Negative Months

Negative months are reported per calendar year for both profiles, and additionally as a full-period audit metric for compounding.

### Ui Policy

Always append the family display_tag to metric labels to avoid ambiguity (e.g., 'Sharpe (Annualized, Full Period)' vs 'Sharpe (Calendar Year, Annualized)').

### Ui Ytd Label Policy

If is_ytd=True for a given year, append '— YTD' to metric labels in displays. E.g., 'Sharpe (Calendar Year, Annualized — YTD)'. Keys remain unchanged.

## NaN/Inf Display & Policy


### Display

| Token | Shown As |
|---|---|
| nan | NaN |
| neg_inf | -Inf |
| pos_inf | Inf |


### Policies

- **calmar_ratio_full_period**: If |eom_max_drawdown_full_period| < eps_den: +Inf if cagr_full_period > 0; −Inf if cagr_full_period < 0; NaN if |cagr_full_period| < eps_zero.
- **generic_ratio**: If |den|<eps_den: return +Inf if num>0; -Inf if num<0; NaN if |num|<eps_zero.
- **payoff_ratio**: payoff = avg_win / |avg_loss|. If count_losses=0 and count_wins>0 ⇒ +Inf. If count_wins=0 and count_losses>0 ⇒ 0. If both zero ⇒ NaN.
- **profit_factor**: PF = sum(R>0) / |sum(R<0)|. If |sum(R<0)|<eps_den and sum(R>0)>eps_zero ⇒ +Inf. If sum(R>0)<eps_zero and |sum(R<0)|≥eps_den ⇒ 0. If both sums <eps_zero ⇒ NaN.
- **sharpe_ratio_annualized**: If stdev_m<eps_den: +Inf if mean_excess>0; -Inf if mean_excess<0; NaN if |mean_excess|<eps_zero.
- **sortino_ratio_annualized**: Requires ≥2 negative months (downside sample, ddof=1). If <2 negatives: NaN. If downside_std_m<eps_den with ≥2 negatives: +Inf if mean_excess>0; -Inf if mean_excess<0; NaN if |mean_excess|<eps_zero.

## No OOS Policy

OOS/Walk-Forward are not applied: a single fixed EMM M5 logic is used with no re-optimization or curve fitting; the purpose of OOS is to detect overfitting after parameter tuning, which is not the case here. Multiple testing bias is absent: a single hypothesis/strategy was tested with no parameter sweep (selection bias is minimal). Profile robustness is supported by a long history (2003–2025-08), 12/36-month rolling windows, drawdown quantiles, BCa confidence intervals, and Monte Carlo (stationary bootstrap across all stated configurations). For audit, on request, a holdout with no retraining and no parameter changes can be prepared for independent verification.

## Rebasing


### Compounding

- **anchor**: BoM of the first observed month: YYYY-MM-01 00:00:00 UTC; NAV=100000
- **notes**: Compounding track: monthly_return = NAV_EoM(m) / NAV_EoM(m-1) - 1; for the first month use the BoM anchor NAV=100000. EoM NAV is ffilled to month-end if needed. (was: Anchor NAV=100000 at t0-ε (just before first observation).)

### Fixed

Anchor NAV=100000 at YYYY-01-01 00:00:00 UTC (anchor NAV=100000) for each calendar year.

## Risk Parametrization

R-metrics are scale-invariant given R = pnl_pct / risk_per_trade_pct. Switching risk_per_trade_pct between 1% and 1.5% changes NAV-based metrics (returns/vol/Sharpe/Sortino/MaxDD) but leaves R-based trade metrics comparable.

## Sharpe Risk-Free Policy

Sharpe computed with rf=0 unless an rf series is supplied (then use effective monthly rf: (1+rf_ann)^(1/12)-1).

## Sortino Target Policy

- **allowed_targets**:
  - risk_free_ann
  - mar_ann
- **conversion_to_monthly**: t_m = (1 + t_ann)^(1/12) − 1
- **default_target_ann**: 0
- **default_target_label**: zero_annual_return
- **preference**: If both risk_free_ann and mar_ann are provided, prefer mar_ann (conservative) unless a mandate specifies rf.

## Supplementary Active-Only Metrics

- **annualization**: Use N_active in exponents; e.g., CAGR_active = (Π(1+r_active))^(12/N_active) - 1.
- **applies_to**:
  - compounding_eoy_soy_base_100k
- **disclosure**: Active-only metrics are computed using months with active_month=True; not a replacement for calendar metrics.
- **does_not_apply_to**:
  - fixed_start_100k
- **min_active_months_for_metrics**: 6
- **rationale**: In fixed-start profiles the equity resets annually to 100k; cross-year aggregation is not meaningful. Institutions typically avoid active-only Sharpe/Sortino/CAGR for fixed-start; keep only activity counts.
- **sharpe_target**: rf_m if supplied, else 0.
- **sortino_target**: t_m from sortino_target_policy (default 0); downside uses min(r_m - t_m, 0) with ddof=1 on negatives.
- **status**: supplementary_only

## Variance

All standard deviations are sample (ddof=1).

## Yearly Annualization Policy

Yearly risk metrics (vol/sharpe/sortino) are computed on the available months within the calendar year and annualized via sqrt(12). If months_in_year_available<12, set insufficient_months=True. Sortino remains undefined (NaN) if negative months < 2.

## Yearly Metrics Policy

- **applies_to**:
  - compounding_eoy_soy_base_100k
- **metrics**:
  - volatility_annualized_year
  - sharpe_ratio_annualized_year
  - sortino_ratio_annualized_year
  - calmar_ratio_year
  - omega_ratio_year_target_0m
  - monthly_var_95_year
  - monthly_es_95_year
- **notes**: Computed strictly within calendar-year monthly returns; ddof=1; see methodology.nan_inf_policy.

## Parameters

| Parameter | Value |
|---|---|
| drawdown_basis | dual |


### risk_per_trade_pct

| Key | Value |
|---|---|
| allowed_values | [0.01, 0.015] |
| default | 0.01 |
| desc | Per-trade risk budget as a fraction of equity; switch between 0.01 (1%) and 0.015 (1.5%). |
| type | float |
| unit | fraction_of_equity |


## Profiles


### risk_1_5pct

| Key | Value |
|---|---|
| runtime.risk_per_trade_pct | 0.015 |


### risk_1pct

| Key | Value |
|---|---|
| runtime.risk_per_trade_pct | 0.01 |


## Rounding Policy


### xrisk_like

| Key | Value |
|---|---|
| applies_to_patterns | ['*xrisk*'] |
| decimals | 2 |


### duration_like_minutes

| Key | Value |
|---|---|
| applies_to_patterns | ['*holding_period*_minutes*'] |
| decimals | 0 |


### r_multiple_like_trades

| Key | Value |
|---|---|
| applies_to_patterns | ['worst_*_trade_run_r', 'edr_*_trades_*_r'] |
| decimals | 2 |


### wealth_like

| Key | Value |
|---|---|
| applies_to_patterns | ['wealth_multiple_*'] |
| decimals | 2 |


### currency_like

| Key | Value |
|---|---|
| applies_to | ['ending_nav_full_period', 'ending_nav_p05', 'ending_nav_p50', 'ending_nav_p95'] |
| decimals | 0 |
| rounding_mode | half_up |


### percent_like

| Key | Value |
|---|---|
| applies_to_patterns | ['*downside_deviation*', '*drawdown*', '*return*', '*share*', '*ulcer*', 'annual_return_*', 'prob_*', 'volatility_*', 'win_rate', '*var_*', '*_es_*', '*pain_index*'] |
| decimals | 4 |


### priority_order

- **priority_order**:
  - currency_like
  - wealth_like
  - sharpe_sortino_like
  - ratio_like
  - pvalue_like
  - r_multiple_like_trades
  - xrisk_like
  - duration_like_minutes
  - percent_like

### pvalue_like

| Key | Value |
|---|---|
| applies_to_patterns | ['*p_value*'] |
| decimals | 3 |


### ratio_like

| Key | Value |
|---|---|
| applies_to_patterns | ['*profit_factor*', '*payoff*', '*martin*', '*skew*', '*kurt*', '*omega*', '*tail_ratio*', '*gain_to_pain*', '*pain_ratio*'] |
| decimals | 2 |


### sharpe_sortino_like

| Key | Value |
|---|---|
| applies_to_patterns | ['*sharpe*', '*sortino*', '*calmar*'] |
| decimals | 2 |


## Runtime Configuration


### compute_active_only_metrics

| Key | Value |
|---|---|
| compounding_eoy_soy_base_100k | True |
| fixed_start_100k | False |


### confidence_intervals

| Key | Value |
|---|---|
| bootstrap | {'block_mean_length_months': [6], 'n_boot': 5000, 'seed': 43} |
| intramonth | {'block_mean_length_days': [5], 'n_boot': 5000, 'seed': 43} |


### trades_bootstrap

| Key | Value |
|---|---|
| block_mean_length_trades | 20 |
| horizons_trades | [100] |
| method | stationary_bootstrap |
| n_paths | 10000 |
| seed | 44 |


### monte_carlo

| Key | Value |
|---|---|
| block_mean_length_months | [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| horizons_months | [12, 36, 'full_period'] |
| method | stationary_bootstrap |
| n_paths | 10000 |
| seed | 42 |
| dd_thresholds_eom_mode | both |


### newey_west

| Key | Value |
|---|---|
| lag_months | 6 |


### risk_per_trade_pct

0.01

## Units

| Key | Value |
|---|---|
| currency | USD |
| returns | decimal (0.012 = +1.2%) |
| risk_per_trade_pct | decimal share of capital (e.g., 0.01 for 1%) |
| duration_minutes | calendar minutes (UTC) |
| x_risk | R-multiple (value divided by risk_per_trade_pct; dimensionless) |
| timezone | UTC (all timestamps, anchors, and buckets) |


### strict_R_classification

| Key | Value |
|---|---|
| eps | 1e-12 |
| win | R > +eps |
| loss | R < -eps |
| zero | |R| ≤ eps |
| note | Zero-R trades are those with |R| ≤ eps; zeros are excluded from PF denominator and from win_rate numerator/denominator; included as-is for expectancy/std/min/max. |


## Metadata


### risk_profiles

| Key | Value |
|---|---|
| active_runtime_value | 0.01 |
| available | ['risk_1pct', 'risk_1_5pct'] |
| default | risk_1pct |


### sample_quality

| Key | Value |
|---|---|
| warnings | ['Annualized metrics may be unstable if months < preferred_min_months_for_annualization.', 'Annual metrics flagged if months<12 (insufficient_months=true).', 'Active-only metrics flagged if active_months_count below threshold.', 'Sortino is undefined (NaN) if negative months < 2 (applies to full-period, yearly, and active-only variants)', 'Trade count warning if yearly trade_count < 12.'] |


### sortino_target_used

| Key | Value |
|---|---|
| annual | 0 |
| monthly | 0 |
| source | methodology.sortino_target_policy |
| type | default |



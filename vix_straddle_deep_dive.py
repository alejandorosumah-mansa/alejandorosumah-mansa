"""
VIX Straddle Deep Dive
========================
1. One-week (5-day) straddle profitability — full history
2. One-week straddle profitability — last year only
3. Straddle profitability when buying above VIX 25 — year-by-year breakdown
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def run_straddle_sim(spy_close, vix_close, common_dates, holding_trading_days, calendar_days):
    """Run straddle simulation with given holding period."""
    results = []
    for i in range(len(common_dates) - holding_trading_days):
        entry_date = common_dates[i]
        exit_date = common_dates[i + holding_trading_days]

        S = spy_close.iloc[i]
        K = S
        sigma = vix_close.iloc[i] / 100.0
        T = calendar_days / 365.0
        r = 0.02

        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        straddle_cost = call_price + put_price

        S_exit = spy_close.iloc[i + holding_trading_days]
        call_payoff = max(S_exit - K, 0)
        put_payoff = max(K - S_exit, 0)
        straddle_payoff = call_payoff + put_payoff

        pnl = straddle_payoff - straddle_cost
        pnl_pct = (pnl / straddle_cost) * 100

        results.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'spy_entry': S,
            'spy_exit': S_exit,
            'spy_move_pct': abs(S_exit - S) / S * 100,
            'vix_at_entry': vix_close.iloc[i],
            'straddle_cost': straddle_cost,
            'straddle_cost_pct': straddle_cost / S * 100,
            'straddle_payoff': straddle_payoff,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'profitable': pnl > 0,
            'year': entry_date.year,
        })
    return pd.DataFrame(results)


def print_summary(df, label):
    """Print straddle summary stats."""
    print(f"  Total trades:    {len(df):,}")
    print(f"  Win rate:        {df['profitable'].mean()*100:.1f}%")
    print(f"  Avg P&L ($):     ${df['pnl'].mean():.2f}")
    print(f"  Avg P&L (%):     {df['pnl_pct'].mean():.1f}%")
    print(f"  Median P&L (%):  {df['pnl_pct'].median():.1f}%")
    print(f"  Avg straddle cost: ${df['straddle_cost'].mean():.2f} ({df['straddle_cost_pct'].mean():.2f}% of SPY)")
    print(f"  Avg SPY move:    {df['spy_move_pct'].mean():.2f}%")
    print(f"  Breakeven move:  ~{df['straddle_cost_pct'].mean():.2f}%  (avg cost as % of SPY)")
    print(f"  Best trade:      {df['pnl_pct'].max():+.1f}% on {df.loc[df['pnl_pct'].idxmax(), 'entry_date'].strftime('%Y-%m-%d')}")
    print(f"  Worst trade:     {df['pnl_pct'].min():+.1f}% on {df.loc[df['pnl_pct'].idxmin(), 'entry_date'].strftime('%Y-%m-%d')}")
    print()


def print_yearly_table(df, label):
    """Print year-by-year breakdown."""
    print(f"  {'Year':>6}  {'N':>5}  {'Win%':>6}  {'Avg P&L%':>9}  {'Med P&L%':>9}  {'Avg $P&L':>9}  {'Avg VIX':>8}  {'Avg Move%':>10}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*10}")
    for year in sorted(df['year'].unique()):
        sub = df[df['year'] == year]
        if len(sub) < 3:
            continue
        print(f"  {year:>6}  {len(sub):>5}  {sub['profitable'].mean()*100:>5.1f}%  "
              f"{sub['pnl_pct'].mean():>+8.1f}%  {sub['pnl_pct'].median():>+8.1f}%  "
              f"${sub['pnl'].mean():>+8.2f}  {sub['vix_at_entry'].mean():>7.1f}  "
              f"{sub['spy_move_pct'].mean():>9.2f}%")
    print()


if __name__ == '__main__':
    # Fetch data
    print("Fetching data...")
    vix = yf.Ticker("^VIX")
    vix_df = vix.history(period="max")
    vix_df.index = vix_df.index.tz_localize(None)

    spy = yf.Ticker("SPY")
    spy_df = spy.history(period="max")
    spy_df.index = spy_df.index.tz_localize(None)

    common_dates = vix_df.index.intersection(spy_df.index)
    vix_close = vix_df.loc[common_dates, 'Close']
    spy_close = spy_df.loc[common_dates, 'Close']

    # ================================================================
    # ANALYSIS 1: One-Week Straddles — Full History
    # ================================================================
    print()
    print("=" * 80)
    print("ANALYSIS 1: ONE-WEEK (5-DAY) STRADDLE PROFITABILITY — FULL HISTORY")
    print("=" * 80)
    print("Buying ATM straddle on SPY, holding for 5 trading days (~7 calendar days)")
    print()

    weekly = run_straddle_sim(spy_close, vix_close, common_dates,
                              holding_trading_days=5, calendar_days=7)

    print("--- Overall ---")
    print_summary(weekly, "Weekly")

    # By VIX regime
    print("--- By VIX Regime at Entry ---")
    bins = [0, 12, 15, 20, 25, 30, 40, 100]
    labels_vix = ['<12', '12-15', '15-20', '20-25', '25-30', '30-40', '40+']
    weekly['vix_bin'] = pd.cut(weekly['vix_at_entry'], bins=bins, labels=labels_vix)
    for label in labels_vix:
        sub = weekly[weekly['vix_bin'] == label]
        if len(sub) > 10:
            print(f"  VIX {label:>5}: Win={sub['profitable'].mean()*100:5.1f}%  "
                  f"Avg P&L={sub['pnl_pct'].mean():+6.1f}%  "
                  f"Avg Move={sub['spy_move_pct'].mean():.2f}%  "
                  f"Avg Cost={sub['straddle_cost_pct'].mean():.2f}%  "
                  f"N={len(sub):,}")
    print()

    print("--- Year-by-Year Breakdown ---")
    print_yearly_table(weekly, "Weekly")

    # ================================================================
    # ANALYSIS 2: One-Week Straddles — Last Year Only
    # ================================================================
    print("=" * 80)
    print("ANALYSIS 2: ONE-WEEK STRADDLE — LAST 12 MONTHS")
    print("=" * 80)
    print()

    cutoff = pd.Timestamp('2025-03-01')
    last_year = weekly[weekly['entry_date'] >= cutoff].copy()

    if len(last_year) > 0:
        print(f"Period: {last_year['entry_date'].iloc[0].strftime('%Y-%m-%d')} to "
              f"{last_year['entry_date'].iloc[-1].strftime('%Y-%m-%d')}")
        print()
        print("--- Overall Last 12 Months ---")
        print_summary(last_year, "Last Year Weekly")

        # Monthly breakdown
        last_year['month'] = last_year['entry_date'].dt.to_period('M')
        print("--- Monthly Breakdown ---")
        print(f"  {'Month':>8}  {'N':>4}  {'Win%':>6}  {'Avg P&L%':>9}  {'Avg VIX':>8}  {'Avg Move%':>10}")
        print(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*10}")
        for month in sorted(last_year['month'].unique()):
            sub = last_year[last_year['month'] == month]
            print(f"  {str(month):>8}  {len(sub):>4}  {sub['profitable'].mean()*100:>5.1f}%  "
                  f"{sub['pnl_pct'].mean():>+8.1f}%  {sub['vix_at_entry'].mean():>7.1f}  "
                  f"{sub['spy_move_pct'].mean():>9.2f}%")
        print()

        # By VIX regime in last year
        last_year['vix_bin'] = pd.cut(last_year['vix_at_entry'], bins=bins, labels=labels_vix)
        print("--- By VIX at Entry (Last 12 Months) ---")
        for label in labels_vix:
            sub = last_year[last_year['vix_bin'] == label]
            if len(sub) > 0:
                print(f"  VIX {label:>5}: Win={sub['profitable'].mean()*100:5.1f}%  "
                      f"Avg P&L={sub['pnl_pct'].mean():+6.1f}%  N={len(sub)}")
        print()
    else:
        print("  No data available for the last 12 months.\n")

    # ================================================================
    # ANALYSIS 3: Buying Straddles When VIX > 25 — Year-by-Year
    # ================================================================
    print("=" * 80)
    print("ANALYSIS 3: STRADDLE PROFITABILITY WHEN BUYING WITH VIX > 25")
    print("=" * 80)
    print("Only entering straddles when VIX is above 25 at time of purchase")
    print()

    # Do this for BOTH weekly and monthly holding periods
    for period_name, hold_days, cal_days in [("1-WEEK (5 trading days)", 5, 7),
                                              ("1-MONTH (21 trading days)", 21, 30)]:
        print(f"--- {period_name} holding period ---")
        print()

        df = run_straddle_sim(spy_close, vix_close, common_dates,
                              holding_trading_days=hold_days, calendar_days=cal_days)
        high_vix = df[df['vix_at_entry'] > 25].copy()

        if len(high_vix) == 0:
            print("  No trades with VIX > 25.\n")
            continue

        print(f"  Total trades with VIX>25: {len(high_vix):,} out of {len(df):,} "
              f"({len(high_vix)/len(df)*100:.1f}%)")
        print()
        print("  Overall VIX>25 stats:")
        print_summary(high_vix, f"VIX>25 {period_name}")

        print(f"  Year-by-Year Breakdown (VIX > 25 entries only):")
        print_yearly_table(high_vix, f"VIX>25 {period_name}")

        # Compare: what if you ONLY buy when VIX > 25 vs ONLY when VIX < 25
        low_vix = df[df['vix_at_entry'] <= 25]
        print(f"  Comparison: VIX>25 vs VIX<=25")
        print(f"  {'':>15}  {'Win%':>6}  {'Avg P&L%':>9}  {'Med P&L%':>9}  {'Avg $P&L':>9}  {'N':>7}")
        print(f"  {'VIX > 25':>15}  {high_vix['profitable'].mean()*100:>5.1f}%  "
              f"{high_vix['pnl_pct'].mean():>+8.1f}%  {high_vix['pnl_pct'].median():>+8.1f}%  "
              f"${high_vix['pnl'].mean():>+8.2f}  {len(high_vix):>7,}")
        print(f"  {'VIX <= 25':>15}  {low_vix['profitable'].mean()*100:>5.1f}%  "
              f"{low_vix['pnl_pct'].mean():>+8.1f}%  {low_vix['pnl_pct'].median():>+8.1f}%  "
              f"${low_vix['pnl'].mean():>+8.2f}  {len(low_vix):>7,}")
        print()

        # Sub-regime within VIX>25
        print(f"  Sub-regimes within VIX>25:")
        sub_bins = [25, 30, 35, 40, 50, 100]
        sub_labels = ['25-30', '30-35', '35-40', '40-50', '50+']
        high_vix['sub_bin'] = pd.cut(high_vix['vix_at_entry'], bins=sub_bins, labels=sub_labels)
        for label in sub_labels:
            sub = high_vix[high_vix['sub_bin'] == label]
            if len(sub) > 5:
                print(f"    VIX {label:>5}: Win={sub['profitable'].mean()*100:5.1f}%  "
                      f"Avg P&L={sub['pnl_pct'].mean():+6.1f}%  "
                      f"Avg Move={sub['spy_move_pct'].mean():.2f}%  N={len(sub)}")
        print()

    # ================================================================
    # CHARTS
    # ================================================================
    print("Generating charts...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    # Chart 1: Weekly straddle P&L over time (rolling avg)
    ax = axes[0]
    weekly_roll = weekly.set_index('entry_date')['pnl_pct'].rolling(126).mean()
    ax.plot(weekly_roll.index, weekly_roll.values, linewidth=0.8, color='darkblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=weekly['pnl_pct'].mean(), color='green', linestyle='--', alpha=0.5,
               label=f'Overall avg: {weekly["pnl_pct"].mean():.1f}%')
    ax.set_title('Rolling 6-Month Avg: Weekly Straddle P&L (%) — Full History', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg P&L (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Year-by-year VIX>25 straddle win rate (weekly)
    ax = axes[1]
    weekly_high = weekly[weekly['vix_at_entry'] > 25].copy()
    yearly_stats = weekly_high.groupby('year').agg(
        win_rate=('profitable', 'mean'),
        avg_pnl=('pnl_pct', 'mean'),
        count=('pnl', 'count')
    )
    yearly_stats = yearly_stats[yearly_stats['count'] >= 3]
    colors = ['green' if x > 0 else 'red' for x in yearly_stats['avg_pnl']]
    bars = ax.bar(yearly_stats.index, yearly_stats['avg_pnl'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    for bar, (yr, row) in zip(bars, yearly_stats.iterrows()):
        y_pos = bar.get_height() + 1 if bar.get_height() >= 0 else bar.get_height() - 4
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{row["win_rate"]*100:.0f}%\nN:{int(row["count"])}',
                ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=6)
    ax.set_title('Weekly Straddle Avg P&L (%) by Year — VIX > 25 Entries Only', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg P&L (%)')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.3, axis='y')

    # Chart 3: VIX at entry vs straddle outcome scatter (weekly, sampled)
    ax = axes[2]
    sample = weekly.sample(min(3000, len(weekly)), random_state=42)
    winners = sample[sample['profitable']]
    losers = sample[~sample['profitable']]
    ax.scatter(losers['vix_at_entry'], losers['pnl_pct'], alpha=0.15, s=5, color='red', label='Loss')
    ax.scatter(winners['vix_at_entry'], winners['pnl_pct'], alpha=0.15, s=5, color='green', label='Win')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=25, color='blue', linestyle='--', alpha=0.7, label='VIX=25 threshold')
    ax.set_title('Weekly Straddle P&L vs VIX at Entry (scatter)', fontsize=13, fontweight='bold')
    ax.set_xlabel('VIX at Entry')
    ax.set_ylabel('Straddle P&L (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 300)

    plt.tight_layout()
    plt.savefig('/home/user/alejandorosumah-mansa/vix_straddle_deep_dive_charts.png',
                dpi=150, bbox_inches='tight')
    print("Charts saved to vix_straddle_deep_dive_charts.png")

    print("\n" + "=" * 80)
    print("DEEP DIVE COMPLETE")
    print("=" * 80)

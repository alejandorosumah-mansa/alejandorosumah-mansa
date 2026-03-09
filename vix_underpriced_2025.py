"""
Is VIX underpriced in 2025?
Core test: Compare implied vol (VIX) vs subsequent realized vol.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def realized_vol(returns, window=21):
    """Annualized realized volatility (21-day rolling)."""
    return returns.rolling(window).std() * np.sqrt(252) * 100


if __name__ == '__main__':
    vix = yf.Ticker("^VIX").history(period="max")
    vix.index = vix.index.tz_localize(None)
    spy = yf.Ticker("SPY").history(period="max")
    spy.index = spy.index.tz_localize(None)

    common = vix.index.intersection(spy.index)
    vix_close = vix.loc[common, 'Close']
    spy_close = spy.loc[common, 'Close']
    spy_ret = spy_close.pct_change()

    # Realized vol: 21-day trailing
    rv_21 = realized_vol(spy_ret, 21)
    # Forward realized vol: what actually happened in the next 21 days
    fwd_rv = spy_ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100

    df = pd.DataFrame({
        'vix': vix_close,
        'rv_trailing': rv_21,
        'rv_forward': fwd_rv,
        'spy': spy_close,
    }).dropna()

    df['vrp'] = df['vix'] - df['rv_forward']  # Variance risk premium: VIX - realized
    df['year'] = df.index.year

    # ================================================================
    # 1. VIX vs Realized Vol — 2025 vs History
    # ================================================================
    print("=" * 80)
    print("IS VIX UNDERPRICED IN 2025?")
    print("=" * 80)
    print()
    print("Variance Risk Premium (VRP) = VIX - Forward 21-day Realized Vol")
    print("Positive VRP = VIX overpriced (normal). Negative VRP = VIX UNDERPRICED.")
    print()

    print(f"{'Year':>6}  {'Avg VIX':>8}  {'Avg Fwd RV':>10}  {'Avg VRP':>8}  {'VRP<0 %':>8}  {'Avg SPY Move':>12}")
    print(f"{'─'*6}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*12}")

    for year in sorted(df['year'].unique()):
        sub = df[df['year'] == year]
        if len(sub) < 20:
            continue
        vrp_neg_pct = (sub['vrp'] < 0).mean() * 100
        daily_move = spy_ret.loc[sub.index].abs().mean() * 100
        print(f"  {year:>4}  {sub['vix'].mean():>7.1f}  {sub['rv_forward'].mean():>9.1f}  "
              f"{sub['vrp'].mean():>+7.1f}  {vrp_neg_pct:>6.1f}%  {daily_move:>10.2f}%")

    print()

    # ================================================================
    # 2. Deep dive on 2025
    # ================================================================
    d25 = df[df['year'] == 2025].copy()
    d24 = df[df['year'] == 2024].copy()
    hist = df[df['year'] < 2025].copy()

    print("=" * 80)
    print("2025 DEEP DIVE")
    print("=" * 80)
    print()

    print("--- 2025 vs 2024 vs All History ---")
    for label, subset in [("All History", hist), ("2024", d24), ("2025", d25)]:
        print(f"  {label:>12}: Avg VIX={subset['vix'].mean():.1f}  "
              f"Avg Fwd RV={subset['rv_forward'].mean():.1f}  "
              f"VRP={subset['vrp'].mean():+.1f}  "
              f"VRP<0: {(subset['vrp']<0).mean()*100:.1f}%")
    print()

    # Monthly breakdown for 2025
    print("--- 2025 Monthly Breakdown ---")
    d25['month'] = d25.index.month
    d25['month_name'] = d25.index.strftime('%b')
    print(f"  {'Month':>5}  {'Avg VIX':>8}  {'Avg Fwd RV':>10}  {'VRP':>6}  {'VRP<0%':>7}  {'Verdict':>15}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*15}")
    for m in sorted(d25['month'].unique()):
        sub = d25[d25['month'] == m]
        if len(sub) < 5:
            continue
        vrp = sub['vrp'].mean()
        neg_pct = (sub['vrp'] < 0).mean() * 100
        verdict = "UNDERPRICED" if vrp < -2 else "FAIR" if abs(vrp) <= 2 else "OVERPRICED"
        mname = sub['month_name'].iloc[0]
        print(f"  {mname:>5}  {sub['vix'].mean():>7.1f}  {sub['rv_forward'].mean():>9.1f}  "
              f"{vrp:>+5.1f}  {neg_pct:>5.1f}%  {verdict:>15}")
    print()

    # ================================================================
    # 3. Tail risk: big move days
    # ================================================================
    print("=" * 80)
    print("TAIL RISK ANALYSIS: BIG MOVE DAYS")
    print("=" * 80)
    print()

    spy_ret_abs = spy_ret.abs() * 100
    for threshold in [1.0, 1.5, 2.0, 3.0]:
        print(f"--- Days with |SPY move| > {threshold}% ---")
        for label, yr in [("All History", None), ("2024", 2024), ("2025", 2025)]:
            if yr:
                mask = (spy_ret_abs.index.year == yr) & (spy_ret_abs > threshold)
                total = (spy_ret_abs.index.year == yr).sum()
            else:
                mask = (spy_ret_abs.index.year < 2025) & (spy_ret_abs > threshold)
                total = (spy_ret_abs.index.year < 2025).sum()
            count = mask.sum()
            pct = count / total * 100 if total > 0 else 0
            print(f"  {label:>12}: {count:>4} days ({pct:.1f}% of trading days)")
        print()

    # ================================================================
    # 4. VIX level vs what followed — 2025 episodes
    # ================================================================
    print("=" * 80)
    print("2025 KEY EPISODES: VIX vs WHAT ACTUALLY HAPPENED")
    print("=" * 80)
    print()

    # Find periods where VIX was notably low relative to what followed
    d25_full = df[df['year'] >= 2025].copy()
    d25_full['underpriced'] = d25_full['vrp'] < -5  # significantly underpriced
    underpriced_streaks = d25_full[d25_full['underpriced']]

    if len(underpriced_streaks) > 0:
        print("Days where VIX was >5 points below subsequent realized vol:")
        print(f"  {'Date':>12}  {'VIX':>6}  {'Fwd RV':>7}  {'VRP':>6}  {'SPY':>8}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}")
        for date, row in underpriced_streaks.iterrows():
            print(f"  {date.strftime('%Y-%m-%d'):>12}  {row['vix']:>5.1f}  {row['rv_forward']:>6.1f}  "
                  f"{row['vrp']:>+5.1f}  {row['spy']:>.0f}")
        print(f"\n  Total: {len(underpriced_streaks)} days where VIX was significantly underpriced")
    else:
        print("  No days found where VIX was >5 pts below realized vol.")
    print()

    # ================================================================
    # 5. Regime comparison: policy-driven vs organic vol
    # ================================================================
    print("=" * 80)
    print("STRUCTURAL ARGUMENT: WHY VIX MIGHT BE SYSTEMATICALLY UNDERPRICED IN 2025")
    print("=" * 80)
    print()
    print("Key factors to consider:")
    print()
    print("1. POLICY-DRIVEN VOLATILITY (tariffs, trade war)")
    print("   - VIX measures implied vol from options market makers")
    print("   - Market makers price vol based on recent realized vol + skew")
    print("   - But tariff announcements create JUMP RISK that continuous vol models miss")
    print("   - VIX can't easily price binary policy events (tariff on/off)")
    print()
    print("2. VIX MEAN-REVERSION BIAS")
    print("   - VIX structurally reverts to ~15-20")
    print("   - After spikes, options desks rapidly sell vol expecting normalization")
    print("   - But if the vol SOURCE (trade policy) is persistent, this selling is premature")
    print()
    print("3. OVERNIGHT / GAP RISK")
    print("   - Many 2025 moves happened on overnight gaps (tweet-driven policy)")
    print("   - VIX options trade during market hours only")
    print("   - Gap risk is not fully captured in VIX pricing")
    print()

    # Quantify: overnight vs intraday vol in 2025
    spy_full = yf.Ticker("SPY").history(period="2y")
    spy_full.index = spy_full.index.tz_localize(None)
    spy_2025 = spy_full[spy_full.index.year >= 2025].copy()
    spy_2024 = spy_full[spy_full.index.year == 2024].copy()

    for label, sdf in [("2024", spy_2024), ("2025", spy_2025)]:
        overnight = (sdf['Open'] / sdf['Close'].shift(1) - 1).dropna()
        intraday = (sdf['Close'] / sdf['Open'] - 1).dropna()
        total = (sdf['Close'] / sdf['Close'].shift(1) - 1).dropna()

        overnight_vol = overnight.std() * np.sqrt(252) * 100
        intraday_vol = intraday.std() * np.sqrt(252) * 100
        total_vol = total.std() * np.sqrt(252) * 100
        overnight_share = overnight.var() / total.var() * 100 if total.var() > 0 else 0

        print(f"  {label}: Total vol={total_vol:.1f}%  Overnight vol={overnight_vol:.1f}%  "
              f"Intraday vol={intraday_vol:.1f}%  Overnight share={overnight_share:.0f}%")

    print()
    print("  If overnight share is elevated in 2025, it suggests jump/gap risk that")
    print("  VIX (which is priced from intraday options) may be systematically missing.")
    print()

    # ================================================================
    # CHARTS
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    # Chart 1: VIX vs Realized Vol — 2024-2025
    ax = axes[0]
    recent = df[df.index >= '2024-01-01']
    ax.plot(recent.index, recent['vix'], label='VIX (Implied)', color='navy', linewidth=1.2)
    ax.plot(recent.index, recent['rv_forward'], label='Forward 21d Realized Vol', color='red', linewidth=1.0, alpha=0.8)
    ax.fill_between(recent.index, recent['vix'], recent['rv_forward'],
                     where=recent['vix'] < recent['rv_forward'],
                     alpha=0.3, color='red', label='VIX Underpriced')
    ax.fill_between(recent.index, recent['vix'], recent['rv_forward'],
                     where=recent['vix'] >= recent['rv_forward'],
                     alpha=0.2, color='green', label='VIX Overpriced')
    ax.axvline(x=pd.Timestamp('2025-01-01'), color='gray', linestyle=':', alpha=0.5)
    ax.set_title('VIX vs Forward Realized Vol (2024-2025)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Volatility (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Chart 2: VRP over time (full history, highlight 2025)
    ax = axes[1]
    yearly_vrp = df.groupby('year')['vrp'].mean()
    colors = ['red' if v < 0 else 'green' for v in yearly_vrp.values]
    bars = ax.bar(yearly_vrp.index, yearly_vrp.values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=yearly_vrp.mean(), color='blue', linestyle='--', alpha=0.5,
               label=f'Historical avg: {yearly_vrp.mean():.1f}')
    ax.set_title('Average Variance Risk Premium by Year (VIX - Realized Vol)', fontsize=14, fontweight='bold')
    ax.set_ylabel('VRP (pts)')
    ax.set_xlabel('Year')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Chart 3: 2025 monthly VIX vs Realized
    ax = axes[2]
    if len(d25) > 0:
        monthly = d25.groupby('month').agg(
            avg_vix=('vix', 'mean'),
            avg_rv=('rv_forward', 'mean'),
        )
        x = np.arange(len(monthly))
        w = 0.35
        ax.bar(x - w/2, monthly['avg_vix'], w, label='Avg VIX (Implied)', color='navy', alpha=0.7)
        ax.bar(x + w/2, monthly['avg_rv'], w, label='Avg Fwd Realized Vol', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]
                            for m in monthly.index])
        ax.set_title('2025 Monthly: VIX vs Forward Realized Vol', fontsize=14, fontweight='bold')
        ax.set_ylabel('Volatility (%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/alejandorosumah-mansa/vix_underpriced_2025_charts.png', dpi=150, bbox_inches='tight')
    print("Charts saved to vix_underpriced_2025_charts.png")
    print("\nDONE")

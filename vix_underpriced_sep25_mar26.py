"""
VIX underpricing analysis: Sep 2025 - Mar 2026 window
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

    rv_21 = realized_vol(spy_ret, 21)
    fwd_rv = spy_ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100

    df = pd.DataFrame({
        'vix': vix_close,
        'rv_trailing': rv_21,
        'rv_forward': fwd_rv,
        'spy': spy_close,
    }).dropna()

    df['vrp'] = df['vix'] - df['rv_forward']
    df['year'] = df.index.year

    # Focus window: Sep 2025 - Mar 2026
    window = df[(df.index >= '2025-09-01') & (df.index <= '2026-03-31')].copy()
    # Comparison windows
    h1_2025 = df[(df.index >= '2025-01-01') & (df.index < '2025-07-01')].copy()
    h2_2025 = df[(df.index >= '2025-07-01') & (df.index < '2026-01-01')].copy()
    q1_2026 = df[(df.index >= '2026-01-01') & (df.index <= '2026-03-31')].copy()
    full_hist = df[df.index < '2025-01-01']

    print("=" * 85)
    print("VIX UNDERPRICING: SEP 2025 - MAR 2026 FOCUS")
    print("=" * 85)
    print()

    # ================================================================
    # 1. Monthly breakdown
    # ================================================================
    print("--- Monthly Breakdown: Sep 2025 - Mar 2026 ---")
    print()
    window['month_label'] = window.index.strftime('%Y-%b')
    window['ym'] = window.index.to_period('M')

    print(f"  {'Month':>8}  {'Avg VIX':>8}  {'Avg Fwd RV':>10}  {'VRP':>6}  {'VRP<0%':>7}  "
          f"{'Min VIX':>8}  {'Max RV':>8}  {'Verdict':>12}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*12}")

    for ym in sorted(window['ym'].unique()):
        sub = window[window['ym'] == ym]
        if len(sub) < 5:
            continue
        vrp = sub['vrp'].mean()
        neg_pct = (sub['vrp'] < 0).mean() * 100
        verdict = "UNDERPRICED" if vrp < -2 else "FAIR" if abs(vrp) <= 2 else "OVERPRICED"
        label = sub.index[0].strftime('%Y-%b')
        print(f"  {label:>8}  {sub['vix'].mean():>7.1f}  {sub['rv_forward'].mean():>9.1f}  "
              f"{vrp:>+5.1f}  {neg_pct:>5.1f}%  {sub['vix'].min():>7.1f}  "
              f"{sub['rv_forward'].max():>7.1f}  {verdict:>12}")
    print()

    # ================================================================
    # 2. Period comparisons
    # ================================================================
    print("--- Period Comparison ---")
    print()
    periods = [
        ("All History (<2025)", full_hist),
        ("H1 2025 (Jan-Jun)", h1_2025),
        ("H2 2025 (Jul-Dec)", h2_2025),
        ("Sep-Dec 2025", df[(df.index >= '2025-09-01') & (df.index < '2026-01-01')]),
        ("Q1 2026 (Jan-Mar)", q1_2026),
        ("Sep25-Mar26 (full)", window),
    ]

    print(f"  {'Period':>25}  {'Days':>5}  {'Avg VIX':>8}  {'Avg RV':>7}  {'VRP':>6}  "
          f"{'VRP<0%':>7}  {'Avg |Move|':>10}")
    print(f"  {'─'*25}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*10}")

    for label, subset in periods:
        if len(subset) < 5:
            continue
        daily_move = spy_ret.loc[subset.index].abs().mean() * 100
        neg_pct = (subset['vrp'] < 0).mean() * 100
        print(f"  {label:>25}  {len(subset):>5}  {subset['vix'].mean():>7.1f}  "
              f"{subset['rv_forward'].mean():>6.1f}  {subset['vrp'].mean():>+5.1f}  "
              f"{neg_pct:>5.1f}%  {daily_move:>8.2f}%")
    print()

    # ================================================================
    # 3. Tail risk in the window
    # ================================================================
    print("--- Tail Risk: Big Move Days ---")
    print()
    spy_ret_abs = spy_ret.abs() * 100

    for threshold in [1.0, 1.5, 2.0]:
        print(f"  Days with |SPY| > {threshold}%:")
        for label, start, end in [
            ("All History", None, '2025-01-01'),
            ("H1 2025", '2025-01-01', '2025-07-01'),
            ("Sep25-Mar26", '2025-09-01', '2026-04-01'),
        ]:
            if start is None:
                mask = (spy_ret_abs.index < end) & (spy_ret_abs > threshold)
                total = (spy_ret_abs.index < end).sum()
            else:
                mask = (spy_ret_abs.index >= start) & (spy_ret_abs.index < end) & (spy_ret_abs > threshold)
                total = ((spy_ret_abs.index >= start) & (spy_ret_abs.index < end)).sum()
            count = mask.sum()
            pct = count / total * 100 if total > 0 else 0
            print(f"    {label:>15}: {count:>3} days ({pct:.1f}%)")
        print()

    # ================================================================
    # 4. VIX underpriced episodes in the window
    # ================================================================
    print("--- Underpriced Episodes (VRP < -3) in Sep25-Mar26 ---")
    print()
    underpriced = window[window['vrp'] < -3]
    if len(underpriced) > 0:
        print(f"  {'Date':>12}  {'VIX':>6}  {'Fwd RV':>7}  {'VRP':>6}  {'SPY':>6}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}")
        for date, row in underpriced.iterrows():
            print(f"  {date.strftime('%Y-%m-%d'):>12}  {row['vix']:>5.1f}  {row['rv_forward']:>6.1f}  "
                  f"{row['vrp']:>+5.1f}  {row['spy']:>.0f}")
        print(f"\n  Total: {len(underpriced)} days significantly underpriced")
    else:
        print("  No days with VRP < -3 in this window.")
    print()

    # ================================================================
    # 5. VIX term structure: was near-term vol cheap?
    # ================================================================
    print("--- VIX Level Distribution: Sep25-Mar26 ---")
    print()
    for bucket_label, lo, hi in [("<15", 0, 15), ("15-20", 15, 20), ("20-25", 20, 25), ("25-30", 25, 30), (">30", 30, 100)]:
        ct = ((window['vix'] >= lo) & (window['vix'] < hi)).sum()
        pct = ct / len(window) * 100
        bar = '█' * int(pct / 2)
        print(f"  VIX {bucket_label:>5}: {ct:>3} days ({pct:>5.1f}%) {bar}")
    print()

    # ================================================================
    # 6. Overnight vs intraday vol
    # ================================================================
    print("--- Overnight vs Intraday Volatility ---")
    print()
    spy_full = yf.Ticker("SPY").history(period="2y")
    spy_full.index = spy_full.index.tz_localize(None)

    for label, start, end in [
        ("H1 2025", '2025-01-01', '2025-07-01'),
        ("Sep-Dec 2025", '2025-09-01', '2026-01-01'),
        ("Q1 2026", '2026-01-01', '2026-04-01'),
        ("Sep25-Mar26", '2025-09-01', '2026-04-01'),
    ]:
        sdf = spy_full[(spy_full.index >= start) & (spy_full.index < end)]
        if len(sdf) < 10:
            continue
        overnight = (sdf['Open'] / sdf['Close'].shift(1) - 1).dropna()
        intraday = (sdf['Close'] / sdf['Open'] - 1).dropna()
        total = (sdf['Close'] / sdf['Close'].shift(1) - 1).dropna()

        overnight_vol = overnight.std() * np.sqrt(252) * 100
        intraday_vol = intraday.std() * np.sqrt(252) * 100
        total_vol = total.std() * np.sqrt(252) * 100
        overnight_share = overnight.var() / total.var() * 100 if total.var() > 0 else 0

        print(f"  {label:>15}: Total={total_vol:>5.1f}%  Overnight={overnight_vol:>5.1f}%  "
              f"Intraday={intraday_vol:>5.1f}%  O/N share={overnight_share:>4.0f}%")
    print()

    # ================================================================
    # 7. Was the vol regime "complacent"?
    # ================================================================
    print("--- Complacency Check: VIX Percentile ---")
    print()
    all_vix = df['vix']
    for label, subset in [("Sep25-Mar26", window), ("H1 2025", h1_2025)]:
        avg_vix = subset['vix'].mean()
        pctile = (all_vix < avg_vix).mean() * 100
        print(f"  {label}: Avg VIX={avg_vix:.1f} → {pctile:.0f}th percentile of all history")
    print()

    # ================================================================
    # CHARTS
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    # Chart 1: VIX vs Realized Vol in the window
    ax = axes[0]
    ax.plot(window.index, window['vix'], label='VIX (Implied)', color='navy', linewidth=1.5)
    ax.plot(window.index, window['rv_forward'], label='Forward 21d Realized Vol',
            color='red', linewidth=1.2, alpha=0.8)
    ax.plot(window.index, window['rv_trailing'], label='Trailing 21d Realized Vol',
            color='orange', linewidth=1.0, alpha=0.6, linestyle='--')
    ax.fill_between(window.index, window['vix'], window['rv_forward'],
                     where=window['vix'] < window['rv_forward'],
                     alpha=0.3, color='red', label='VIX Underpriced')
    ax.fill_between(window.index, window['vix'], window['rv_forward'],
                     where=window['vix'] >= window['rv_forward'],
                     alpha=0.2, color='green', label='VIX Overpriced')
    ax.set_title('VIX vs Realized Vol: Sep 2025 - Mar 2026', fontsize=14, fontweight='bold')
    ax.set_ylabel('Volatility (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Chart 2: Daily VRP
    ax = axes[1]
    colors = ['red' if v < 0 else 'green' for v in window['vrp']]
    ax.bar(window.index, window['vrp'], color=colors, alpha=0.6, width=1)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=window['vrp'].mean(), color='blue', linestyle='--', alpha=0.5,
               label=f'Period avg: {window["vrp"].mean():.1f}')
    ax.axhline(y=full_hist['vrp'].mean(), color='purple', linestyle=':', alpha=0.5,
               label=f'Historical avg: {full_hist["vrp"].mean():.1f}')
    ax.set_title('Daily Variance Risk Premium: Sep 2025 - Mar 2026', fontsize=14, fontweight='bold')
    ax.set_ylabel('VRP (VIX - Realized Vol, pts)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Chart 3: SPY daily returns with VIX overlay
    ax = axes[2]
    spy_window = spy_ret.loc[window.index] * 100
    colors_ret = ['red' if r < 0 else 'green' for r in spy_window]
    ax.bar(spy_window.index, spy_window.values, color=colors_ret, alpha=0.5, width=1)
    ax2 = ax.twinx()
    ax2.plot(window.index, window['vix'], color='navy', linewidth=1.2, alpha=0.7, label='VIX')
    ax.set_title('SPY Daily Returns with VIX Overlay: Sep 2025 - Mar 2026', fontsize=14, fontweight='bold')
    ax.set_ylabel('SPY Daily Return (%)')
    ax2.set_ylabel('VIX Level', color='navy')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/alejandorosumah-mansa/vix_sep25_mar26_charts.png', dpi=150, bbox_inches='tight')
    print("Charts saved to vix_sep25_mar26_charts.png")
    print("\nDONE")

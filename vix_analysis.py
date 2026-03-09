"""
VIX Historical Analysis
========================
1. Historic VIX price behavior & statistics
2. Straddle profitability analysis (using VIX as IV proxy)
3. Major spike events, drivers, and recovery timelines
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: HISTORICAL VIX PRICES
# ============================================================

def fetch_vix_data():
    """Fetch full VIX history from Yahoo Finance."""
    vix = yf.Ticker("^VIX")
    df = vix.history(period="max")
    df.index = df.index.tz_localize(None)
    return df

def analyze_vix_history(df):
    """Compute key VIX statistics."""
    print("=" * 70)
    print("PART 1: VIX HISTORICAL PRICE ANALYSIS")
    print("=" * 70)
    print(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(df):,}")
    print()

    close = df['Close']
    print("--- VIX Summary Statistics ---")
    print(f"  Mean:    {close.mean():.2f}")
    print(f"  Median:  {close.median():.2f}")
    print(f"  Std Dev: {close.std():.2f}")
    print(f"  Min:     {close.min():.2f} ({close.idxmin().strftime('%Y-%m-%d')})")
    print(f"  Max:     {close.max():.2f} ({close.idxmax().strftime('%Y-%m-%d')})")
    print(f"  25th %:  {close.quantile(0.25):.2f}")
    print(f"  75th %:  {close.quantile(0.75):.2f}")
    print(f"  90th %:  {close.quantile(0.90):.2f}")
    print(f"  95th %:  {close.quantile(0.95):.2f}")
    print(f"  99th %:  {close.quantile(0.99):.2f}")
    print()

    # Time spent in ranges
    print("--- Time Spent in VIX Ranges ---")
    ranges = [(0, 12), (12, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 100)]
    for low, high in ranges:
        pct = ((close >= low) & (close < high)).mean() * 100
        print(f"  VIX {low:>3}-{high:<3}: {pct:5.1f}% of days")
    print()

    # VIX regime analysis
    print("--- VIX Regime Characteristics ---")
    regimes = {
        "Low Vol (VIX < 15)": close < 15,
        "Normal (15-20)": (close >= 15) & (close < 20),
        "Elevated (20-30)": (close >= 20) & (close < 30),
        "High (30-50)": (close >= 30) & (close < 50),
        "Extreme (50+)": close >= 50,
    }
    for name, mask in regimes.items():
        if mask.sum() > 0:
            print(f"  {name}: {mask.mean()*100:.1f}% of days, avg VIX={close[mask].mean():.1f}")
    print()

    return close


# ============================================================
# PART 2: STRADDLE PROFITABILITY ANALYSIS
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def analyze_straddle_profitability(vix_df):
    """
    Simulate ATM straddle on SPY using VIX as implied volatility.
    For each day, price an ATM straddle and check P&L at expiration.
    We test 30-day (monthly) straddles.
    """
    print("=" * 70)
    print("PART 2: HISTORICAL STRADDLE PROFITABILITY ANALYSIS")
    print("=" * 70)
    print("Methodology: ATM straddle on SPY, using VIX/100 as annualized IV,")
    print("priced via Black-Scholes. Holding period = 30 calendar days.")
    print()

    # Fetch SPY data
    spy = yf.Ticker("SPY")
    spy_df = spy.history(period="max")
    spy_df.index = spy_df.index.tz_localize(None)

    # Align dates
    common_dates = vix_df.index.intersection(spy_df.index)
    vix_close = vix_df.loc[common_dates, 'Close']
    spy_close = spy_df.loc[common_dates, 'Close']

    results = []
    holding_days = 21  # ~30 calendar days = 21 trading days

    for i in range(len(common_dates) - holding_days):
        entry_date = common_dates[i]
        exit_date = common_dates[i + holding_days]

        S = spy_close.iloc[i]
        K = S  # ATM
        sigma = vix_close.iloc[i] / 100.0  # VIX is in percentage points
        T = 30 / 365.0  # 30 calendar days
        r = 0.02  # approximate risk-free rate

        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        straddle_cost = call_price + put_price

        S_exit = spy_close.iloc[i + holding_days]
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
            'vix_at_entry': vix_close.iloc[i],
            'straddle_cost': straddle_cost,
            'straddle_payoff': straddle_payoff,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'profitable': pnl > 0,
        })

    rdf = pd.DataFrame(results)

    print(f"Total straddles analyzed: {len(rdf):,}")
    print(f"Date range: {rdf['entry_date'].iloc[0].strftime('%Y-%m-%d')} to {rdf['entry_date'].iloc[-1].strftime('%Y-%m-%d')}")
    print()

    print("--- Overall Straddle P&L ---")
    print(f"  Win rate:        {rdf['profitable'].mean()*100:.1f}%")
    print(f"  Avg P&L ($):     ${rdf['pnl'].mean():.2f}")
    print(f"  Avg P&L (%):     {rdf['pnl_pct'].mean():.1f}%")
    print(f"  Median P&L (%):  {rdf['pnl_pct'].median():.1f}%")
    print(f"  Avg cost:        ${rdf['straddle_cost'].mean():.2f}")
    print(f"  Avg payoff:      ${rdf['straddle_payoff'].mean():.2f}")
    print()

    # By VIX regime at entry
    print("--- Straddle P&L by VIX Regime at Entry ---")
    bins = [0, 12, 15, 20, 25, 30, 40, 100]
    labels = ['<12', '12-15', '15-20', '20-25', '25-30', '30-40', '40+']
    rdf['vix_bin'] = pd.cut(rdf['vix_at_entry'], bins=bins, labels=labels)

    for label in labels:
        subset = rdf[rdf['vix_bin'] == label]
        if len(subset) > 10:
            print(f"  VIX {label:>5}: Win={subset['profitable'].mean()*100:5.1f}%  "
                  f"Avg P&L={subset['pnl_pct'].mean():+6.1f}%  "
                  f"Avg $P&L={subset['pnl'].mean():+7.2f}  "
                  f"N={len(subset):,}")
    print()

    # By decade
    print("--- Straddle P&L by Decade ---")
    rdf['decade'] = (rdf['entry_date'].dt.year // 10) * 10
    for decade in sorted(rdf['decade'].unique()):
        subset = rdf[rdf['decade'] == decade]
        print(f"  {decade}s: Win={subset['profitable'].mean()*100:5.1f}%  "
              f"Avg P&L={subset['pnl_pct'].mean():+6.1f}%  N={len(subset):,}")
    print()

    # Selling straddles (inverse)
    print("--- SHORT Straddle (Selling) P&L ---")
    print(f"  Win rate:        {(~rdf['profitable']).mean()*100:.1f}%")
    print(f"  Avg P&L ($):     ${-rdf['pnl'].mean():.2f}")
    print(f"  Avg P&L (%):     {-rdf['pnl_pct'].mean():.1f}%")
    print()

    return rdf


# ============================================================
# PART 3: VIX SPIKE EVENTS & RECOVERY ANALYSIS
# ============================================================

def analyze_spike_events(df):
    """Analyze major VIX spike events, their causes, and recovery times."""
    print("=" * 70)
    print("PART 3: VIX SPIKE EVENTS, DRIVERS & RECOVERY TIMELINES")
    print("=" * 70)
    print()

    close = df['Close']

    # Define major historical VIX spike events
    events = [
        {
            'name': 'LTCM / Russian Crisis',
            'date': '1998-10-08',
            'driver': 'Russian debt default, LTCM collapse, emerging market contagion',
        },
        {
            'name': '9/11 Attacks',
            'date': '2001-09-20',
            'driver': 'September 11 terrorist attacks, market shutdown & reopening',
        },
        {
            'name': 'WorldCom / Enron Accounting Scandals',
            'date': '2002-07-22',
            'driver': 'Corporate fraud scandals, accounting crises, investor confidence collapse',
        },
        {
            'name': 'Bear Stearns Collapse',
            'date': '2008-03-17',
            'driver': 'Bear Stearns hedge fund failures, subprime mortgage crisis begins',
        },
        {
            'name': 'Lehman Brothers Bankruptcy',
            'date': '2008-09-15',
            'driver': 'Lehman bankruptcy, AIG bailout, global financial system near-collapse',
        },
        {
            'name': 'GFC Peak Fear',
            'date': '2008-11-20',
            'driver': 'Peak of Global Financial Crisis, bank solvency fears, auto bailouts',
        },
        {
            'name': 'Flash Crash',
            'date': '2010-05-06',
            'driver': 'Flash crash - Dow dropped ~1000pts in minutes, HFT/market structure fears',
        },
        {
            'name': 'US Debt Downgrade',
            'date': '2011-08-08',
            'driver': 'S&P downgrades US debt from AAA, European sovereign debt crisis',
        },
        {
            'name': 'China Yuan Devaluation',
            'date': '2015-08-24',
            'driver': 'China surprise yuan devaluation, global growth fears, "Black Monday"',
        },
        {
            'name': 'Volmageddon',
            'date': '2018-02-05',
            'driver': 'XIV/short-vol blow-up, VIX doubled in a day, systematic strategy unwind',
        },
        {
            'name': 'COVID-19 Pandemic',
            'date': '2020-03-16',
            'driver': 'Global pandemic lockdowns, economic shutdown fears, liquidity crisis',
        },
        {
            'name': 'Russia-Ukraine War',
            'date': '2022-02-24',
            'driver': 'Russian invasion of Ukraine, energy crisis, nuclear fears',
        },
    ]

    print(f"{'Event':<35} {'Date':>10}  {'VIX Peak':>9}  {'Pre-VIX':>8}  {'Spike':>7}  {'Days to <25':>11}  {'Days to <20':>11}")
    print("-" * 110)

    event_data = []
    for event in events:
        try:
            event_date = pd.Timestamp(event['date'])

            # Find nearest trading day
            idx = close.index.searchsorted(event_date)
            if idx >= len(close):
                continue
            actual_date = close.index[idx]

            # Pre-event VIX (20 trading days before)
            pre_idx = max(0, idx - 20)
            pre_vix = close.iloc[pre_idx:idx].mean()

            # Find peak VIX in window around event (+/- 30 trading days)
            window_start = max(0, idx - 5)
            window_end = min(len(close), idx + 60)
            peak_vix = close.iloc[window_start:window_end].max()
            peak_date = close.iloc[window_start:window_end].idxmax()

            spike_pct = ((peak_vix - pre_vix) / pre_vix) * 100

            # Days to recover below 25
            post_peak_idx = close.index.searchsorted(peak_date)
            days_to_25 = None
            days_to_20 = None
            for j in range(post_peak_idx, min(post_peak_idx + 500, len(close))):
                if days_to_25 is None and close.iloc[j] < 25:
                    days_to_25 = j - post_peak_idx
                if days_to_20 is None and close.iloc[j] < 20:
                    days_to_20 = j - post_peak_idx
                if days_to_25 is not None and days_to_20 is not None:
                    break

            d25 = f"{days_to_25}d" if days_to_25 is not None else "N/A"
            d20 = f"{days_to_20}d" if days_to_20 is not None else "N/A"

            print(f"  {event['name']:<33} {actual_date.strftime('%Y-%m-%d'):>10}  "
                  f"{peak_vix:>8.1f}  {pre_vix:>7.1f}  {spike_pct:>+6.0f}%  "
                  f"{d25:>11}  {d20:>11}")

            event_data.append({
                'name': event['name'],
                'date': actual_date,
                'peak_date': peak_date,
                'peak_vix': peak_vix,
                'pre_vix': pre_vix,
                'spike_pct': spike_pct,
                'days_to_25': days_to_25,
                'days_to_20': days_to_20,
                'driver': event['driver'],
            })
        except Exception as e:
            print(f"  {event['name']:<33} Error: {e}")

    print()
    print("--- Detailed Event Narratives ---")
    for ev in event_data:
        print(f"\n  {ev['name']} ({ev['date'].strftime('%Y-%m-%d')})")
        print(f"    Driver: {ev['driver']}")
        print(f"    VIX went from ~{ev['pre_vix']:.0f} to {ev['peak_vix']:.1f} (peak on {ev['peak_date'].strftime('%Y-%m-%d')})")
        print(f"    Spike magnitude: {ev['spike_pct']:+.0f}%")
        if ev['days_to_25'] is not None:
            print(f"    Recovery to VIX<25: {ev['days_to_25']} trading days (~{ev['days_to_25']*1.4:.0f} calendar days)")
        if ev['days_to_20'] is not None:
            print(f"    Recovery to VIX<20: {ev['days_to_20']} trading days (~{ev['days_to_20']*1.4:.0f} calendar days)")
    print()

    # Summary statistics on spikes
    edf = pd.DataFrame(event_data)
    print("--- Spike Recovery Summary ---")
    print(f"  Average peak VIX during crises:       {edf['peak_vix'].mean():.1f}")
    print(f"  Average spike magnitude:               {edf['spike_pct'].mean():+.0f}%")
    valid_25 = edf[edf['days_to_25'].notna()]['days_to_25']
    valid_20 = edf[edf['days_to_20'].notna()]['days_to_20']
    if len(valid_25) > 0:
        print(f"  Average days to recover below 25:      {valid_25.mean():.0f} trading days")
    if len(valid_20) > 0:
        print(f"  Average days to recover below 20:      {valid_20.mean():.0f} trading days")
    print(f"  Fastest recovery (to <25):             {valid_25.min():.0f}d — {edf.loc[valid_25.idxmin(), 'name']}")
    print(f"  Slowest recovery (to <25):             {valid_25.max():.0f}d — {edf.loc[valid_25.idxmax(), 'name']}")
    print()

    return event_data


# ============================================================
# PART 4: GENERATE CHARTS
# ============================================================

def generate_charts(vix_df, straddle_df, events):
    """Generate visualization charts."""

    close = vix_df['Close']
    fig, axes = plt.subplots(4, 1, figsize=(16, 24))

    # Chart 1: VIX Historical Price
    ax = axes[0]
    ax.plot(close.index, close.values, linewidth=0.5, color='navy')
    ax.axhline(y=close.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean={close.mean():.1f}')
    ax.axhline(y=close.median(), color='green', linestyle='--', alpha=0.7, label=f'Median={close.median():.1f}')
    ax.axhline(y=20, color='orange', linestyle=':', alpha=0.5, label='VIX=20')
    for ev in events:
        ax.annotate(ev['name'], xy=(ev['peak_date'], ev['peak_vix']),
                     fontsize=6, ha='center', va='bottom', rotation=45,
                     arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                     xytext=(ev['peak_date'], ev['peak_vix'] + 5))
    ax.set_title('VIX Historical Price (Full History)', fontsize=14, fontweight='bold')
    ax.set_ylabel('VIX Level')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 2: VIX Distribution
    ax = axes[1]
    ax.hist(close.values, bins=100, edgecolor='navy', alpha=0.7, color='steelblue', density=True)
    ax.axvline(x=close.mean(), color='red', linestyle='--', label=f'Mean={close.mean():.1f}')
    ax.axvline(x=close.median(), color='green', linestyle='--', label=f'Median={close.median():.1f}')
    ax.set_title('VIX Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('VIX Level')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 3: Rolling Straddle Win Rate
    ax = axes[2]
    rolling_win = straddle_df.set_index('entry_date')['profitable'].rolling(252).mean() * 100
    ax.plot(rolling_win.index, rolling_win.values, linewidth=0.8, color='darkgreen')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% breakeven')
    ax.axhline(y=straddle_df['profitable'].mean()*100, color='blue', linestyle='--',
               alpha=0.7, label=f'Overall avg={straddle_df["profitable"].mean()*100:.1f}%')
    ax.set_title('Rolling 1-Year Straddle Win Rate (30-day ATM on SPY)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Chart 4: Straddle P&L by VIX at entry
    ax = axes[3]
    bins = [0, 12, 15, 20, 25, 30, 40, 100]
    labels = ['<12', '12-15', '15-20', '20-25', '25-30', '30-40', '40+']
    straddle_df['vix_bin'] = pd.cut(straddle_df['vix_at_entry'], bins=bins, labels=labels)
    grouped = straddle_df.groupby('vix_bin', observed=False).agg(
        win_rate=('profitable', 'mean'),
        avg_pnl=('pnl_pct', 'mean'),
        count=('pnl', 'count')
    )
    colors = ['green' if x > 0 else 'red' for x in grouped['avg_pnl']]
    bars = ax.bar(grouped.index.astype(str), grouped['avg_pnl'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.5)
    for bar, wr, cnt in zip(bars, grouped['win_rate'], grouped['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'Win:{wr*100:.0f}%\nN:{cnt}', ha='center', va='bottom', fontsize=7)
    ax.set_title('Avg Straddle Return by VIX at Entry', fontsize=14, fontweight='bold')
    ax.set_xlabel('VIX Range at Entry')
    ax.set_ylabel('Avg P&L (%)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/alejandorosumah-mansa/vix_analysis_charts.png', dpi=150, bbox_inches='tight')
    print("Charts saved to vix_analysis_charts.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("Fetching VIX data...")
    vix_df = fetch_vix_data()

    close = analyze_vix_history(vix_df)

    print("Fetching SPY data for straddle analysis...")
    straddle_df = analyze_straddle_profitability(vix_df)

    events = analyze_spike_events(vix_df)

    print("\nGenerating charts...")
    generate_charts(vix_df, straddle_df, events)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

"""
TEST TASK: Futures vs Spot Basis Trading

Author: Kirill Serdyukov

DESCRIPTION:
This project implements an end-to-end research pipeline for analyzing and trading the basis between a futures contract and its underlying spot asset.

The focus of this work is on building a robust statistical trading framework rather than optimizing for immediate profitability.

Pipeline includes:
1) Limit order book reconstruction from raw event data
2) Construction of mid-prices from order book snapshots
3) Time synchronization of spot and futures markets
4) Basis spread estimation with dynamic hedge ratio
5) Mean-reversion trading strategy based on statistical deviations
   (Kalman-filter-based hedge ratio estimation)
6) Backtesting with realistic transaction costs and slippage assumptions

INSTRUMENTS:
- Spot: SBER
- Futures: SBERF

NOTES:
- The hedge ratio is estimated dynamically using a Kalman filter to account for time-varying market relationships.
- The spread is modeled as a mean-reverting process for statistical arbitrage purposes.
- The backtest includes simplified execution assumptions and does not account for full market impact or queue position effects.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from collections import defaultdict

print("="*60)
print("АНАЛИЗ СПРЕДА: ФЬЮЧЕРС vs СПОТ")
print("="*60)

# =====================================================
# 1. ЗАГРУЗКА
# =====================================================

spot   = pd.read_parquet('20250610_SBER.parquet')
future = pd.read_parquet('SBERF_2025_06_10.parquet')

print(f"Spot: {len(spot):,}")
print(f"Future: {len(future):,}")

# =====================================================
# 2. ВРЕМЯ
# =====================================================

def parse_spot_time(x):
    s = str(int(x)).zfill(12)
    return pd.Timestamp(f"2025-06-10 {s[0:2]}:{s[2:4]}:{s[4:6]}.{s[6:12]}")

spot['datetime'] = spot['TIME'].apply(parse_spot_time)

future['datetime'] = pd.to_datetime(
    future['MOMENT'].astype(str).str[:14],
    format='%Y%m%d%H%M%S', errors='coerce'
)
future = future.dropna(subset=['datetime'])

spot   = spot.sort_values('datetime').reset_index(drop=True)
future = future.sort_values('datetime').reset_index(drop=True)

# =====================================================
# 3. РЕКОНСТРУКЦИЯ ЛИМИТНОГО СТАКАНА (ORDER BOOK RECONSTRUCTION)
# =====================================================

def build_order_book(df, instrument='spot', snapshot_interval_ms=200):
    if instrument == 'spot':
        id_col, side_col = 'ORDERNO', 'BUYSELL'
    else:
        id_col, side_col = 'ID', 'TYPE'

    df = df.copy()
    df['side'] = df[side_col].astype(str).str.strip()
    df = df[df['PRICE'] > 0]
    
    # Сортировка ПОСЛЕ фильтра, и больше не пересортировываем
    df['action_order'] = df['ACTION'].map({1: 0, 2: 1, 0: 2})
    df = df.sort_values(['datetime', 'action_order']).drop(columns='action_order').reset_index(drop=True)

    snap_start = pd.Timestamp('2025-06-10 09:50:00')
    snap_end   = pd.Timestamp('2025-06-10 18:50:00')

    bids_prices = {}
    asks_prices = {}
    bids_sl = SortedList()
    asks_sl = SortedList()

    snapshots = []
    last_snap = None
    interval  = pd.Timedelta(milliseconds=snapshot_interval_ms)

    first_trade_time = df[df['ACTION'] == 2]['datetime'].min()
    print(f"  [{instrument}] первая сделка: {first_trade_time}")

    for row in df.itertuples(index=False):
        oid    = getattr(row, id_col)
        side   = row.side
        price  = row.PRICE
        action = row.ACTION
        t      = row.datetime

        if action == 1:
            if side == 'B':
                if oid in bids_prices: bids_sl.discard(bids_prices[oid])
                bids_prices[oid] = price
                bids_sl.add(price)
            elif side == 'S':
                if oid in asks_prices: asks_sl.discard(asks_prices[oid])
                asks_prices[oid] = price
                asks_sl.add(price)

        elif action in (0, 2):
            if side == 'B' and oid in bids_prices:
                bids_sl.discard(bids_prices.pop(oid))
            elif side == 'S' and oid in asks_prices:
                asks_sl.discard(asks_prices.pop(oid))

        if t < snap_start or t > snap_end:
            continue
        if not bids_sl or not asks_sl:
            continue
        if last_snap is not None and (t - last_snap) < interval:
            continue

        bb = bids_sl[-1]
        ba = asks_sl[0]

        if bb < ba:
            snapshots.append({
                'datetime': t,
                'best_bid': bb,
                'best_ask': ba,
                'mid':      (bb + ba) / 2
            })
            last_snap = t

    print(f"  [{instrument}] snapshots: {len(snapshots)}, "
          f"bids: {len(bids_prices)}, asks: {len(asks_prices)}")
    return pd.DataFrame(snapshots)


def get_book_at(df_events, instrument, timestamp, depth=5):
    if instrument == 'spot':
        id_col, side_col = 'ORDERNO', 'BUYSELL'
        BID, ASK = 'B', 'S'
    else:
        id_col, side_col = 'ID', 'TYPE'
        BID, ASK = 'B', 'S'

    df = df_events.copy()
    df['side'] = df[side_col].astype(str).str.strip()
    df = df[(df['PRICE'] > 0) & (df['datetime'] <= timestamp)].sort_values('datetime')
    df['action_order'] = df['ACTION'].map({1: 0, 2: 1, 0: 2})
    df = df.sort_values(['datetime', 'action_order']).drop(columns='action_order').reset_index(drop=True)

    vol_map = {}
    for row in df.itertuples(index=False):
        oid = getattr(row, id_col)
        if row.ACTION == 1:
            vol_map[oid] = (row.PRICE, row.VOLUME, row.side)
        elif row.ACTION in (0, 2):
            vol_map.pop(oid, None)

    bid_levels = defaultdict(float)
    ask_levels = defaultdict(float)
    for oid, (price, vol, side) in vol_map.items():
        if side == BID:   bid_levels[price] += vol
        elif side == ASK: ask_levels[price] += vol

    bid_book = sorted(bid_levels.items(), reverse=True)[:depth]
    ask_book = sorted(ask_levels.items())[:depth]

    print(f"\n{'='*42}")
    print(f"Стакан {instrument.upper()} на {timestamp}")
    print(f"{'='*42}")
    print(f"{'Цена':>10}  {'Объём':>10}  {'Сторона'}")
    print(f"{'-'*35}")
    for price, vol in reversed(ask_book):
        print(f"{price:>10.2f}  {vol:>10.0f}  ASK  ↑")
    print(f"{'··· СПРЕД ···':^35}")
    for price, vol in bid_book:
        print(f"{price:>10.2f}  {vol:>10.0f}  BID  ↓")

    if bid_book and ask_book:
        bb = bid_book[0][0]
        ba = ask_book[0][0]
        print(f"\n  Best bid: {bb:.2f}")
        print(f"  Best ask: {ba:.2f}")
        print(f"  Spread:   {ba-bb:.2f}")
        print(f"  Valid:    {bb < ba}")

    return bid_book, ask_book


# =====================================================
# 4. ПОСТРОЕНИЕ СНАПШОТОВ СТАКАНА (BEST BID / ASK, MID)
# =====================================================

print("\nСтроим стаканы...")
spot_book   = build_order_book(spot,   'spot',   200)
future_book = build_order_book(future, 'future', 200)

print(f"\nSpot snapshots:   {len(spot_book)}")
print(f"Future snapshots: {len(future_book)}")

# Пример стакана на произвольный момент
example_time = pd.Timestamp('2025-06-10 17:15:39')
get_book_at(spot,   'spot',   example_time, depth=5)
get_book_at(future, 'future', example_time, depth=5)

# =====================================================
# 5. ЕСЛИ СТАКАН НЕ РАБОТАЕТ — FALLBACK НА СДЕЛКИ
# =====================================================

if len(spot_book) < 100 or len(future_book) < 100:
    print("\n⚠️  Мало снапшотов из стакана, используем цены сделок как mid")

    snap_start = pd.Timestamp('2025-06-10 09:50:00')
    snap_end   = pd.Timestamp('2025-06-10 18:50:00')

    spot_trades = spot[
        (spot['ACTION'] == 2) &
        (spot['TRADEPRICE'] > 0) &
        (spot['datetime'] >= snap_start) &
        (spot['datetime'] <= snap_end)
    ].copy()

    spot_mid = (spot_trades
        .set_index('datetime')['TRADEPRICE']
        .resample('1s').last()
        .ffill()
        .reset_index()
        .rename(columns={'TRADEPRICE': 'mid'})
    )
    spot_mid.columns = ['datetime', 'mid']

    future_trades = future[
        (future['ACTION'] == 2) &
        (future['PRICE_DEAL'] > 0) &
        (future['datetime'] >= snap_start) &
        (future['datetime'] <= snap_end)
    ].copy()

    future_mid = (future_trades
        .set_index('datetime')['PRICE_DEAL']
        .resample('1s').last()
        .ffill()
        .reset_index()
        .rename(columns={'PRICE_DEAL': 'mid'})
    )

    spot_book   = spot_mid
    future_book = future_mid
    use_mid_col = 'mid'
else:
    use_mid_col = 'mid'

# =====================================================
# 6. СИНХРОНИЗАЦИЯ СПОТ И ФЬЮЧЕРСА (MERGE_ASOF)
# =====================================================

merged = pd.merge_asof(
    spot_book.sort_values('datetime').rename(columns={use_mid_col: 'mid_spot'}),
    future_book.sort_values('datetime').rename(columns={use_mid_col: 'mid_future'}),
    on='datetime',
    direction='nearest',
    tolerance=pd.Timedelta('1s')
).dropna(subset=['mid_spot', 'mid_future'])

print(f"\nMerged points: {len(merged):,}")

# =====================================================
# 7. РАСЧЁТ СПРЕДА (FUTURE - SPOT)
# =====================================================

merged['spread'] = merged['mid_future'] - merged['mid_spot']

print("\nSpread stats:")
print(merged['spread'].describe())


# =====================================================
# 8. ГРАФИКИ
# =====================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 14))

axes[0].plot(merged['datetime'], merged['mid_spot'],   label='Spot',   linewidth=0.8)
axes[0].plot(merged['datetime'], merged['mid_future'], label='Future', linewidth=0.8)
axes[0].set_title('Mid price: Spot vs Future')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(merged['datetime'], merged['spread'], linewidth=0.8, label='Spread')
axes[1].set_title('Spread (Future - Spot)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('spread_analysis.png', dpi=150)
plt.show()

print("\nГОТОВО 🚀")


# =====================================================
# PART 2: TRADING STRATEGY
# =====================================================

# =====================================================
# PARAMETERS
# =====================================================


WINDOW_VOL = 200
LOT = 100

SPOT_FEE = 0.0005
FUT_FEE = 1.0
SLIPPAGE = 0.01

COOLDOWN = 5  # bars

df = merged.copy().sort_values("datetime").reset_index(drop=True)

# =====================================================
# 1. KALMAN HEDGE RATIO
# =====================================================

beta = 1.0
P = 1.0
Q = 1e-5
R = 1e-2

betas = []

for i in range(len(df)):
    if i == 0:
        betas.append(beta)
        continue

    x = df.loc[i, "mid_spot"]
    y = df.loc[i, "mid_future"]

    P = P + Q
    K = P * x / (x * P * x + R)

    beta = beta + K * (y - beta * x)
    P = (1 - K * x) * P

    betas.append(beta)

df["beta"] = betas

# =====================================================
# 2. CONSISTENT SPREAD
# =====================================================

df["spread"] = df["mid_future"] - df["beta"] * df["mid_spot"]

# =====================================================
# 3. VOL FILTER (ROBUST)
# =====================================================

df["vol"] = df["spread"].rolling(WINDOW_VOL).std()
vol_thresh = df["vol"].quantile(0.6)

# =====================================================
# 4. THRESHOLDS (ROBUST QUANTILES)
# =====================================================

z = (df["spread"] - df["spread"].mean()) / df["spread"].std()

ENTRY_Z = 1.5
EXIT_Z = 0.3

# =====================================================
# 5. EXECUTION MODEL
# =====================================================

def exec_price(mid, side):
    return mid + SLIPPAGE if side == "buy" else mid - SLIPPAGE

# =====================================================
# 6. STATE
# =====================================================

pos_f = 0
pos_s = 0
pos_dir = 0

entry_f = 0
entry_s = 0

pnl = 0
curve = []

cooldown = 0

# =====================================================
# 7. BACKTEST LOOP
# =====================================================

for i, row in df.iterrows():

    spread_z = z[i]

    mtm = pnl

    # mark-to-market always
    if pos_dir != 0:
        mtm += pos_f * (row["mid_future"] - entry_f)
        mtm += pos_s * (row["mid_spot"] - entry_s)

    curve.append(mtm)

    if cooldown > 0:
        cooldown -= 1
        continue

    if np.isnan(row["vol"]) or row["vol"] < vol_thresh:
        continue

    # =====================
    # ENTRY
    # =====================

    if pos_dir == 0:

        if spread_z > ENTRY_Z:

            pos_f = -LOT
            pos_s = +LOT * row["beta"]
            pos_dir = -1

            entry_f = exec_price(row["mid_future"], "sell")
            entry_s = exec_price(row["mid_spot"], "buy")

            cooldown = COOLDOWN

        elif spread_z < -ENTRY_Z:

            pos_f = +LOT
            pos_s = -LOT * row["beta"]
            pos_dir = +1

            entry_f = exec_price(row["mid_future"], "buy")
            entry_s = exec_price(row["mid_spot"], "sell")

            cooldown = COOLDOWN

    # =====================
    # EXIT
    # =====================

    elif pos_dir != 0:

        if abs(spread_z) < EXIT_Z:

            exit_f = exec_price(row["mid_future"], "buy" if pos_f < 0 else "sell")
            exit_s = exec_price(row["mid_spot"], "sell" if pos_s > 0 else "buy")

            pnl_f = pos_f * (exit_f - entry_f)
            pnl_s = pos_s * (exit_s - entry_s)

            trade_pnl = pnl_f + pnl_s

            spot_cost = abs(pos_s) * (entry_s + exit_s) * SPOT_FEE
            fut_cost = FUT_FEE * 2

            pnl += trade_pnl - spot_cost - fut_cost

            pos_f = 0
            pos_s = 0
            pos_dir = 0

            cooldown = COOLDOWN

# =====================================================
# 8. RESULTS
# =====================================================

df["pnl"] = curve

print("="*60)
print("IMPROVED BASIS STRATEGY RESULTS")
print("="*60)

print(f"Final PnL: {df['pnl'].iloc[-1]:.2f}")
print(f"Max DD: {(df['pnl'] - df['pnl'].cummax()).min():.2f}")
print(f"Trades: {(df['pnl'].diff()!=0).sum()}")

# =====================================================
# 8. PLOT
# =====================================================

plt.figure(figsize=(12,5))
plt.plot(df["datetime"], df["pnl"])
plt.title("Hedge Fund Basis Strategy (Kalman + Regime Filter)")
plt.grid()
plt.show()








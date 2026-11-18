#!/usr/bin/env python3
"""
Coinbase spot PAPER trading bot: scan -> optimize -> trade -> learn -> compound.
No API keys needed (public market data only). No real orders are ever placed.

Usage:
  pip install requests pandas numpy
  python bot.py optimize   # test strategies on history, show what (if anything) has an edge
  python bot.py run        # hourly paper-trading loop
  python bot.py report     # show equity, trades, per-strategy stats
"""
import sys, json, time, itertools, os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG ----------------
CFG = dict(
    start_equity=1000.0,
    fee=0.006,          # per side. VERIFY your Coinbase fee tier; this is deliberately conservative
    slip=0.001,         # per side slippage assumption
    risk_per_trade=0.01,   # risk 1% of equity per trade (stop distance sizes the position)
    max_pos_pct=0.20,      # max 20% of equity in one coin
    max_positions=5,
    max_drawdown=0.15,     # kill switch: stop opening trades at -15% from peak
    top_pairs=30,          # liquid pairs to trade
    min_usd_vol=500_000,   # min 24h USD volume
    reopt_every=24,        # re-optimize every N cycles (hours)
    min_train_trades=20, min_test_trades=8, min_test_expectancy=0.003,
)
STATE_FILE, TRADES_FILE, EQUITY_FILE, DATA_FILE = "state.json", "trades.csv", "equity.csv", "data.json"
API = "https://api.exchange.coinbase.com"
COLS = ["t", "low", "high", "open", "close", "volume"]

# ---------------- DATA ----------------
def get(path, params=None):
    for i in range(4):
        r = requests.get(API + path, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(1 + i); continue
        r.raise_for_status()
        time.sleep(0.12)
        return r.json()
    raise RuntimeError("rate limited: " + path)

def usd_pairs():
    return [p["id"] for p in get("/products")
            if p["quote_currency"] == "USD" and p["status"] == "online"
            and not p.get("trading_disabled") and not p.get("auction_mode")]

def history(pid, bars=900, gran=3600):
    out, end = [], datetime.now(timezone.utc)
    while sum(len(x) for x in out) < bars:
        start = end - timedelta(seconds=gran * 300)
        rows = get(f"/products/{pid}/candles",
                   {"granularity": gran, "start": start.isoformat(), "end": end.isoformat()})
        if not rows: break
        out.append(pd.DataFrame(rows, columns=COLS)); end = start
    if not out: return None
    df = pd.concat(out).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return df.tail(bars).reset_index(drop=True)

def scan():
    """Rank ALL USD pairs by 24h dollar volume; return the most liquid."""
    ranked = []
    for pid in usd_pairs():
        try:
            s = get(f"/products/{pid}/stats")
            usd = float(s["last"]) * float(s["volume"])
            if usd >= CFG["min_usd_vol"]: ranked.append((usd, pid))
        except Exception:
            continue
    ranked.sort(reverse=True)
    return [p for _, p in ranked[:CFG["top_pairs"]]]

# ---------------- INDICATORS / STRATEGIES ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def sig_breakout(df, p):
    hh = df.high.rolling(p["n"]).max().shift(1)
    return (df.close > hh) & (df.volume > 1.5 * df.volume.rolling(24).mean()) & (df.close > ema(df.close, 100))

def sig_meanrev(df, p):
    m, s = df.close.rolling(20).mean(), df.close.rolling(20).std()
    return (rsi(df.close) < p["rsi"]) & (df.close < m - 2 * s)

def sig_trend(df, p):
    f, sl = ema(df.close, p["fast"]), ema(df.close, p["slow"])
    return (f > sl) & (f.shift() <= sl.shift()) & (rsi(df.close) > 50)

EXIT_GRID = [dict(stop_atr=a, tp_atr=b, max_hold=h) for a in (2, 3) for b in (3, 6) for h in (24, 72)]
STRATS = {
    "breakout": (sig_breakout, [dict(n=n) for n in (20, 48, 96)]),
    "meanrev":  (sig_meanrev,  [dict(rsi=r) for r in (25, 30, 35)]),
    "trend":    (sig_trend,    [dict(fast=f, slow=s) for f, s in ((12, 48), (24, 96), (8, 32))]),
}

# ---------------- SHARED EXIT LOGIC (same code in backtest and live) ----------------
def check_exit(pos, high, low, close):
    """Update position with one completed bar; return exit price or None. Stop checked first (conservative)."""
    pos["bars"] += 1
    if low <= pos["stop"]: return pos["stop"]
    if high >= pos["tp"]: return pos["tp"]
    pos["hi"] = max(pos["hi"], high)
    pos["stop"] = max(pos["stop"], pos["hi"] - pos["stop_atr"] * pos["atr"])  # trailing stop
    if pos["bars"] >= pos["max_hold"]: return close
    return None

def new_pos(entry, a, p):
    return dict(entry=entry, atr=a, stop=entry - p["stop_atr"] * a, tp=entry + p["tp_atr"] * a,
                hi=entry, bars=0, stop_atr=p["stop_atr"], max_hold=p["max_hold"])

def net_return(entry, exit_px):
    return exit_px * (1 - CFG["slip"]) / entry * (1 - CFG["fee"]) ** 2 - 1

# ---------------- BACKTEST ----------------
def backtest(df, sig, a, p):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    sig = sig.fillna(False).values; av = a.values
    rets, i, n = [], 0, len(df)
    while i < n - 1:
        if sig[i] and av[i] > 0:
            ent = o[i + 1] * (1 + CFG["slip"])           # signal on close of bar i, fill at next open
            pos, j, ex = new_pos(ent, av[i], p), i + 1, None
            while j < n:
                ex = check_exit(pos, h[j], l[j], c[j])
                if ex is not None: break
                j += 1
            if ex is None: ex, j = c[n - 1], n - 1
            rets.append(net_return(ent, ex)); i = j + 1
        else:
            i += 1
    return rets

def optimize(dfs, log=print):
    """Pick params per strategy on first 70% of data; accept ONLY if still profitable on last 30%."""
    split = {k: (d.iloc[:int(len(d) * .7)].reset_index(drop=True),
                 d.iloc[int(len(d) * .7):].reset_index(drop=True)) for k, d in dfs.items()}
    atrs = {k: (atr(tr), atr(te)) for k, (tr, te) in split.items()}
    result = {}
    for name, (fn, grid) in STRATS.items():
        best = None
        for sp in grid:
            sigs = {k: (fn(tr, sp), fn(te, sp)) for k, (tr, te) in split.items()}
            for ep in EXIT_GRID:
                p = {**sp, **ep}
                tr_r = [r for k in split for r in backtest(split[k][0], sigs[k][0], atrs[k][0], p)]
                if len(tr_r) < CFG["min_train_trades"]: continue
                score = np.mean(tr_r) * np.sqrt(len(tr_r))
                if best is None or score > best[0]: best = (score, p, np.mean(tr_r), len(tr_r), sigs)
        if not best:
            log(f"  {name}: not enough trades to evaluate"); continue
        _, p, tr_exp, tr_n, sigs = best
        te_r = [r for k in split for r in backtest(split[k][1], sigs[k][1], atrs[k][1], p)]
        te_exp = float(np.mean(te_r)) if te_r else 0.0
        ok = len(te_r) >= CFG["min_test_trades"] and te_exp >= CFG["min_test_expectancy"]
        log(f"  {name}: train {tr_exp:+.2%} ({tr_n} trades) | test {te_exp:+.2%} ({len(te_r)}) -> "
            f"{'ENABLED' if ok else 'rejected (no out-of-sample edge)'}")
        if ok: result[name] = p
    return result

# ---------------- STATE ----------------
def load():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return dict(cash=CFG["start_equity"], peak=CFG["start_equity"], positions={}, params={},
                stats={k: dict(n=0, sum=0.0) for k in STRATS}, cycle=0, halted=False)

def save(st): json.dump(st, open(STATE_FILE, "w"), indent=1)

def log_trade(row):
    pd.DataFrame([row]).to_csv(TRADES_FILE, mode="a", header=not os.path.exists(TRADES_FILE), index=False)

def weight(st, name):
    """Learning: shrunk live expectancy -> capital weight. Losers get cut to a floor, winners up to 2x."""
    s = st["stats"][name]
    exp = s["sum"] / (s["n"] + 10)
    return float(np.clip(1 + exp * 50, 0.25, 2.0))

def equity(st, last):
    return st["cash"] + sum(p["qty"] * last.get(k, p["entry"]) for k, p in st["positions"].items())

# ---------------- LIVE (PAPER) CYCLE ----------------
def cycle(st):
    st["cycle"] += 1
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"\n[{ts}] cycle {st['cycle']}: scanning all USD pairs...")
    pairs = scan()
    held = list(st["positions"])
    data = {}
    for pid in dict.fromkeys(pairs + held):
        df = history(pid, bars=300)
        if df is not None and len(df) > 150: data[pid] = df
    last = {k: d.close.iloc[-1] for k, d in data.items()}

    # 1) manage open positions on newly completed bars
    for pid in list(st["positions"]):
        pos = st["positions"][pid]
        if pid not in data: continue
        done = data[pid].iloc[:-1]
        ex = None
        for _, b in done[done.t > pos["last_t"]].iterrows():
            ex = check_exit(pos, b.high, b.low, b.close); pos["last_t"] = int(b.t)
            if ex is not None: break
        if ex is not None:
            ret = net_return(pos["entry"], ex)
            st["cash"] += pos["qty"] * ex * (1 - CFG["slip"]) * (1 - CFG["fee"])
            s = st["stats"][pos["strategy"]]; s["n"] += 1; s["sum"] += ret
            fees = pos["qty"] * pos["entry"] * CFG["fee"] + pos["qty"] * ex * CFG["fee"]
            pnl = pos["qty"] * ex * (1 - CFG["slip"]) * (1 - CFG["fee"]) - pos["qty"] * pos["entry"] * (1 + CFG["fee"])
            log_trade(dict(time=ts, pair=pid, strategy=pos["strategy"], entry=pos["entry"], exit=ex,
                           qty=pos["qty"], ret=ret, pnl=pnl, fees=fees))
            print(f"  EXIT {pid} [{pos['strategy']}] {ret:+.2%}")
            del st["positions"][pid]

    # 2) periodically re-optimize (and disable strategies that fail live)
    if st["cycle"] % CFG["reopt_every"] == 1 or not st["params"]:
        print("  optimizing strategies on recent history...")
        hist = {pid: h for pid in pairs[:25] if (h := history(pid, 900)) is not None and len(h) > 400}
        if hist: st["params"] = optimize(hist)
    for name, s in st["stats"].items():
        if s["n"] >= 20 and s["sum"] / (s["n"] + 10) < 0 and name in st["params"]:
            print(f"  live results negative -> disabling {name}"); st["params"].pop(name)

    # 3) risk checks
    eq = equity(st, last); st["peak"] = max(st["peak"], eq)
    dd = 1 - eq / st["peak"]
    if dd > CFG["max_drawdown"] and not st["halted"]:
        st["halted"] = True; print(f"  !! KILL SWITCH: drawdown {dd:.1%}. No new entries.")

    # 4) look for entries on the last COMPLETED bar
    if not st["halted"]:
        for name, p in st["params"].items():
            fn = STRATS[name][0]
            for pid, df in data.items():
                if pid in st["positions"] or len(st["positions"]) >= CFG["max_positions"]: continue
                done = df.iloc[:-1].reset_index(drop=True)
                if not bool(fn(done, p).fillna(False).iloc[-1]): continue
                a = atr(done).iloc[-1]
                entry = last[pid] * (1 + CFG["slip"])
                pos = new_pos(entry, a, p)
                risk_amt = eq * CFG["risk_per_trade"] * weight(st, name)
                qty = risk_amt / (entry - pos["stop"])
                qty = min(qty, eq * CFG["max_pos_pct"] / entry, st["cash"] / (entry * (1 + CFG["fee"])))
                if qty * entry < 10: continue
                st["cash"] -= qty * entry * (1 + CFG["fee"])
                pos.update(qty=qty, strategy=name, last_t=int(done.t.iloc[-1]))
                st["positions"][pid] = pos
                print(f"  ENTER {pid} [{name}] ${qty * entry:,.0f} stop {pos['stop']:.4g}")

    eq = equity(st, last)
    print(f"  equity ${eq:,.2f} | cash ${st['cash']:,.2f} | open {len(st['positions'])} | "
          f"drawdown {1 - eq / st['peak']:.1%} | enabled: {list(st['params']) or 'none'}")
    record(st, last, eq, ts)
    save(st)


def _js(o):
    return o.item() if hasattr(o, "item") else str(o)

def record(st, last, eq, ts):
    """Append equity point and write data.json for the dashboard."""
    pd.DataFrame([dict(time=ts, equity=round(eq, 2))]).to_csv(
        EQUITY_FILE, mode="a", header=not os.path.exists(EQUITY_FILE), index=False)
    trades = pd.read_csv(TRADES_FILE).to_dict("records") if os.path.exists(TRADES_FILE) else []
    curve = pd.read_csv(EQUITY_FILE).tail(3000).to_dict("records")
    positions = []
    for k, p in st["positions"].items():
        px = last.get(k, p["entry"])
        positions.append(dict(pair=k, strategy=p["strategy"], entry=p["entry"], last=px, qty=p["qty"],
                              value=p["qty"] * px, pnl_pct=px / p["entry"] - 1,
                              stop=p["stop"], target=p["tp"], bars=p["bars"]))
    strategies = []
    for k, s_ in st["stats"].items():
        strategies.append(dict(name=k, enabled=k in st["params"], n=s_["n"],
                               avg=(s_["sum"] / s_["n"]) if s_["n"] else 0, weight=weight(st, k)))
    out = dict(updated=ts, equity=eq, start_equity=CFG["start_equity"], cash=st["cash"], peak=st["peak"],
               halted=st["halted"], cycle=st["cycle"], positions=positions, strategies=strategies,
               trades=trades[-500:], curve=curve)
    json.dump(out, open(DATA_FILE, "w"), default=_js)

def report():
    st = load()
    print(f"cash ${st['cash']:,.2f} | open positions {len(st['positions'])} | halted {st['halted']}")
    for k, s in st["stats"].items():
        e = s["sum"] / s["n"] if s["n"] else 0
        print(f"  {k}: {s['n']} trades, avg {e:+.2%}, weight {weight(st, k):.2f}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "optimize":
        pairs = scan()[:25]
        hist = {p: h for p in pairs if (h := history(p, 900)) is not None and len(h) > 400}
        print(f"Testing on {len(hist)} pairs:"); optimize(hist)
    elif mode == "report":
        report()
    elif mode == "once":      # one cycle then exit (used by GitHub Actions)
        cycle(load())
    else:
        st = load()
        while True:
            try: cycle(st)
            except Exception as e: print("  cycle error:", e)
            time.sleep(3600 - time.time() % 3600 + 30)   # wake 30s after each hour closes

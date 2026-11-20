#!/usr/bin/env python3
"""
Coinbase spot PAPER trading TOURNAMENT. Real market data, fake money, no API keys, no real orders.
Strategies trade from one paper wallet. Losers get paused, winners get more capital.
  python bot.py once     # one cycle (used by GitHub Actions every hour)
  python bot.py report   # print the leaderboard
"""
import sys, json, time, os, random
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests

CFG = dict(
    start_equity=1000.0,
    fee=0.006,            # per side. Verify your Coinbase fee tier.
    slip=0.001,           # per side slippage assumption
    risk_per_trade=0.02,
    max_pos_pct=0.10,     # max 10% of the wallet in one coin
    max_deploy=0.85,      # FIX: never have more than 90% of equity invested; keep a cash reserve
    max_positions=12, per_strategy_max=2,
    max_drawdown=0.50,
    top_pairs=80, min_usd_vol=150_000,
    pause_after=25, pause_cycles=48,
)
STATE_FILE, TRADES_FILE, EQUITY_FILE, DATA_FILE = "state.json", "trades.csv", "equity.csv", "data.json"
API = "https://api.exchange.coinbase.com"
COLS = ["t", "low", "high", "open", "close", "volume"]
TRADE_COLS = ["time", "pair", "strategy", "entry", "exit", "qty", "ret", "pnl", "fees"]

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

def history(pid, bars=300, gran=3600):
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

# ---------------- INDICATORS ----------------
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

def vol_x(df, m): return df.volume > m * df.volume.rolling(24).mean()

# ---------------- STRATEGIES ----------------
def sig_breakout(df, p):
    return (df.close > df.high.rolling(p["n"]).max().shift(1)) & vol_x(df, 1.5) & (df.close > ema(df.close, 100))
def sig_meanrev(df, p):
    m, s = df.close.rolling(20).mean(), df.close.rolling(20).std()
    return (rsi(df.close) < p["rsi"]) & (df.close < m - 2 * s)
def sig_trend(df, p):
    f, sl = ema(df.close, p["fast"]), ema(df.close, p["slow"])
    return (f > sl) & (f.shift() <= sl.shift()) & (rsi(df.close) > 50)
def sig_spike(df, p):
    ret = df.close / df.close.shift(p["n"]) - 1
    return (ret > p["thr"]) & vol_x(df, p["vm"]) & (rsi(df.close) < 80) & (df.close > ema(df.close, 50))
def sig_knife(df, p):
    return (df.close / df.close.shift(24) - 1 < -p["drop"]) & (rsi(df.close) < 25)
def sig_scalp(df, p):
    return (df.close / df.close.shift(1) - 1 > p["thr"]) & vol_x(df, p["vm"])
def sig_moon(df, p):
    return (df.close / df.close.shift(24) - 1 > p["thr"]) & vol_x(df, p["vm"])

ATR_EXIT = dict(stop_atr=2, tp_atr=6, max_hold=72)
def pct_exit(stop, tp, hold): return dict(stop_pct=stop, tp_pct=tp, max_hold=hold)

STRATS = {
    "breakout": dict(kind="sig", fn=sig_breakout, p=dict(n=48, **ATR_EXIT)),
    "meanrev":  dict(kind="sig", fn=sig_meanrev,  p=dict(rsi=30, **ATR_EXIT)),
    "trend":    dict(kind="sig", fn=sig_trend,    p=dict(fast=12, slow=48, **ATR_EXIT)),
    "spike":    dict(kind="sig", fn=sig_spike,    p=dict(n=6, thr=0.03, vm=2.0, **ATR_EXIT)),
    "knife":    dict(kind="sig", fn=sig_knife,    p=dict(drop=0.08, **pct_exit(.06, .08, 48))),
    # FIX: scalp stop/target now clear the ~1.4% round-trip cost, so the strategy is not structurally losing
    "scalp":    dict(kind="sig", fn=sig_scalp,    p=dict(thr=0.012, vm=2.0, **pct_exit(.02, .05, 6))),
    "moonshot": dict(kind="sig", fn=sig_moon,     p=dict(thr=0.10, vm=3.0, **pct_exit(.08, .50, 72))),
    "gainers":  dict(kind="rank",                 p=dict(k=2, min=0.03, **pct_exit(.05, .15, 48))),
    "random":   dict(kind="random",               p=dict(**pct_exit(.03, .06, 24))),
}

def candidates(spec, done, p=None):
    p, kind = (p or spec["p"]), spec["kind"]
    if kind == "sig":
        return [pid for pid, df in done.items() if len(df) > 120 and bool(spec["fn"](df, p).fillna(False).iloc[-1])]
    if kind == "rank":
        r = {pid: df.close.iloc[-1] / df.close.iloc[-25] - 1 for pid, df in done.items() if len(df) > 30}
        top = sorted(r, key=r.get, reverse=True)[:p["k"]]
        return [x for x in top if r[x] >= p["min"]]
    return [random.choice(list(done))] if done else []

# ---------------- EXIT LOGIC ----------------
def check_exit(pos, high, low, close):
    pos["bars"] += 1
    if low <= pos["stop"]: return pos["stop"]
    if high >= pos["tp"]: return pos["tp"]
    pos["hi"] = max(pos["hi"], high)
    pos["stop"] = max(pos["stop"], pos["hi"] - pos["stop_atr"] * pos["atr"])
    if pos["bars"] >= pos["max_hold"]: return close
    return None

def new_pos(entry, a, p):
    if "stop_pct" in p:
        d = entry * p["stop_pct"]
        return dict(entry=entry, atr=d, stop=entry - d, tp=entry * (1 + p["tp_pct"]), hi=entry, bars=0, stop_atr=1, max_hold=p["max_hold"])
    return dict(entry=entry, atr=a, stop=entry - p["stop_atr"] * a, tp=entry + p["tp_atr"] * a, hi=entry, bars=0,
                stop_atr=p["stop_atr"], max_hold=p["max_hold"])

def net_return(entry, exit_px):
    return exit_px * (1 - CFG["slip"]) / entry * (1 - CFG["fee"]) ** 2 - 1

# ---------------- OPTIMIZATION LAB ----------------
def _atr_exits(): return [dict(stop_atr=a, tp_atr=b, max_hold=h) for a in (2, 3) for b in (4, 8) for h in (48, 96)]
def _pct(stops, tps, holds): return [pct_exit(s_, t_, h_) for s_ in stops for t_ in tps for h_ in holds]
LAB = {
    "breakout": ([dict(n=n) for n in (24, 48, 96)], _atr_exits()),
    "meanrev":  ([dict(rsi=r) for r in (20, 25, 30, 35)], _atr_exits()),
    "trend":    ([dict(fast=f, slow=sl) for f, sl in ((8, 32), (12, 48), (24, 96))], _atr_exits()),
    "spike":    ([dict(n=n, thr=t, vm=2.0) for n in (3, 6, 12) for t in (.02, .03, .05)], _atr_exits()),
    "knife":    ([dict(drop=d) for d in (.05, .08, .12)], _pct((.04, .06, .10), (.06, .10, .15), (24, 48))),
    "scalp":    ([dict(thr=t, vm=2.0) for t in (.008, .012, .02)], _pct((.02, .03), (.04, .05, .08), (4, 8))),
    "moonshot": ([dict(thr=t, vm=3.0) for t in (.06, .10, .15)], _pct((.05, .08, .12), (.5,), (48, 96))),
}

def backtest(df, sig, a, p):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    sg = sig.fillna(False).values; av = None if a is None else a.values
    out, i, n = [], 0, len(df)
    while i < n - 1:
        if sg[i] and (av is None or av[i] > 0):
            ent = o[i + 1] * (1 + CFG["slip"])
            pos = new_pos(ent, None if av is None else av[i], p)
            j, ex = i + 1, None
            while j < n:
                ex = check_exit(pos, h[j], l[j], c[j])
                if ex is not None: break
                j += 1
            if ex is None: ex, j = c[n - 1], n - 1
            out.append((i / n, net_return(ent, ex))); i = j + 1
        else:
            i += 1
    return out

def run_lab(pairs, log=print):
    dfs = {}
    for pid in pairs:
        h = history(pid, 900)
        if h is not None and len(h) > 400: dfs[pid] = h
    atrs = {k: atr(d) for k, d in dfs.items()}
    results, tested, notes = [], 0, []
    for name, (sgrid, egrid) in LAB.items():
        try:
            fn, best, max_n = STRATS[name]["fn"], None, 0
            for sp in sgrid:
                sigs = {k: fn(d, sp) for k, d in dfs.items()}
                for ep in egrid:
                    tested += 1
                    p = {**sp, **ep}; use_atr = "stop_pct" not in ep
                    allr = []
                    for k, d in dfs.items(): allr += backtest(d, sigs[k], atrs[k] if use_atr else None, p)
                    n = len(allr); max_n = max(max_n, n)
                    if n < 10: continue
                    rets = [r for _, r in allr]; recent = [r for f, r in allr if f >= 0.7]
                    # FIX: a setting must also win on the recent 30% of data, or it is treated as overfit and ignored
                    if len(recent) < 5 or float(np.mean(recent)) <= 0: continue
                    avg = float(np.mean(rets)); score = avg * np.sqrt(n)
                    if best is None or score > best[0]:
                        best = (score, dict(name=name, params=p, n=n, avg=avg, win=float(np.mean([r > 0 for r in rets])),
                                            test_n=len(recent), test_avg=float(np.mean(recent))))
            if best: results.append(best[1]); log(f"  lab {name}: best avg {best[1]['avg']:+.2%} over {best[1]['n']} trades")
            else: notes.append(f"{name}: no setting won on both history and recent data (max {max_n} trades)")
        except Exception as e:
            notes.append(f"{name}: error {type(e).__name__}: {e}"[:120]); log(f"  lab {name} error: {e}")
    return dict(updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), pairs=len(dfs), tested=tested,
                results=results, notes=notes[:9])

# ---------------- STATE ----------------
def blank_stats(): return dict(n=0, sum=0.0, wins=0, pnl=0.0)

def load():
    if os.path.exists(STATE_FILE):
        st = json.load(open(STATE_FILE))
        st.setdefault("stats", {}); st.setdefault("disabled", {})
        for k in STRATS:
            s = st["stats"].setdefault(k, blank_stats())
            for f, v in blank_stats().items(): s.setdefault(f, v)
        return st
    return dict(cash=CFG["start_equity"], peak=CFG["start_equity"], positions={}, disabled={},
                stats={k: blank_stats() for k in STRATS}, cycle=0, halted=False)

def save(st): json.dump(st, open(STATE_FILE, "w"), indent=1)

def log_trade(row):
    pd.DataFrame([row]).to_csv(TRADES_FILE, mode="a", header=not os.path.exists(TRADES_FILE), index=False)

def weight(st, name):
    if name == "random": return 1.0
    s = st["stats"][name]
    return float(np.clip(1 + (s["sum"] / (s["n"] + 10)) * 50, 0.25, 2.0))

def equity(st, last):
    return st["cash"] + sum(p["qty"] * last.get(k, p["entry"]) for k, p in st["positions"].items())

# ---------------- ONE CYCLE ----------------
def cycle(st):
    st["cycle"] += 1
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"\n[{ts}] cycle {st['cycle']}: scanning all USD pairs...")
    pairs = scan()
    if st["cycle"] - st.get("lab_cycle", -999) >= 24:
        print("  running optimization lab on real history...")
        try:
            lab = run_lab(pairs[:30])
            if lab["results"]:
                st["tuned"] = {r["name"]: r["params"] for r in lab["results"]}
                st["lab_cycle"] = st["cycle"]
            json.dump(lab, open("lab.json", "w"), default=_js)
        except Exception as e:
            print("  lab error:", e)
            if not os.path.exists("lab.json"):
                json.dump(dict(updated=ts, pairs=0, tested=0, results=[], error=f"{type(e).__name__}: {e}"[:300]), open("lab.json", "w"))
    data = {}
    for pid in dict.fromkeys(pairs + list(st["positions"])):
        df = history(pid, bars=300)
        if df is not None and len(df) > 60: data[pid] = df
    last = {k: d.close.iloc[-1] for k, d in data.items()}
    done = {k: d.iloc[:-1].reset_index(drop=True) for k, d in data.items()}

    # 1) exits
    for pid in list(st["positions"]):
        pos = st["positions"][pid]
        if pid not in done: continue
        d, ex = done[pid], None
        for _, b in d[d.t > pos["last_t"]].iterrows():
            ex = check_exit(pos, b.high, b.low, b.close); pos["last_t"] = int(b.t)
            if ex is not None: break
        if ex is not None:
            ret = net_return(pos["entry"], ex)
            fees = pos["qty"] * pos["entry"] * CFG["fee"] + pos["qty"] * ex * CFG["fee"]
            pnl = pos["qty"] * ex * (1 - CFG["slip"]) * (1 - CFG["fee"]) - pos["qty"] * pos["entry"] * (1 + CFG["fee"])
            st["cash"] += pos["qty"] * ex * (1 - CFG["slip"]) * (1 - CFG["fee"])
            s = st["stats"][pos["strategy"]]; s["n"] += 1; s["sum"] += ret; s["pnl"] += pnl; s["wins"] += int(ret > 0)
            log_trade(dict(time=ts, pair=pid, strategy=pos["strategy"], entry=pos["entry"], exit=ex,
                           qty=pos["qty"], ret=ret, pnl=pnl, fees=fees))
            print(f"  EXIT  {pid} [{pos['strategy']}] {ret:+.2%}")
            del st["positions"][pid]

    # 2) learn: pause losers, retry later
    for name, c in list(st["disabled"].items()):
        if st["cycle"] - c >= CFG["pause_cycles"]:
            del st["disabled"][name]; st["stats"][name] = blank_stats(); print(f"  retrying {name}")
    for name, s in st["stats"].items():
        if name != "random" and name not in st["disabled"] and s["n"] >= CFG["pause_after"] and s["sum"] / (s["n"] + 10) < 0:
            st["disabled"][name] = st["cycle"]; print(f"  net-negative after {s['n']} trades -> pausing {name}")

    # 3) risk check
    if st["cash"] < -1e-6:
        st["halted"] = True; print("  !! WALLET ERROR: cash went negative. Entries halted.")
    eq = equity(st, last); st["peak"] = max(st["peak"], eq)
    if 1 - eq / st["peak"] > CFG["max_drawdown"] and not st["halted"]:
        st["halted"] = True; print("  !! KILL SWITCH: big drawdown. No new entries.")

    # 4) entries
    if not st["halted"]:
        for name, spec in STRATS.items():
            if name in st["disabled"]: continue
            p = st.get("tuned", {}).get(name, spec["p"])
            mine = sum(1 for q in st["positions"].values() if q["strategy"] == name)
            try: cands = candidates(spec, done, p)
            except Exception as e:
                print(f"  {name} error: {e}"); continue
            for pid in cands:
                if mine >= CFG["per_strategy_max"] or len(st["positions"]) >= CFG["max_positions"]: break
                if pid in st["positions"] or pid not in last: continue
                a = None if "stop_pct" in p else atr(done[pid]).iloc[-1]
                if a is not None and not a > 0: continue
                entry = last[pid] * (1 + CFG["slip"])
                pos = new_pos(entry, a, p)
                qty = eq * CFG["risk_per_trade"] * weight(st, name) / (entry - pos["stop"])
                # FIX: total invested can never exceed max_deploy of equity; cash can never go negative
                invested = sum(q["qty"] * last.get(k, q["entry"]) for k, q in st["positions"].items())
                room = max(0.0, eq * CFG["max_deploy"] - invested)
                qty = min(qty, eq * CFG["max_pos_pct"] / entry, room / entry,
                          st["cash"] / (entry * (1 + CFG["fee"])))
                if qty * entry < 10: continue
                st["cash"] -= qty * entry * (1 + CFG["fee"])
                pos.update(qty=qty, strategy=name, last_t=int(done[pid].t.iloc[-1]))
                st["positions"][pid] = pos; mine += 1
                print(f"  ENTER {pid} [{name}] ${qty * entry:,.0f}")

    eq = equity(st, last)
    print(f"  equity ${eq:,.2f} | cash ${st['cash']:,.2f} | open {len(st['positions'])} | paused: {list(st['disabled']) or 'none'}")
    record(st, last, eq, ts)
    save(st)

# ---------------- DASHBOARD OUTPUT ----------------
def _js(o): return o.item() if hasattr(o, "item") else str(o)

def record(st, last, eq, ts):
    if not os.path.exists(TRADES_FILE):
        pd.DataFrame(columns=TRADE_COLS).to_csv(TRADES_FILE, index=False)
    pd.DataFrame([dict(time=ts, equity=round(eq, 2))]).to_csv(
        EQUITY_FILE, mode="a", header=not os.path.exists(EQUITY_FILE), index=False)
    trades = pd.read_csv(TRADES_FILE).to_dict("records")
    curve = pd.read_csv(EQUITY_FILE).tail(3000).to_dict("records")
    positions = []
    for k, p in st["positions"].items():
        px = last.get(k, p["entry"])
        positions.append(dict(pair=k, strategy=p["strategy"], entry=p["entry"], last=px, qty=p["qty"],
                              value=p["qty"] * px, pnl_pct=px / p["entry"] - 1, stop=p["stop"], target=p["tp"], bars=p["bars"]))
    strategies = []
    for k, s_ in st["stats"].items():
        strategies.append(dict(name=k, paused=k in st["disabled"], enabled=k not in st["disabled"], n=s_["n"], wins=s_["wins"],
                               pnl=s_["pnl"], avg=(s_["sum"] / s_["n"]) if s_["n"] else 0, weight=weight(st, k),
                               open=sum(1 for q in st["positions"].values() if q["strategy"] == k)))
    out = dict(updated=ts, equity=eq, start_equity=CFG["start_equity"], cash=st["cash"], peak=st["peak"], halted=st["halted"],
               cycle=st["cycle"], positions=positions, strategies=strategies, trades=trades[-500:], curve=curve,
               lab=(json.load(open("lab.json")) if os.path.exists("lab.json") else None))
    json.dump(out, open(DATA_FILE, "w"), default=_js)

def report():
    st = load()
    print(f"cash ${st['cash']:,.2f} | open {len(st['positions'])} | halted {st['halted']}")
    for k, s in sorted(st["stats"].items(), key=lambda kv: -kv[1]["pnl"]):
        print(f"  {k:9s} {s['n']:3d} trades | P&L ${s['pnl']:+.2f} | wins {s['wins']}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    if mode == "report": report()
    else: cycle(load())

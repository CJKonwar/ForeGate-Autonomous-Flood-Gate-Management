"""Shared helpers for the FYP notebooks (03 onwards).

Everything here follows the rules in CLAUDE.md:
  * rain is always `precip_local` (rain moved to the local clock in notebook 01);
  * every threshold / lag / eps used here is estimated on the TRAINING years only;
  * features use trailing windows only (no centred windows, no future flow);
  * future rain appears ONLY as explicit scenario features (A = perfect, B = degraded, C = none);
  * the test year is never read for its flow values here.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import lfilter

N = 61_368                 # hourly rows per station
SEED = 42
H_REPORT = [1, 3, 6, 12, 24, 48, 72, 120]                          # lead times we report
H_RIDGE = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72, 96, 120]     # lead times ridge is fitted at
H_LGBM = [1, 6, 12, 24, 48, 72, 120]                                # lead times LightGBM is fitted at
FUT_BINS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 72), (73, 120)]   # future-rain bins, hours after origin
RAIN_WINDOWS = [3, 6, 12, 24, 48, 72, 168]
FUT_SIGMA = 0.7            # scenario B: lognormal magnitude error on every future-rain bin


# --------------------------------------------------------------------------------------------- loading
def load_grid(root="."):
    """Load the cleaned data as 125 x 61,368 arrays plus all the things notebook 02 saved."""
    root = Path(root); proc = root / "processed"
    df = pd.read_parquet(proc / "waterbench_ts_clean.parquet",
                         columns=["station", "datetime", "split", "precip_local", "q_cms", "flag_spike"])
    g = SimpleNamespace()
    g.stations = list(df["station"].astype(str).iloc[::N]); S = len(g.stations)
    assert len(df) == S * N
    assert (df["station"].astype(str).to_numpy().reshape(S, N) == np.array(g.stations)[:, None]).all()
    g.idx = {s: i for i, s in enumerate(g.stations)}
    g.time = pd.DatetimeIndex(df["datetime"].iloc[:N])
    g.split = df["split"].astype(str).iloc[:N].to_numpy()
    q = df["q_cms"].to_numpy().reshape(S, N)
    g.Qm = np.where(df["flag_spike"].to_numpy().reshape(S, N), np.nan, q)       # flow m3/s, 15 flagged spike hours blanked
    g.P = df["precip_local"].to_numpy().reshape(S, N)                           # rain on the LOCAL clock
    g.ntr = int((g.split == "train").sum())
    g.static = pd.read_csv(proc / "static_attributes.csv", dtype={"station": str}).set_index("station").loc[g.stations]
    groups = pd.read_csv(proc / "station_groups.csv", dtype={"station": str}).set_index("station").loc[g.stations]
    g.size_group = groups["size_group"]
    g.cluster = groups["cluster"]
    thr = pd.read_csv(proc / "flood_thresholds_train.csv", dtype={"station": str}).set_index("station").loc[g.stations]
    g.q95 = thr["q95"].to_numpy(); g.q99 = thr["q99"].to_numpy()
    # eps for log(Q + eps): 1% of the station's TRAINING-year mean flow
    g.eps = 0.01 * np.nanmean(g.Qm[:, :g.ntr], axis=1)
    g.L = np.log(g.Qm + g.eps[:, None])
    # shared split spec: every sample lies entirely inside one split (window L+H)
    g.spec = json.loads((proc / "splits.json").read_text())
    g.origins = {}
    for s, d in g.spec["splits"].items():
        a, b = g.time.get_loc(pd.Timestamp(d["first_origin"])), g.time.get_loc(pd.Timestamp(d["last_origin"]))
        g.origins[s] = np.arange(a, b + 1)
    # upstream gauges and their peak-method travel lags (estimated on training years in notebook 02)
    edges = pd.read_csv(proc / "upstream_lags.csv", dtype={"upstream": str, "downstream": str})
    g.up = {}
    for d, e in edges.groupby("downstream"):
        lag = e.loc[e["n_events"] >= 5, "event_lag_h"].median()
        g.up[d] = dict(ups=[g.idx[u] for u in e["upstream"]], lag=float(lag) if np.isfinite(lag) else 12.0)
    return g


# --------------------------------------------------------------------------------------------- features
def shift(x, k):
    """x[t-k] aligned to t (NaN where it does not exist). Negative k looks ahead."""
    out = np.full(x.shape, np.nan, dtype=float)
    if k == 0:
        return x.astype(float).copy()
    if k > 0:
        out[k:] = x[:len(x) - k]
    else:
        out[:k] = x[-k:]
    return out

def _rollmean(x, w):
    return pd.Series(x).rolling(w, min_periods=int(0.8 * w)).mean().to_numpy()

def _rollmax(x, w):
    return pd.Series(x).rolling(w, min_periods=int(0.8 * w)).max().to_numpy()

def future_bins(g, i, scenario):
    """Rain accumulated over hours (a..b] after the origin. 'A' = observed (an ORACLE, not a forecast),
    'B' = observed x lognormal magnitude error, 'C' = not provided."""
    if scenario == "C":
        return {}
    P = np.nan_to_num(g.P[i]); cs = np.cumsum(P)
    out = {}
    for a, b in FUT_BINS:
        out[f"Pfut_{a}_{b}"] = shift(cs, -b) - shift(cs, -(a - 1))
    if scenario == "B":
        rng = np.random.RandomState(SEED + i)
        fac = np.exp(rng.normal(-0.5 * FUT_SIGMA ** 2, FUT_SIGMA, size=(len(P), len(FUT_BINS))))
        for j, k in enumerate(out):
            out[k] = out[k] * fac[:, j]
    return out

def station_features(g, i, static=True):
    """Trailing-window features for station i at every hour t (as of the end of hour t). Returns dict name -> array(N)."""
    L = g.L[i]; P = np.nan_to_num(g.P[i]); f = {}
    f["L0"] = L
    for k in (1, 3, 6, 24):
        f[f"dL{k}"] = L - shift(L, k)
    f["accel"] = (L - shift(L, 1)) - (shift(L, 1) - shift(L, 2))
    for w in (24, 72, 168):
        f[f"Lmean{w}"] = _rollmean(L, w)
    f["Lmax24"] = _rollmax(L, 24)
    f["L_vs_mean168"] = L - f["Lmean168"]
    cs = np.cumsum(P)
    for w in RAIN_WINDOWS:
        sh = shift(cs, w)
        f[f"P{w}h"] = np.where(np.isnan(sh), cs, cs - sh)
    f["P0"] = P; f["dP"] = P - shift(P, 1)
    f["api_fast"] = lfilter([1.0], [1.0, -0.85 ** (1 / 24)], P)     # decay 0.85 per day
    f["api_slow"] = lfilter([1.0], [1.0, -0.95 ** (1 / 24)], P)     # decay 0.95 per day
    hr, doy = g.time.hour.to_numpy(), g.time.dayofyear.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24); f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # upstream gauges (direct upstream only): summed flow, several lags around the measured travel time
    s = g.stations[i]; info = g.up.get(s)
    f["has_up"] = np.full(N, 1.0 if info else 0.0)
    ks = [0, 3, 6, 12, 24]
    if info:
        qs = g.Qm[info["ups"]]
        qu = np.where(np.isnan(qs).any(axis=0), np.nan, qs.sum(axis=0))
        eps_u = 0.01 * np.nanmean(qu[:g.ntr])
        Lu = np.log(qu + eps_u)
        ks = sorted({int(round(info["lag"] * m)) for m in (0, 0.5, 1, 1.5, 2)})
        while len(ks) < 5:
            ks.append(ks[-1] + 1)
        ks = ks[:5]
        for j, k in enumerate(ks):
            f[f"Lup_lag{j}"] = shift(Lu, k)
        f["dLup6"] = Lu - shift(Lu, 6)
        f["Lup_minus_L"] = Lu - L
    else:
        for j in range(5):
            f[f"Lup_lag{j}"] = np.full(N, np.nan)
        f["dLup6"] = np.full(N, np.nan); f["Lup_minus_L"] = np.full(N, np.nan)
    if static:                                   # constant per station: only useful for regional models
        st = g.static.iloc[i]
        f["log_area"] = np.full(N, np.log10(st["area"])); f["log_tc"] = np.full(N, np.log10(st["travel_time"]))
        f["slope"] = np.full(N, st["slope"])
        for c in ("loam", "silt", "sandy_clay_loam", "silty_clay_loam"):
            f[c] = np.full(N, st[c])
    return f

def target_delta(g, i, h):
    """Change in log flow between the origin and h hours later (NaN if either is missing)."""
    return shift(g.L[i], -h) - g.L[i]


# --------------------------------------------------------------------------------------------- metrics
def nse(y, yh):
    d = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yh) ** 2).sum() / d if d > 0 else np.nan

def kge_parts(y, yh):
    if y.std() == 0 or yh.std() == 0 or y.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(y, yh)[0, 1]; a = yh.std() / y.std(); b = yh.mean() / y.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), r, a, b

def skill(y, yh, q95, min_n=300, min_ex=15):
    """All per-station metrics on matched arrays y (observed m3/s) and yh (forecast m3/s)."""
    n = len(y)
    out = dict(n=n, nse=np.nan, kge=np.nan, r=np.nan, alpha=np.nan, beta=np.nan, rmse=np.nan,
               pod=np.nan, far=np.nan, csi=np.nan, flood_bias=np.nan, n_flood=0)
    if n < min_n:
        return out
    out["nse"] = nse(y, yh)
    out["kge"], out["r"], out["alpha"], out["beta"] = kge_parts(y, yh)
    out["rmse"] = float(np.sqrt(np.mean((y - yh) ** 2)))
    oe, pe = y > q95, yh > q95
    out["n_flood"] = int(oe.sum())
    if oe.sum() >= min_ex:
        hit, miss, fa = (oe & pe).sum(), (oe & ~pe).sum(), (~oe & pe).sum()
        out["pod"] = hit / (hit + miss)
        out["far"] = fa / (hit + fa) if (hit + fa) > 0 else np.nan
        out["csi"] = hit / (hit + miss + fa)
        out["flood_bias"] = float(np.median((yh[oe] - y[oe]) / y[oe]))     # < 0 means the forecast under-predicts floods
    return out

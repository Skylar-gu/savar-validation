"""
T0 — spatial de-aliasing identifiability floor (clean 1-D Fourier testbed, no GNN).

Go/no-go gate for the moving-mechanism sub-resolution battery
(notes/moving_mechanism_subres_spec.md; machinery from
notes/subres_spatial_dealiasing_plan.md, Testbed alpha).

Setup
-----
Fine 1-D grid length Lf, coarse Lc = Lf/s. Wavenumbers k_low and
k_high = k_low + Lc COLLIDE under D_sub (decimation keeps every s-th pixel):
cos(2*pi*k_high*(s*n)/Lf) = cos(2*pi*k_low*n/Lc) exactly (same for sin when
k_high = k_low + Lc). Each amplitude a_k(t) is AR(1) with a DISTINCT phi within
the colliding pair, skew-normal innovations, scaled to UNIT stationary variance
(no amplitude cue). Fine field f(x,t) = sum_k a_k(t) cos(.) + b_k(t) sin(.),
observed through D_sub (aliasing regime, primary) or D_avg (block-average s,
low-pass/destructive control) plus per-pixel Gaussian noise.

Recovery target: a_high(t). Methods:
  * ORACLE  — Kalman filter + RTS smoother on the numerically-derived exact
    measurement model m(t) = <u, y_t> = c_low*a_low + c_high*a_high + v
    (u = the observed collided pattern, unit norm; c's from projecting D@basis
    onto u — handles D_avg attenuation automatically). Optimal linear method
    with known dynamics.
  * LINEAR  — supervised ridge from a (2*HW+1)-frame window of ALL coarse
    pixels -> a_high(t); fit on the first half, corr on the second half
    (non-oracle linear ceiling).
Baseline: corr(m(t), a_high(t)) — the raw mixture (what you get with NO
temporal de-aliasing; ~1/sqrt(2) at equal variance under D_sub).

Also: identical-twin pair (delta_phi = 0, T4 anchor) and a delta-phi
dose-response sweep (T5 anchor / identifiability curve).

Gate: PASS if under D_sub the oracle recovers a_high well above the mixture
baseline at large delta-phi while D_avg stays at/near its (attenuated)
baseline. If the oracle fails badly under D_sub, the de-aliasing thread is
dead regardless of movement (spec sequencing step 1).

Output: results/subres_t0_identifiability.npy; table to stdout.
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

rng_global = np.random.default_rng(0)

# ── config ────────────────────────────────────────────────────────────────────
LF        = 64
T         = 4000
SIGMA_Y   = np.sqrt(0.05)          # per-pixel obs noise (matches SAVAR DY_SCALE)
N_SEEDS   = 8
HW        = 12                     # ridge window half-width (frames)
NG_SKEW   = 4.0                    # skew-normal shape (matches generators)

# configs: (s, k_low, k_high) — k_high collides with k_low under decimation-by-s
#   s=4: k_high = k_low + Lc (=16); s=2: k_high = Lc - k_low (=32-k, sin flips —
#   handled numerically). s=2 mirrors the movmech generator's operator.
CONFIGS = [
    dict(s=4, k_low=3, k_high=19),
    dict(s=2, k_low=3, k_high=29),
]

PHI_LOW  = 0.90
PHI_HIGH_SWEEP = [0.90, 0.85, 0.80, 0.70, 0.60, 0.45, 0.30, 0.15]


def draw_skewnorm(rng, shape, a=NG_SKEW):
    delta = a / np.sqrt(1.0 + a * a)
    z0 = np.abs(rng.standard_normal(shape))
    z1 = rng.standard_normal(shape)
    x = delta * z0 + np.sqrt(1.0 - delta * delta) * z1
    mean = delta * np.sqrt(2.0 / np.pi)
    var  = 1.0 - 2.0 * delta * delta / np.pi
    return (x - mean) / np.sqrt(var)


def ar1(rng, T, phi):
    """AR(1), skew-normal innovations, unit stationary variance."""
    e = draw_skewnorm(rng, T) * np.sqrt(1.0 - phi * phi)
    x = np.empty(T)
    x[0] = draw_skewnorm(rng, 1)[0]
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def make_operators(Lf, s):
    x = np.arange(Lf)
    D_sub = np.zeros((Lf // s, Lf)); D_sub[np.arange(Lf // s), np.arange(0, Lf, s)] = 1.0
    D_avg = np.zeros((Lf // s, Lf))
    for i in range(Lf // s):
        D_avg[i, i * s:(i + 1) * s] = 1.0 / s
    return x, D_sub, D_avg


def kalman_rts(m, H, phis, R):
    """KF + RTS smoother; state = per-wavenumber amplitudes (unit stat var).
    m: (T,) measurement; H: (2,) row; phis: (2,); R: scalar noise var.
    Returns smoothed state means (T, 2)."""
    A = np.diag(phis)
    Q = np.diag(1.0 - np.asarray(phis) ** 2)
    n = len(phis)
    xf = np.zeros((len(m), n)); Pf = np.zeros((len(m), n, n))
    xp = np.zeros((len(m), n)); Pp = np.zeros((len(m), n, n))
    x = np.zeros(n); P = np.eye(n)
    for t in range(len(m)):
        x = A @ x; P = A @ P @ A.T + Q
        xp[t] = x; Pp[t] = P
        S = H @ P @ H + R
        Kg = P @ H / S
        x = x + Kg * (m[t] - H @ x)
        P = P - np.outer(Kg, H @ P)
        xf[t] = x; Pf[t] = P
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(len(m) - 2, -1, -1):
        J = Pf[t] @ A.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
    return xs


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def run_case(seed, s, k_low, k_high, phi_low, phi_high, operator):
    """Generate one realisation, recover a_high; return dict of corrs."""
    rng = np.random.default_rng(seed)
    x, D_sub, D_avg = make_operators(LF, s)
    D = D_sub if operator == "sub" else D_avg

    a_low  = ar1(rng, T, phi_low);  b_low  = ar1(rng, T, phi_low)
    a_high = ar1(rng, T, phi_high); b_high = ar1(rng, T, phi_high)

    basis = {  # fine-grid spatial patterns
        "cl": np.cos(2 * np.pi * k_low  * x / LF), "sl": np.sin(2 * np.pi * k_low  * x / LF),
        "ch": np.cos(2 * np.pi * k_high * x / LF), "sh": np.sin(2 * np.pi * k_high * x / LF),
    }
    f = (np.outer(a_low, basis["cl"]) + np.outer(b_low, basis["sl"])
         + np.outer(a_high, basis["ch"]) + np.outer(b_high, basis["sh"]))   # (T, Lf)
    y = f @ D.T + SIGMA_Y * rng.standard_normal((T, LF // s))               # (T, Lc)

    # observed (coarse) patterns of each basis fn under D
    obs = {k: D @ v for k, v in basis.items()}
    # measurement direction: the collided cos pattern (unit norm); numerically
    # derive the mixing coefficients c = <u, D@basis>. For the sin channel use
    # the collided sin pattern.
    out = {}
    for chan, (lo, hi, alo, ahi) in {
        "cos": ("cl", "ch", a_low, a_high),
        "sin": ("sl", "sh", b_low, b_high),
    }.items():
        u = obs[lo].copy()
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u /= nu
        c_lo = float(u @ obs[lo]); c_hi = float(u @ obs[hi])
        m = y @ u                                       # (T,)
        R = SIGMA_Y ** 2                                # u unit norm
        xs = kalman_rts(m, np.array([c_lo, c_hi]), [phi_low, phi_high], R)
        out[chan] = dict(
            oracle=corr(xs[:, 1], ahi),
            baseline=abs(corr(m, ahi)),   # |r|: sin channel can collide sign-flipped
            c_lo=c_lo, c_hi=c_hi,
        )

    # supervised linear (windowed ridge on all coarse pixels) — cos channel target
    Lc = LF // s
    half = T // 2
    idx = np.arange(HW, T - HW)
    Xw = np.stack([y[i - HW:i + HW + 1].ravel() for i in idx])   # (T-2HW, (2HW+1)*Lc)
    tw = a_high[idx]
    tr = idx < half; te = ~tr
    Xtr, Xte = Xw[tr], Xw[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    lam = 10.0
    Wr = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ tw[tr])
    ridge_corr = corr(Xte @ Wr, tw[te])

    mean_or = np.mean([v["oracle"] for v in out.values()])
    mean_bl = np.mean([v["baseline"] for v in out.values()])
    return dict(oracle=mean_or, baseline=mean_bl, ridge=ridge_corr,
                c_lo=out["cos"]["c_lo"], c_hi=out["cos"]["c_hi"])


def avg_seeds(**kw):
    rs = [run_case(seed=1000 + i, **kw) for i in range(N_SEEDS)]
    return {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
            for k in rs[0]}


results = {}
print(f"T0 — de-aliasing identifiability floor   Lf={LF} T={T} sigma_y={SIGMA_Y:.3f} "
      f"seeds={N_SEEDS}  phi_low={PHI_LOW}")

for cfg in CONFIGS:
    s, kl, kh = cfg["s"], cfg["k_low"], cfg["k_high"]
    key = f"s{s}_k{kl}-{kh}"
    print(f"\n== config {key}  (Lc={LF//s}, coarse Nyquist={LF//s//2}) ==")
    print(f"{'phi_high':>8} {'dphi':>6} | {'D_sub oracle':>13} {'base':>6} {'ridge':>6} "
          f"| {'D_avg oracle':>13} {'base':>6} {'ridge':>6} | {'c_hi(avg)':>9}")
    rows = []
    for ph in PHI_HIGH_SWEEP:
        r_sub = avg_seeds(s=s, k_low=kl, k_high=kh, phi_low=PHI_LOW, phi_high=ph, operator="sub")
        r_avg = avg_seeds(s=s, k_low=kl, k_high=kh, phi_low=PHI_LOW, phi_high=ph, operator="avg")
        rows.append(dict(phi_high=ph, sub=r_sub, avg=r_avg))
        print(f"{ph:>8.2f} {PHI_LOW-ph:>6.2f} | {r_sub['oracle'][0]:>6.3f}±{r_sub['oracle'][1]:<5.3f} "
              f"{r_sub['baseline'][0]:>6.3f} {r_sub['ridge'][0]:>6.3f} "
              f"| {r_avg['oracle'][0]:>6.3f}±{r_avg['oracle'][1]:<5.3f} "
              f"{r_avg['baseline'][0]:>6.3f} {r_avg['ridge'][0]:>6.3f} "
              f"| {r_avg['c_hi'][0]:>9.3f}")
    results[key] = rows

os.makedirs("results", exist_ok=True)
np.save("results/subres_t0_identifiability.npy",
        dict(results=results, config=dict(LF=LF, T=T, SIGMA_Y=SIGMA_Y, N_SEEDS=N_SEEDS,
                                          PHI_LOW=PHI_LOW, sweep=PHI_HIGH_SWEEP,
                                          configs=CONFIGS)),
        allow_pickle=True)
print("\nsaved -> results/subres_t0_identifiability.npy")

# gate verdict
best = max(r["sub"]["oracle"][0] for key in results for r in results[key]
           if r["phi_high"] <= 0.45)
base = np.mean([r["sub"]["baseline"][0] for key in results for r in results[key]
                if r["phi_high"] <= 0.45])
print(f"\nGATE: max D_sub oracle corr at dphi>=0.45 = {best:.3f} "
      f"(mixture baseline ~{base:.3f}) -> {'PASS' if best > 0.85 and best - base > 0.1 else 'CHECK/FAIL'}")

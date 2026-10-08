"""Core physical and statistical functions for the phase-space interferometry study."""
import numpy as np
import pandas as pd
from scipy.stats import beta
PI = np.pi

def wrap(x):
    return (x + PI) % (2 * PI) - PI

def phase_code(lams_nm, d_um):
    lams = np.asarray(lams_nm, dtype=float) * 1e-9
    d = np.asarray(d_um, dtype=float) * 1e-6
    return wrap(d[..., None] * (4 * PI / lams))

def alias_margin(lams_nm, D_um=100.0, min_delta_um=0.25, step_um=0.002):
    delta = np.arange(min_delta_um, D_um + step_um / 2, step_um)
    ph = phase_code(lams_nm, delta)
    norm = np.linalg.norm(ph, axis=1)
    i = int(np.argmin(norm))
    return float(norm[i]), float(delta[i]), delta, norm

def cp_ci(successes, n, alpha=0.05):
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lo), float(hi)

def simulate(lams_nm, D_um=100.0, N=10000, seed=0, sigma_phi_rad=0.015,
             sigma_lam_pm=0.5, coarse_step_um=0.05, success_tol_um=0.25):
    rng = np.random.default_rng(seed)
    lams_nom = np.asarray(lams_nm, dtype=float) * 1e-9
    k = 4 * PI / lams_nom
    grid_um = np.arange(0, D_um + coarse_step_um / 2, coarse_step_um)
    code = wrap((grid_um[:, None] * 1e-6) * k[None, :])
    feat_code = np.concatenate([np.cos(code), np.sin(code)], axis=1).T

    d_true_um = rng.uniform(0, D_um, N)
    lams_true = lams_nom[None, :] + rng.normal(0, sigma_lam_pm * 1e-12, size=(N, 3))
    phi = wrap((d_true_um[:, None] * 1e-6) * (4 * PI / lams_true)
               + rng.normal(0, sigma_phi_rad, size=(N, 3)))
    feat = np.concatenate([np.cos(phi), np.sin(phi)], axis=1)

    idx = np.empty(N, dtype=np.int32)
    for s in range(0, N, 2500):
        score = feat[s:s + 2500] @ feat_code
        idx[s:s + len(score)] = np.argmax(score, axis=1)

    d0_um = grid_um[idx]
    pred = wrap((d0_um[:, None] * 1e-6) * k[None, :])
    resid = wrap(phi - pred)
    delta_m = (resid @ k) / np.sum(k * k)
    d_hat_um = np.clip(d0_um + delta_m * 1e6, 0, D_um)
    err_nm = (d_hat_um - d_true_um) * 1000
    ok = np.abs(err_nm) < success_tol_um * 1000
    s = int(ok.sum())
    lo, hi = cp_ci(s, N)
    abs_e = np.abs(err_nm)
    metrics = {
        'N': N, 'successes': s,
        'ambiguity_success_pct': 100 * s / N,
        'alias_failure_pct': 100 * (1 - s / N),
        'success_ci_low_pct': 100 * lo, 'success_ci_high_pct': 100 * hi,
        'rmse_all_nm': float(np.sqrt(np.mean(err_nm ** 2))),
        'mae_all_nm': float(np.mean(abs_e)),
        'rmse_success_nm': float(np.sqrt(np.mean(err_nm[ok] ** 2))) if s else np.nan,
        'mae_success_nm': float(np.mean(abs_e[ok])) if s else np.nan,
        'p95_abs_success_nm': float(np.percentile(abs_e[ok], 95)) if s else np.nan,
        'max_abs_success_nm': float(np.max(abs_e[ok])) if s else np.nan,
    }
    detail = pd.DataFrame({'d_true_um': d_true_um, 'd_hat_um': d_hat_um,
                           'error_nm': err_nm, 'correct_ambiguity': ok.astype(int)})
    return metrics, detail

"""
Generate new figures with updated Jones law of motion and comparison plots.

New equation: S' = k / (f^f * (1-f)^(1-f)) * L^(1-f) * C^f * S^(1-beta)
Old equation: S' = k * (L/(1-f))^alpha * C^zeta * S^(1-beta)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expit, logit
import os

# Global configuration
YEAR_START = 2026
YEAR_CUTOFF = 2070
YEAR_NOT_REACHED = 2100
YEAR_PLOT_END = 2055
T_END = YEAR_CUTOFF - YEAR_START

# =============================================================================
# Shared helper functions
# =============================================================================

def triangular_sample(low, high):
    mode = (low + high) / 2
    return np.random.triangular(low, mode, high)

def sigmoid(x):
    return expit(x)

def f_automation(C, S, v, E_hac):
    return sigmoid(v * (np.log(C * S) - np.log(E_hac)))

def ff_prefactor_scalar(f):
    if f <= 0 or f >= 1:
        return 1.0
    return 1.0 / (f**f * (1 - f)**(1 - f))

def ff_prefactor_array(f):
    f = np.asarray(f, dtype=float)
    result = np.ones_like(f)
    mask = (f > 0) & (f < 1)
    fm = f[mask]
    result[mask] = 1.0 / (fm**fm * (1 - fm)**(1 - fm))
    return result

# =============================================================================
# Compute and Labor schedules (shared)
# =============================================================================

def _build_compute_schedule_cache(C_2026, t_max=100):
    cache = {0: C_2026}
    for t in range(1, 4):
        cache[t] = cache[t-1] * 2.6
    cache[4] = cache[3] * 2.0
    for t in range(5, 33):
        rate = 2.0 - 0.75 * (t - 1 - 4) / 28
        cache[t] = cache[t-1] * rate
    for t in range(33, t_max + 1):
        cache[t] = cache[t-1] * 1.25
    return cache

def _build_labor_schedule_cache(L_2026, t_max=100):
    cache = {0: L_2026}
    for t in range(1, 4):
        cache[t] = cache[t-1] * 2.0
    for t in range(4, t_max + 1):
        cache[t] = cache[t-1] * 1.10
    return cache

_C_CACHE = _build_compute_schedule_cache(1.0)
_L_CACHE = _build_labor_schedule_cache(1.0)

def compute_schedule(t, C_2026):
    if t < 0:
        return C_2026 * (2.6 ** t)
    t_floor = int(t)
    t_ceil = t_floor + 1
    frac = t - t_floor
    C_floor = _C_CACHE.get(t_floor, _C_CACHE[max(_C_CACHE.keys())]) * C_2026
    C_ceil = _C_CACHE.get(t_ceil, _C_CACHE[max(_C_CACHE.keys())] * 1.25) * C_2026
    if frac == 0:
        return C_floor
    return C_floor * (C_ceil / C_floor) ** frac

def labor_schedule(t, L_2026):
    if t < 0:
        return L_2026 * (2.0 ** t)
    t_floor = int(t)
    t_ceil = t_floor + 1
    frac = t - t_floor
    L_floor = _L_CACHE.get(t_floor, _L_CACHE[max(_L_CACHE.keys())]) * L_2026
    L_ceil = _L_CACHE.get(t_ceil, _L_CACHE[max(_L_CACHE.keys())] * 1.10) * L_2026
    if frac == 0:
        return L_floor
    return L_floor * (L_ceil / L_floor) ** frac

# =============================================================================
# OLD MODEL: S' = k * (L/(1-f))^alpha * C^zeta * S^(1-beta)
# =============================================================================

def old_sample_parameters():
    one_over_v = triangular_sample(1.5, 4.2)
    v = 1.0 / one_over_v
    f_2026 = triangular_sample(0.25, 0.5)
    alpha_ratio = triangular_sample(0.12, 0.35)
    alpha_plus_zeta = triangular_sample(0.8, 1.0)
    beta = triangular_sample(0.3, 1.0)

    alpha = alpha_ratio * alpha_plus_zeta
    zeta = (1 - alpha_ratio) * alpha_plus_zeta

    L_2026 = 1.0
    S_2026 = 1.0
    C_2026 = 1.0

    log_E_hac = np.log(C_2026 * S_2026) - logit(f_2026) / v
    E_hac = np.exp(log_E_hac)

    dS_dt_2026 = S_2026 * np.log(5)
    labor_term = (L_2026 / (1 - f_2026)) ** alpha
    C_term = C_2026 ** zeta
    S_term = S_2026 ** (1 - beta)
    k = dS_dt_2026 / (labor_term * C_term * S_term)

    return {
        'alpha': alpha, 'beta': beta, 'zeta': zeta,
        'v': v, 'one_over_v': one_over_v,
        'L_2026': L_2026, 'C_2026': C_2026, 'S_2026': S_2026,
        'E_hac': E_hac, 'f_2026': f_2026,
        'alpha_ratio': alpha_ratio, 'alpha_plus_zeta': alpha_plus_zeta,
        'k': k
    }

def old_jones_derivative(t, S, alpha, beta, zeta, v, k, L_2026, C_2026, E_hac):
    if S <= 0:
        return 0
    C = compute_schedule(t, C_2026)
    L = labor_schedule(t, L_2026)
    f = f_automation(C, S, v, E_hac)
    if f >= 0.999999:
        f = 0.999999
    effective_labor = L / (1 - f)
    dS_dt = k * (effective_labor ** alpha) * (C ** zeta) * (S ** (1 - beta))
    return dS_dt

def old_simulate_trajectory(params, t_start=0, t_end=T_END, max_S=1e12):
    alpha = params['alpha']
    beta = params['beta']
    zeta = params['zeta']
    v = params['v']
    k = params['k']
    L_2026 = params['L_2026']
    C_2026 = params['C_2026']
    E_hac = params['E_hac']
    S_2026 = params['S_2026']

    dlogS_dt = np.log(5)
    S_init = S_2026 * np.exp(dlogS_dt * t_start)

    def automation_99_event(t, S, *args):
        C = compute_schedule(t, C_2026)
        f = f_automation(C, S[0], v, E_hac)
        return f - 0.99
    automation_99_event.terminal = False
    automation_99_event.direction = 1

    def max_S_event(t, S, *args):
        return S[0] - max_S
    max_S_event.terminal = True
    max_S_event.direction = 1

    try:
        sol = solve_ivp(
            old_jones_derivative,
            (t_start, t_end), [S_init],
            args=(alpha, beta, zeta, v, k, L_2026, C_2026, E_hac),
            dense_output=True,
            events=[automation_99_event, max_S_event],
            max_step=0.1, atol=1e-8
        )

        t_dense = np.linspace(t_start, sol.t[-1], 500)
        S_dense = sol.sol(t_dense)[0]
        C_dense = np.array([compute_schedule(t, C_2026) for t in t_dense])
        L_dense = np.array([labor_schedule(t, L_2026) for t in t_dense])
        f_dense = np.array([f_automation(C_dense[i], S_dense[i], v, E_hac) for i in range(len(t_dense))])

        cap_mask = f_dense >= 0.999999
        if np.any(cap_mask):
            cap_idx = np.argmax(cap_mask) + 1
            t_dense = t_dense[:cap_idx]
            S_dense = S_dense[:cap_idx]
            C_dense = C_dense[:cap_idx]
            L_dense = L_dense[:cap_idx]
            f_dense = f_dense[:cap_idx]

        f_capped = np.minimum(f_dense, 0.999999)
        R_dense = ((L_dense) ** (1-f_capped)) * (C_dense ** f_capped)

        reached_99 = None
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            reached_99 = sol.t_events[0][0] + YEAR_START

        return t_dense, S_dense, f_dense, R_dense, reached_99
    except Exception as e:
        print(f"Old model integration error: {e}")
        return None, None, None, None, None

def old_generate_trajectories(n=40, t_start=0, t_end=T_END, seed=42):
    np.random.seed(seed)
    trajectories = []
    params_list = []
    times_to_99 = []
    for i in range(n):
        params = old_sample_parameters()
        result = old_simulate_trajectory(params, t_start=t_start, t_end=t_end)
        t_arr, S_arr, f_arr, R_arr, time_99 = result
        if t_arr is not None:
            trajectories.append((t_arr, S_arr, f_arr, R_arr))
            params_list.append(params)
            if time_99 is None:
                time_99 = YEAR_NOT_REACHED
            times_to_99.append(time_99)
    return trajectories, params_list, times_to_99

# =============================================================================
# NEW MODEL: S' = k / (f^f * (1-f)^(1-f)) * L^(1-f) * C^f * S^(1-beta)
# =============================================================================

def new_sample_parameters():
    one_over_v = triangular_sample(1.5, 4.2)
    v = 1.0 / one_over_v
    f_2026 = triangular_sample(0.25, 0.5)
    beta = triangular_sample(0.3, 1.0)

    L_2026 = 1.0
    S_2026 = 1.0
    C_2026 = 1.0

    log_E_hac = np.log(C_2026 * S_2026) - logit(f_2026) / v
    E_hac = np.exp(log_E_hac)

    dS_dt_2026 = S_2026 * np.log(5)
    prefactor = ff_prefactor_scalar(f_2026)
    labor_term = L_2026 ** (1 - f_2026)
    C_term = C_2026 ** f_2026
    S_term = S_2026 ** (1 - beta)
    k = dS_dt_2026 / (prefactor * labor_term * C_term * S_term)

    return {
        'beta': beta, 'v': v, 'one_over_v': one_over_v,
        'L_2026': L_2026, 'C_2026': C_2026, 'S_2026': S_2026,
        'E_hac': E_hac, 'f_2026': f_2026, 'k': k
    }

def new_jones_derivative(t, S, beta, v, k, L_2026, C_2026, E_hac):
    if S <= 0:
        return 0
    C = compute_schedule(t, C_2026)
    L = labor_schedule(t, L_2026)
    f = f_automation(C, S, v, E_hac)
    if f >= 0.999999:
        f = 0.999999
    if f <= 0.000001:
        f = 0.000001
    prefactor = ff_prefactor_scalar(f)
    dS_dt = k * prefactor * (L ** (1 - f)) * (C ** f) * (S ** (1 - beta))
    return dS_dt

def new_simulate_trajectory(params, t_start=0, t_end=T_END, max_S=1e12):
    beta = params['beta']
    v = params['v']
    k = params['k']
    L_2026 = params['L_2026']
    C_2026 = params['C_2026']
    E_hac = params['E_hac']
    S_2026 = params['S_2026']

    dlogS_dt = np.log(5)
    S_init = S_2026 * np.exp(dlogS_dt * t_start)

    def automation_99_event(t, S, *args):
        C = compute_schedule(t, C_2026)
        f = f_automation(C, S[0], v, E_hac)
        return f - 0.99
    automation_99_event.terminal = False
    automation_99_event.direction = 1

    def max_S_event(t, S, *args):
        return S[0] - max_S
    max_S_event.terminal = True
    max_S_event.direction = 1

    try:
        sol = solve_ivp(
            new_jones_derivative,
            (t_start, t_end), [S_init],
            args=(beta, v, k, L_2026, C_2026, E_hac),
            dense_output=True,
            events=[automation_99_event, max_S_event],
            max_step=0.1, atol=1e-8
        )

        t_dense = np.linspace(t_start, sol.t[-1], 500)
        S_dense = sol.sol(t_dense)[0]
        C_dense = np.array([compute_schedule(t, C_2026) for t in t_dense])
        L_dense = np.array([labor_schedule(t, L_2026) for t in t_dense])
        f_dense = np.array([f_automation(C_dense[i], S_dense[i], v, E_hac) for i in range(len(t_dense))])

        cap_mask = f_dense >= 0.999999
        if np.any(cap_mask):
            cap_idx = np.argmax(cap_mask) + 1
            t_dense = t_dense[:cap_idx]
            S_dense = S_dense[:cap_idx]
            C_dense = C_dense[:cap_idx]
            L_dense = L_dense[:cap_idx]
            f_dense = f_dense[:cap_idx]

        f_capped = np.minimum(np.maximum(f_dense, 0.000001), 0.999999)
        prefactor_arr = ff_prefactor_array(f_capped)
        R_dense = prefactor_arr * (L_dense ** (1 - f_capped)) * (C_dense ** f_capped)

        reached_99 = None
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            reached_99 = sol.t_events[0][0] + YEAR_START

        return t_dense, S_dense, f_dense, R_dense, reached_99
    except Exception as e:
        print(f"New model integration error: {e}")
        return None, None, None, None, None

def new_generate_trajectories(n=40, t_start=0, t_end=T_END, seed=42):
    np.random.seed(seed)
    trajectories = []
    params_list = []
    times_to_99 = []
    for i in range(n):
        params = new_sample_parameters()
        result = new_simulate_trajectory(params, t_start=t_start, t_end=t_end)
        t_arr, S_arr, f_arr, R_arr, time_99 = result
        if t_arr is not None:
            trajectories.append((t_arr, S_arr, f_arr, R_arr))
            params_list.append(params)
            if time_99 is None:
                time_99 = YEAR_NOT_REACHED
            times_to_99.append(time_99)
    return trajectories, params_list, times_to_99

# =============================================================================
# Plot generation functions
# =============================================================================

def compute_median_by_buckets(param_values, times_to_99, n_buckets=10):
    if len(param_values) < n_buckets:
        return None, None, None
    percentiles = np.linspace(0, 100, n_buckets + 1)
    bucket_edges = np.percentile(param_values, percentiles)
    bucket_centers = []
    bucket_medians = []
    for i in range(n_buckets):
        low, high = bucket_edges[i], bucket_edges[i + 1]
        if i == n_buckets - 1:
            mask = [(low <= p <= high) for p in param_values]
        else:
            mask = [(low <= p < high) for p in param_values]
        times_in_bucket = [t for t, m in zip(times_to_99, mask) if m]
        params_in_bucket = [p for p, m in zip(param_values, mask) if m]
        if len(times_in_bucket) >= 3:
            bucket_centers.append(np.median(params_in_bucket))
            bucket_medians.append(np.median(times_in_bucket))
    return bucket_centers, bucket_medians, bucket_edges


def plot_trajectories(trajectories, times_to_99, save_path, title_prefix=""):
    """Plot 4-panel trajectories figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.07, hspace=0.3, wspace=0.3)
    fig.suptitle(f'{title_prefix}Other Metrics Over Time (40 sampled trajectories)', fontsize=16)

    times_for_color = [min(t, YEAR_CUTOFF) for t in times_to_99]
    t_min, t_max = min(times_for_color), max(times_for_color)
    cmap = plt.cm.viridis

    for i, (t_arr, S_arr, f_arr, R_arr) in enumerate(trajectories):
        if times_to_99[i] < YEAR_NOT_REACHED:
            color = cmap((times_for_color[i] - t_min) / (t_max - t_min + 0.01))
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.4

        axes[0, 0].plot(t_arr + YEAR_START, S_arr, color=color, alpha=alpha, linewidth=1)
        axes[1, 0].plot(t_arr + YEAR_START, R_arr, color=color, alpha=alpha, linewidth=1)

        # Compute:labor ratio
        f_capped = np.minimum(np.maximum(f_arr, 0.000001), 0.999999)
        C_dense = np.array([compute_schedule(t, 1.0) for t in t_arr])
        L_dense = np.array([labor_schedule(t, 1.0) for t in t_arr])
        ratio = (C_dense ** f_capped) / (L_dense ** (1 - f_capped))
        idx_2026 = np.argmin(np.abs(t_arr - 0))
        ratio_norm = ratio / ratio[idx_2026]
        axes[0, 1].plot(t_arr + YEAR_START, ratio_norm, color=color, alpha=alpha, linewidth=1)
        axes[1, 1].plot(t_arr + YEAR_START, ratio_norm, color=color, alpha=alpha, linewidth=1)

    axes[0, 0].set_xlabel('Year', fontsize=12)
    axes[0, 0].set_ylabel('Algorithmic Efficiency', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Algorithmic Efficiency S', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(YEAR_START, YEAR_PLOT_END)

    axes[0, 1].set_xlabel('Year', fontsize=12)
    axes[0, 1].set_ylabel('Compute:Labor Ratio (2026=1)', fontsize=12)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Compute:Labor Ratio $C^f / L^{1-f}$', fontsize=14)
    axes[0, 1].axhline(1, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(YEAR_START, YEAR_PLOT_END)

    axes[1, 0].set_xlabel('Year', fontsize=12)
    axes[1, 0].set_ylabel('Research Production', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Research Production R(t)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(YEAR_START, YEAR_PLOT_END)

    axes[1, 1].set_xlabel('Year', fontsize=12)
    axes[1, 1].set_ylabel('Compute:Labor Ratio (2026=1)', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Compute:Labor Ratio $C^f / L^{1-f}$', fontsize=14)
    axes[1, 1].axhline(1, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(YEAR_START, YEAR_PLOT_END)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t_min, vmax=t_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Year of 99% automation')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_automation(trajectories, times_to_99, times_to_99_hist, save_path, title_prefix=""):
    """Plot automation trajectories + histogram."""
    times_for_color = [min(t, YEAR_CUTOFF) for t in times_to_99]
    t_min, t_max = min(times_for_color), max(times_for_color)
    cmap = plt.cm.viridis
    median_99 = np.median(times_to_99_hist)

    fig, (ax_f, ax_hist) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                          gridspec_kw={'hspace': 0.05})

    for i, (t_arr, S_arr, f_arr, R_arr) in enumerate(trajectories):
        if times_to_99[i] < YEAR_NOT_REACHED:
            color = cmap((times_for_color[i] - t_min) / (t_max - t_min + 0.01))
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.4
        ax_f.plot(t_arr + YEAR_START, f_arr, color=color, alpha=alpha, linewidth=1)

    ax_f.set_ylabel('Automation fraction f', fontsize=12)
    ax_f.set_title(f'{title_prefix}Time Until 99% AI R&D Automation (40 sampled trajectories)', fontsize=14)
    ax_f.set_yscale('logit')
    ax_f.set_ylim(0.1, 0.999)
    ax_f.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_f.axhline(0.9, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_f.axhline(0.98, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_f.axhline(0.99, color='red', linestyle='--', alpha=0.7, linewidth=2, label='99% automation')
    ax_f.legend(fontsize=11)
    ax_f.grid(True, alpha=0.3)
    ax_f.set_xlim(YEAR_START, YEAR_PLOT_END)
    ax_f.tick_params(labelbottom=False)
    ax_f.text(0.97, 0.05, f'Median: {median_99:.1f}', transform=ax_f.transAxes,
              fontsize=12, ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax_hist.hist(times_to_99_hist, bins=np.arange(2027, YEAR_PLOT_END + 0.5, 0.5),
                 edgecolor='black', linewidth=0.5, alpha=0.7, color='steelblue')
    ax_hist.axvline(median_99, color='red', linestyle='--', linewidth=2,
                    label=f'Median: {median_99:.1f}')
    ax_hist.set_xlabel('Year', fontsize=12)
    ax_hist.set_ylabel('Count', fontsize=12)
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_xlim(YEAR_START, YEAR_PLOT_END)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_sensitivity(params_list, times_to_99, save_path, model_type="new", title_prefix=""):
    """Plot sensitivity analysis."""
    betas = [p['beta'] for p in params_list]
    one_over_vs = [p['one_over_v'] for p in params_list]
    f_2026s = [p['f_2026'] for p in params_list]

    variables = [
        (betas, r'$\beta$', 'Beta'),
        (one_over_vs, '1/v', 'Automation slowness'),
        (f_2026s, 'f(2026)', 'Initial automation'),
    ]

    if model_type == "old":
        alphas = [p['alpha'] for p in params_list]
        zetas = [p['zeta'] for p in params_list]
        alpha_ratios = [p['alpha_ratio'] for p in params_list]
        variables = [
            (alphas, r'$\alpha$', 'Alpha'),
            (betas, r'$\beta$', 'Beta'),
            (zetas, r'$\zeta$', 'Zeta'),
            (one_over_vs, '1/v', 'Automation slowness'),
            (f_2026s, 'f(2026)', 'Initial automation'),
            (alpha_ratios, r'$\alpha/(\alpha+\zeta)$', 'Labor share'),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (values, label, title) in zip(np.array(axes).flat, variables):
        centers, medians, edges = compute_median_by_buckets(values, times_to_99, n_buckets=10)
        if centers is not None:
            ax.plot(centers, medians, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Median year of 99% automation', fontsize=11)
            ax.set_title(f'Conditional on {title}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(min(medians) - 1, max(medians) + 1)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle(f'{title_prefix}Sensitivity Analysis', fontsize=16)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def plot_comparison_trajectories(old_trajs, old_times, new_trajs, new_times, save_path):
    """Side-by-side comparison of trajectory plots."""
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle('Trajectory Comparison: Original (left) vs New Equation (right)', fontsize=18, y=0.98)

    # Left half: old model
    for col_offset, (trajs, times, label) in enumerate([
        (old_trajs, old_times, "Original: $S' = k(L/(1-f))^\\alpha C^\\zeta S^{1-\\beta}$"),
        (new_trajs, new_times, "New: $S' = \\frac{k}{f^f(1-f)^{1-f}} L^{1-f} C^f S^{1-\\beta}$")
    ]):
        c0 = col_offset * 2  # column offset

        times_for_color = [min(t, YEAR_CUTOFF) for t in times]
        t_min_c, t_max_c = min(times_for_color), max(times_for_color)
        cmap = plt.cm.viridis

        for i, (t_arr, S_arr, f_arr, R_arr) in enumerate(trajs):
            if times[i] < YEAR_NOT_REACHED:
                color = cmap((times_for_color[i] - t_min_c) / (t_max_c - t_min_c + 0.01))
                alpha = 0.7
            else:
                color = 'gray'
                alpha = 0.4

            axes[0, c0].plot(t_arr + YEAR_START, S_arr, color=color, alpha=alpha, linewidth=1)
            axes[1, c0].plot(t_arr + YEAR_START, R_arr, color=color, alpha=alpha, linewidth=1)
            axes[0, c0+1].plot(t_arr + YEAR_START, f_arr, color=color, alpha=alpha, linewidth=1)

            # Compute:labor ratio
            f_capped = np.minimum(np.maximum(f_arr, 0.000001), 0.999999)
            C_d = np.array([compute_schedule(t, 1.0) for t in t_arr])
            L_d = np.array([labor_schedule(t, 1.0) for t in t_arr])
            ratio = (C_d ** f_capped) / (L_d ** (1 - f_capped))
            idx0 = np.argmin(np.abs(t_arr - 0))
            ratio_norm = ratio / ratio[idx0]
            axes[1, c0+1].plot(t_arr + YEAR_START, ratio_norm, color=color, alpha=alpha, linewidth=1)

        # S(t)
        axes[0, c0].set_ylabel('Algorithmic Efficiency', fontsize=11)
        axes[0, c0].set_yscale('log')
        axes[0, c0].set_title(f'S(t) - {label}', fontsize=11)
        axes[0, c0].grid(True, alpha=0.3)
        axes[0, c0].set_xlim(YEAR_START, YEAR_PLOT_END)

        # f(t)
        axes[0, c0+1].set_ylabel('Automation fraction f', fontsize=11)
        axes[0, c0+1].set_yscale('logit')
        axes[0, c0+1].set_ylim(0.1, 0.999)
        axes[0, c0+1].axhline(0.99, color='red', linestyle='--', alpha=0.7, linewidth=1)
        axes[0, c0+1].set_title(f'f(t) - {label}', fontsize=11)
        axes[0, c0+1].grid(True, alpha=0.3)
        axes[0, c0+1].set_xlim(YEAR_START, YEAR_PLOT_END)

        # R(t)
        axes[1, c0].set_ylabel('Research Production', fontsize=11)
        axes[1, c0].set_yscale('log')
        axes[1, c0].set_title(f'R(t) - {label}', fontsize=11)
        axes[1, c0].grid(True, alpha=0.3)
        axes[1, c0].set_xlim(YEAR_START, YEAR_PLOT_END)

        # C:L ratio
        axes[1, c0+1].set_ylabel('Compute:Labor Ratio (2026=1)', fontsize=11)
        axes[1, c0+1].set_yscale('log')
        axes[1, c0+1].axhline(1, color='gray', linestyle='--', alpha=0.5)
        axes[1, c0+1].set_title(f'C:L Ratio - {label}', fontsize=11)
        axes[1, c0+1].grid(True, alpha=0.3)
        axes[1, c0+1].set_xlim(YEAR_START, YEAR_PLOT_END)

    for ax in axes.flat:
        ax.set_xlabel('Year', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_comparison_automation(old_trajs, old_times, old_times_hist,
                               new_trajs, new_times, new_times_hist, save_path):
    """Side-by-side comparison of automation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('Automation Comparison: Original (left) vs New Equation (right)', fontsize=16, y=0.98)

    for col, (trajs, times, times_hist, label) in enumerate([
        (old_trajs, old_times, old_times_hist, "Original"),
        (new_trajs, new_times, new_times_hist, "New Equation")
    ]):
        times_for_color = [min(t, YEAR_CUTOFF) for t in times]
        t_min_c, t_max_c = min(times_for_color), max(times_for_color)
        cmap = plt.cm.viridis
        median_99 = np.median(times_hist)

        # Top: trajectories
        for i, (t_arr, S_arr, f_arr, R_arr) in enumerate(trajs):
            if times[i] < YEAR_NOT_REACHED:
                color = cmap((times_for_color[i] - t_min_c) / (t_max_c - t_min_c + 0.01))
                alpha = 0.7
            else:
                color = 'gray'
                alpha = 0.4
            axes[0, col].plot(t_arr + YEAR_START, f_arr, color=color, alpha=alpha, linewidth=1)

        axes[0, col].set_ylabel('Automation fraction f', fontsize=12)
        axes[0, col].set_title(f'{label} - Automation Trajectories', fontsize=13)
        axes[0, col].set_yscale('logit')
        axes[0, col].set_ylim(0.1, 0.999)
        axes[0, col].axhline(0.99, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_xlim(YEAR_START, YEAR_PLOT_END)
        axes[0, col].text(0.97, 0.05, f'Median: {median_99:.1f}', transform=axes[0, col].transAxes,
                          fontsize=12, ha='right', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

        # Bottom: histogram
        axes[1, col].hist(times_hist, bins=np.arange(2027, YEAR_PLOT_END + 0.5, 0.5),
                          edgecolor='black', linewidth=0.5, alpha=0.7, color='steelblue')
        axes[1, col].axvline(median_99, color='red', linestyle='--', linewidth=2,
                             label=f'Median: {median_99:.1f}')
        axes[1, col].set_xlabel('Year', fontsize=12)
        axes[1, col].set_ylabel('Count', fontsize=12)
        axes[1, col].set_title(f'{label} - Distribution of 99% Automation', fontsize=13)
        axes[1, col].legend(fontsize=10)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_xlim(YEAR_START, YEAR_PLOT_END)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_comparison_sensitivity(old_params, old_times, new_params, new_times, save_path):
    """Side-by-side comparison of sensitivity analysis."""
    # Common parameters: beta, 1/v, f(2026)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sensitivity Comparison: Original (blue) vs New Equation (red)', fontsize=16, y=0.98)

    common_vars = [
        ('beta', r'$\beta$', 'Beta'),
        ('one_over_v', '1/v', 'Automation slowness'),
        ('f_2026', 'f(2026)', 'Initial automation'),
    ]

    # Top row: Old model
    # Bottom row: New model
    for row, (params_list, times, label, color) in enumerate([
        (old_params, old_times, "Original", "steelblue"),
        (new_params, new_times, "New Equation", "firebrick")
    ]):
        for j, (key, xlabel, title) in enumerate(common_vars):
            values = [p[key] for p in params_list]
            centers, medians, _ = compute_median_by_buckets(values, times, n_buckets=10)
            if centers is not None:
                axes[row, j].plot(centers, medians, 'o-', linewidth=2, markersize=8, color=color)
                axes[row, j].set_xlabel(xlabel, fontsize=12)
                axes[row, j].set_ylabel('Median year of 99% automation', fontsize=11)
                axes[row, j].set_title(f'{label}: Conditional on {title}', fontsize=12)
                axes[row, j].grid(True, alpha=0.3)
                axes[row, j].set_ylim(min(medians) - 1, max(medians) + 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs('new_figures', exist_ok=True)
    os.makedirs('comparison_figures', exist_ok=True)

    # -------------------------------------------------------------------------
    # Generate OLD model data
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Generating OLD model trajectories...")
    print("=" * 60)
    old_trajs_40, old_params_40, old_times_40 = old_generate_trajectories(n=40, seed=42)
    print(f"  40 trajectories: {sum(1 for t in old_times_40 if t < YEAR_NOT_REACHED)}/{len(old_times_40)} reached 99%")

    print("  Generating 500 for histogram...")
    _, old_params_500, old_times_500 = old_generate_trajectories(n=500, seed=123)
    print(f"  Median 99% automation: {np.median(old_times_500):.1f}")

    # -------------------------------------------------------------------------
    # Generate NEW model data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating NEW model trajectories...")
    print("=" * 60)
    new_trajs_40, new_params_40, new_times_40 = new_generate_trajectories(n=40, seed=42)
    print(f"  40 trajectories: {sum(1 for t in new_times_40 if t < YEAR_NOT_REACHED)}/{len(new_times_40)} reached 99%")

    print("  Generating 500 for histogram...")
    _, new_params_500, new_times_500 = new_generate_trajectories(n=500, seed=123)
    print(f"  Median 99% automation: {np.median(new_times_500):.1f}")

    # -------------------------------------------------------------------------
    # Generate NEW model plots (in new_figures/)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating new model plots...")
    print("=" * 60)
    plot_trajectories(new_trajs_40, new_times_40, 'new_figures/trajectories.png')
    plot_automation(new_trajs_40, new_times_40, new_times_500, 'new_figures/automation.png')
    plot_sensitivity(new_params_500, new_times_500, 'new_figures/sensitivity.png', model_type="new")

    # -------------------------------------------------------------------------
    # Generate COMPARISON plots (in comparison_figures/)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)
    plot_comparison_trajectories(old_trajs_40, old_times_40, new_trajs_40, new_times_40,
                                 'comparison_figures/comparison_trajectories.png')
    plot_comparison_automation(old_trajs_40, old_times_40, old_times_500,
                               new_trajs_40, new_times_40, new_times_500,
                               'comparison_figures/comparison_automation.png')
    plot_comparison_sensitivity(old_params_500, old_times_500, new_params_500, new_times_500,
                                'comparison_figures/comparison_sensitivity.png')

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOld model: S' = k * (L/(1-f))^alpha * C^zeta * S^(1-beta)")
    print(f"  Parameters: alpha, beta, zeta, v, f(2026)")
    print(f"  Median 99% automation year: {np.median(old_times_500):.1f}")
    print(f"\nNew model: S' = k / (f^f * (1-f)^(1-f)) * L^(1-f) * C^f * S^(1-beta)")
    print(f"  Parameters: beta, v, f(2026)")
    print(f"  Median 99% automation year: {np.median(new_times_500):.1f}")
    print(f"\nFiles generated:")
    print(f"  new_figures/trajectories.png")
    print(f"  new_figures/automation.png")
    print(f"  new_figures/sensitivity.png")
    print(f"  comparison_figures/comparison_trajectories.png")
    print(f"  comparison_figures/comparison_automation.png")
    print(f"  comparison_figures/comparison_sensitivity.png")

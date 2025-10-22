# paired_param_sweep_selfcontained.py
# Self-contained paired parameter sweep for the growth->memory model.
# Paste into Colab / Jupyter and run. Produces ./sweep_out/ CSVs + PNG heatmaps.
import os, time, itertools
import numpy as np
from numpy.random import default_rng
from scipy.spatial import cKDTree
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------- Output directory ---------------------------
OUT = "./sweep_out"
os.makedirs(OUT, exist_ok=True)

# --------------------------- Global model parameters (default) -----------
# You can tweak these before running.
RNG_SEED = 42
N = 80
L = 100.0
dt_g = 0.5
T_grow_steps = 2000
alpha0 = 0.6
D0 = 0.2
beta = 0.05
k_alpha = 1.2
k_D = 3.0
r_syn = 2.0
w0 = 1.0
a_B = 2.0
w_max = 3.0
gamma_dist = 0.0
tau = 10.0
phi = np.tanh

M = 2
A_m = [1.0, 1.0]
mu_m = [np.array([30.,50.]), np.array([70.,50.])]
sigma_m = [15.0, 15.0]

t_pk = T_grow_steps / 3.0
sigma_B = T_grow_steps / 6.0

C_baseline = 0.01
A_c = 1.5
t_tr = t_pk
sigma_c = T_grow_steps / 30.0

n_holes = 3
hole_centers = [np.array([40.,40.]), np.array([60.,60.]), np.array([50.,30.])]
hole_rho = 6.0
hole_H = 0.9

dt_rate = 0.1
T_cue = 50.0
T_test = 400.0
cue_amplitude = 2.0
pattern_sparsity = 0.1
S_thresh = 0.5

APPLY_EI = True
FRAC_INH = 0.1
INH_STRENGTH = 0.2
ROW_NORM_DESIRED = 1.0

SPEC_SCALING = 0.9

# --------------------------- Helper math functions ---------------------------
def B0_func(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    val = np.zeros(x.shape[0], dtype=float)
    for A, mu, sig in zip(A_m, mu_m, sigma_m):
        diff = x - mu
        d2 = np.sum(diff**2, axis=-1)
        val += A * np.exp(-d2 / (2.0 * sig**2))
    return val

def grad_B0(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    grad = np.zeros_like(x, dtype=float)
    for A, mu, sig in zip(A_m, mu_m, sigma_m):
        diff = x - mu
        exp_part = np.exp(-np.sum(diff**2, axis=-1) / (2.0 * sig**2))
        grad += A * exp_part[:, None] * (-(diff) / (sig**2))
    return grad

def Xi_func(x, trauma_active=False, rng_local=None):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    if rng_local is None:
        rng_local = default_rng(RNG_SEED)
    noise = 0.02 * rng_local.standard_normal(size=(x.shape[0],))
    val = noise.copy()
    if trauma_active:
        for center in hole_centers:
            diff = x - center
            d2 = np.sum(diff**2, axis=-1)
            val += -hole_H * np.exp(-d2 / (2.0 * hole_rho**2))
    return val

def B_field_and_grad(x, t_step, trauma_active=False, rng_local=None):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    if rng_local is None:
        rng_local = default_rng(RNG_SEED)
    S_B_t = np.exp(-((t_step - t_pk)**2) / (2.0 * sigma_B**2))
    b0_vals = B0_func(x)
    grad_b0 = grad_B0(x)
    xi_vals = Xi_func(x, trauma_active=trauma_active, rng_local=rng_local)
    B_vals = b0_vals * S_B_t + xi_vals
    grad_B = grad_b0 * S_B_t
    if trauma_active:
        for center in hole_centers:
            diff = x - center
            d2 = np.sum(diff**2, axis=-1)
            exp_part = np.exp(-d2 / (2.0 * hole_rho**2))
            grad_h = (hole_H * exp_part)[:, None] * (diff / (hole_rho**2))
            grad_B += grad_h
    return B_vals, grad_B

def cortisol_schedule(t_step, trauma_active=False):
    if trauma_active:
        return C_baseline + A_c * np.exp(-((t_step - t_tr)**2) / (2.0 * sigma_c**2))
    return C_baseline

def apply_ei_balance(W, seed, frac_inh=FRAC_INH, inh_strength=INH_STRENGTH, row_norm=ROW_NORM_DESIRED):
    rng_local = default_rng(int(seed) + 99999)
    Nloc = W.shape[0]
    n_inh = max(1, int(np.round(frac_inh * Nloc)))
    inh_idx = rng_local.choice(Nloc, size=n_inh, replace=False)
    Wm = W.copy().astype(float)
    for i in inh_idx:
        Wm[i, :] = -inh_strength * np.abs(Wm[i, :])
    row_abs = np.sum(np.abs(Wm), axis=1, keepdims=True)
    row_abs[row_abs == 0.0] = 1.0
    Wm = Wm / row_abs * row_norm
    for i in inh_idx:
        Wm[i, :] = -np.abs(Wm[i, :])
    return Wm, inh_idx

# --------------------------- Growth simulation (paired) ---------------------------
def simulate_growth(somata, targets_idx, trauma_active=False, seed=None):
    if seed is None:
        rng_local = default_rng(RNG_SEED)
    else:
        rng_local = default_rng(int(seed))
    Nloc = somata.shape[0]
    target_centers = np.array([mu_m[idx] for idx in targets_idx])

    tips = somata + 0.5 * (rng_local.random((Nloc,2)) - 0.5)
    n_trace = min(12, Nloc)
    trace_idx = np.arange(n_trace)
    trajectories = np.zeros((T_grow_steps+1, n_trace, 2))
    trajectories[0,:,:] = tips[trace_idx]

    W = np.zeros((Nloc, Nloc), dtype=float)
    connect_time = np.full((Nloc, Nloc), np.inf)
    connected_flag = np.zeros((Nloc, Nloc), dtype=bool)
    tree = cKDTree(somata)

    for n in range(T_grow_steps):
        B_vals, grad_B = B_field_and_grad(tips, n, trauma_active=trauma_active, rng_local=rng_local)
        C_t = cortisol_schedule(n, trauma_active=trauma_active)
        alpha_t = alpha0 * np.exp(-k_alpha * C_t)
        D_t = D0 * (1.0 + k_D * C_t)

        diff_to_target = target_centers - tips
        dists = np.linalg.norm(diff_to_target, axis=1)
        p_vec = np.zeros_like(diff_to_target)
        nonzero = dists>1e-8
        p_vec[nonzero] = (diff_to_target[nonzero] / dists[nonzero][:,None])

        xi = rng_local.standard_normal(size=(Nloc,2))
        tips = tips + dt_g * (alpha_t * grad_B + beta * p_vec) + np.sqrt(2.0 * D_t * dt_g) * xi
        tips = np.minimum(np.maximum(tips, 0.0), L)
        trajectories[n+1,:,:] = tips[trace_idx]

        neighbors_list = tree.query_ball_point(tips, r_syn)
        for i, neighs in enumerate(neighbors_list):
            if not neighs:
                continue
            for j in neighs:
                if j==i or connected_flag[i,j]:
                    continue
                B_local = B_vals[i] if np.ndim(B_vals)>0 else B_vals
                B_max_ref = max(A_m)
                B_norm = float(B_local / (B_max_ref + 1e-9))
                delta_ij = np.linalg.norm(tips[i] - somata[j])
                w_ij = w0 * (1.0 + a_B * B_norm) * np.exp(-gamma_dist * delta_ij)
                if w_ij < 0: w_ij = 0.0
                if w_max is not None:
                    w_ij = np.clip(w_ij, 0.0, w_max)
                W[i,j] = w_ij
                connected_flag[i,j] = True
                connect_time[i,j] = n
    return W, connect_time, trajectories

# --------------------------- Memory & metrics ---------------------------
def run_memory_and_metrics(W_raw, targets_idx, pattern_vec, seed_for_ei=0):
    W_work = W_raw.copy().astype(float)
    if APPLY_EI:
        W_work, inh_idx = apply_ei_balance(W_work, seed_for_ei, frac_inh=FRAC_INH, inh_strength=INH_STRENGTH, row_norm=ROW_NORM_DESIRED)

    eigs = np.linalg.eigvals(W_work)
    spec = max(np.abs(eigs)) if eigs.size>0 else 0.0
    if spec>0:
        W_sim = W_work * (SPEC_SCALING / spec)
    else:
        W_sim = W_work.copy()

    time_steps = int(np.ceil(T_test / dt_rate))
    cue_steps = int(np.ceil(T_cue / dt_rate))
    x = np.zeros(W_raw.shape[0], dtype=float)
    similarity = np.zeros(time_steps)
    for t_idx in range(time_steps):
        Icue = cue_amplitude * pattern_vec if t_idx < cue_steps else np.zeros_like(pattern_vec)
        dx = (-x + W_sim.dot(phi(x)) + Icue) * (dt_rate / tau)
        x = x + dx
        norm_x = np.linalg.norm(x) + 1e-12
        norm_p = np.linalg.norm(pattern_vec) + 1e-12
        similarity[t_idx] = np.dot(x, pattern_vec) / (norm_x * norm_p)

    post_cue = similarity[cue_steps:]
    drop_idxs = np.where(post_cue < S_thresh)[0]
    if drop_idxs.size>0:
        retention_after_offset = drop_idxs[0] * dt_rate
        retention_absolute = (cue_steps + drop_idxs[0]) * dt_rate
        hit_cap = False
    else:
        retention_after_offset = (T_test - T_cue)
        retention_absolute = T_test
        hit_cap = True

    window_steps = int(min(50.0 / dt_rate, max(1, int(0.2 * time_steps))))
    final_overlap = float(np.mean(similarity[-window_steps:]))
    pat_idx = (pattern_vec>0)
    energy_in_pattern = np.sum(x[pat_idx]**2) if np.any(pat_idx) else 0.0
    total_energy = np.sum(x**2)
    final_energy_frac = float(energy_in_pattern / total_energy) if total_energy>1e-12 else 0.0

    rows, cols = np.where(W_raw>0)
    if rows.size>0:
        dists = np.linalg.norm(somata_global[rows] - somata_global[cols], axis=1)
        mean_edge_length = float(np.mean(dists))
        mean_weight = float(np.mean(W_raw[rows, cols]))
        same_cluster = np.sum(targets_idx[rows] == targets_idx[cols])
        frac_within_cluster = float(same_cluster / len(rows))
        edge_count = len(rows)
    else:
        mean_edge_length = np.nan; mean_weight = np.nan; frac_within_cluster = 0.0; edge_count = 0

    eigvals_sim, eigvecs_sim = np.linalg.eig(W_sim)
    idxs = np.argsort(-np.abs(eigvals_sim))
    eigvals_sorted = eigvals_sim[idxs]
    eigvecs_sorted = eigvecs_sim[:, idxs]
    v1 = eigvecs_sorted[:, 0].real
    v1_energy_cluster0 = float(np.sum((np.abs(v1)**2) * (targets_idx==0)))

    metrics = {
        'retention_after_offset': retention_after_offset,
        'retention_absolute': retention_absolute,
        'hit_cap': hit_cap,
        'final_overlap': final_overlap,
        'final_energy_frac': final_energy_frac,
        'mean_edge_length': mean_edge_length,
        'mean_weight': mean_weight,
        'frac_within_cluster': frac_within_cluster,
        'edge_count': edge_count,
        'v1_energy_cluster0': v1_energy_cluster0,
        'eigvals_sim': eigvals_sorted
    }
    return metrics, similarity

# --------------------------- Parameter sweep (paired seeds) ---------------------------
# Grid and seeds (edit these for faster or deeper runs)
hole_H_list = [0.0, 0.3, 0.6, 0.9]    # BDNF hole magnitude
A_c_list = [0.0, 0.5, 1.5]            # cortisol amplitude
SPEC_list = [0.8, 0.9, 0.95]          # spectral scaling applied before dynamics
SEEDS = list(range(12))               # default 12 paired seeds; increase to 30 when ready

grid = list(itertools.product(hole_H_list, A_c_list, SPEC_list))
print("Grid size:", len(grid), "Paired seeds:", len(SEEDS))
rows_out = []
start_all = time.time()

# We'll use global somata variable inside run_memory_and_metrics; set each seed
for (h_val, Ac_val, spec_val) in grid:
    print("\n--- grid point hole_H=%.3f A_c=%.3f SPEC=%.3f" % (h_val, Ac_val, spec_val))
    # set globals for this grid point and remember old values
    old_h = hole_H; old_Ac = A_c; old_SPEC = SPEC_SCALING
    hole_H = float(h_val); A_c = float(Ac_val); SPEC_SCALING = float(spec_val)

    diffs_rand = []
    diffs_cluster = []
    nt_ret_rand = []; tr_ret_rand = []
    # paired runs
    for s in SEEDS:
        # deterministic somata + targets per seed
        rng_seed = default_rng(int(s))
        somata = rng_seed.random((N,2)) * L
        targets_idx = np.repeat(np.arange(M), N//M)
        if len(targets_idx) < N:
            extra = list(range(N - len(targets_idx)))
            targets_idx = np.concatenate([targets_idx, rng_seed.choice(np.arange(M), size=len(extra))])
        rng_seed.shuffle(targets_idx)

        # make patterns
        kpat = max(1, int(np.floor(pattern_sparsity * N)))
        rng_pat = default_rng(int(s) + 12345)
        rand_idx = rng_pat.choice(N, size=kpat, replace=False)
        p_rand = np.zeros(N); p_rand[rand_idx] = 1.0
        unique, counts = np.unique(targets_idx, return_counts=True)
        cluster0 = unique[np.argmax(counts)]
        cluster_idxs = np.where(targets_idx==cluster0)[0]
        if cluster_idxs.size >= kpat:
            pick = rng_pat.choice(cluster_idxs, size=kpat, replace=False)
            p_cluster = np.zeros(N); p_cluster[pick] = 1.0
        else:
            p_cluster = np.zeros(N); p_cluster[cluster_idxs] = 1.0

        # simulate paired growth with same seed
        W_nt, ct_nt, tr_nt = simulate_growth(somata, targets_idx, trauma_active=False, seed=s)
        W_tr, ct_tr, tr_tr = simulate_growth(somata, targets_idx, trauma_active=True, seed=s)

        # set global somata variable used by run_memory_and_metrics
        somata_global = somata  # used below inside function via closure (we'll pass in via global name)
        # run memory
        mnt_r, sim_nt_r = run_memory_and_metrics(W_nt, targets_idx, p_rand, seed_for_ei=s)
        mtr_r, sim_tr_r = run_memory_and_metrics(W_tr, targets_idx, p_rand, seed_for_ei=s)
        mnt_c, sim_nt_c = run_memory_and_metrics(W_nt, targets_idx, p_cluster, seed_for_ei=s)
        mtr_c, sim_tr_c = run_memory_and_metrics(W_tr, targets_idx, p_cluster, seed_for_ei=s)

        diffs_rand.append(mtr_r['retention_after_offset'] - mnt_r['retention_after_offset'])
        diffs_cluster.append(mtr_c['retention_after_offset'] - mnt_c['retention_after_offset'])
        nt_ret_rand.append(mnt_r['retention_after_offset'])
        tr_ret_rand.append(mtr_r['retention_after_offset'])

    # statistics (random pattern)
    dif = np.array(diffs_rand)
    mean_diff = float(np.mean(dif)); sd = float(np.std(dif, ddof=1)) if len(dif)>1 else 0.0
    n = len(dif)
    # bootstrap normal approx
    B = 2000
    rng_boot = default_rng(99999)
    boot_means = np.empty(B)
    for b in range(B):
        idx = rng_boot.integers(0, n, n)
        boot_means[b] = np.mean(dif[idx])
    boot_mu = float(np.mean(boot_means)); boot_se = float(np.std(boot_means, ddof=1))
    z = stats.norm.ppf(0.975)
    ci_low = boot_mu - z*boot_se; ci_high = boot_mu + z*boot_se
    # paired one-sample t-test on diffs
    tstat, pval = stats.ttest_1samp(dif, 0.0) if n>1 else (np.nan, np.nan)
    cohen_d = float(mean_diff / (sd + 1e-12)) if n>1 else np.nan

    rows_out.append({
        'hole_H': h_val, 'A_c': Ac_val, 'SPEC': spec_val, 'pattern': 'random',
        'mean_diff_ret': mean_diff, 'sd_diff': sd, 'n': n,
        'boot_ci_low': ci_low, 'boot_ci_high': ci_high, 'tstat': float(tstat), 'pval': float(pval),
        'cohen_d': cohen_d, 'nt_mean_ret': float(np.mean(nt_ret_rand)), 'tr_mean_ret': float(np.mean(tr_ret_rand))
    })

    # statistics (cluster pattern)
    dif = np.array(diffs_cluster)
    mean_diff = float(np.mean(dif)); sd = float(np.std(dif, ddof=1)) if len(dif)>1 else 0.0
    n = len(dif)
    rng_boot = default_rng(99999)
    boot_means = np.empty(B)
    for b in range(B):
        idx = rng_boot.integers(0, n, n)
        boot_means[b] = np.mean(dif[idx])
    boot_mu = float(np.mean(boot_means)); boot_se = float(np.std(boot_means, ddof=1))
    ci_low = boot_mu - z*boot_se; ci_high = boot_mu + z*boot_se
    tstat, pval = stats.ttest_1samp(dif, 0.0) if n>1 else (np.nan, np.nan)
    cohen_d = float(mean_diff / (sd + 1e-12)) if n>1 else np.nan

    rows_out.append({
        'hole_H': h_val, 'A_c': Ac_val, 'SPEC': spec_val, 'pattern': 'cluster',
        'mean_diff_ret': mean_diff, 'sd_diff': sd, 'n': n,
        'boot_ci_low': ci_low, 'boot_ci_high': ci_high, 'tstat': float(tstat), 'pval': float(pval),
        'cohen_d': cohen_d, 'nt_mean_ret': np.nan, 'tr_mean_ret': np.nan
    })

    # restore globals
    hole_H = old_h; A_c = old_Ac; SPEC_SCALING = old_SPEC

end_all = time.time()
print("Sweep finished in %.1f s" % (end_all - start_all))

# Save CSV
df = pd.DataFrame(rows_out)
csv_path = os.path.join(OUT, "sweep_retention_diffs.csv")
df.to_csv(csv_path, index=False)
print("Saved CSV to", csv_path)

# Make a sample heatmap for random pattern at SPEC=0.95 (if present)
for spec_v in SPEC_list:
    sub = df[(df.pattern=='random') & (np.isclose(df.SPEC, spec_v))]
    if sub.shape[0]==0: continue
    pivot = sub.pivot(index='hole_H', columns='A_c', values='mean_diff_ret')
    plt.figure(figsize=(6,5))
    data = pivot.values
    vmax = np.nanmax(np.abs(data)) if data.size>0 else 1.0
    plt.imshow(data, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='mean (TR-NT) retention diff (s)')
    plt.title(f'Random pattern — mean retention diff (SPEC={spec_v})')
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns)
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.xlabel('A_c'); plt.ylabel('hole_H')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"heatmap_random_SPEC{spec_v}.png"), dpi=150)
    plt.close()

print("All outputs (CSV + heatmaps) written to:", OUT)

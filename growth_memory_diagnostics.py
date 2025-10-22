# growth_memory_diagnostics.py
# Corrected chemo-tactic axon-growth + synaptogenesis + memory test
# Produces CSV summary tables and diagnostic figures.
# Requires: numpy, scipy, networkx, matplotlib, pandas
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import default_rng
import os
import pandas as pd

# --------------------------- User config ---------------------------
OUTDIR = "./data"
os.makedirs(OUTDIR, exist_ok=True)

RNG_SEED = 42
rng = default_rng(RNG_SEED)  # reproducible

# --------------------------- Parameters (baseline) ---------------------------
N = 80
L = 100.0                 # domain size [0,L] x [0,L]
dt_g = 0.5                # growth timestep
T_grow_steps = 2000       # number of growth steps
alpha0 = 0.6
D0 = 0.2
beta = 0.05
k_alpha = 1.2
k_D = 3.0
r_syn = 2.0
w0 = 1.0
a_B = 2.0
w_max = 3.0
gamma_dist = 0.0  # no distance penalty by default
tau = 10.0         # rate-model time constant
phi = np.tanh      # activation nonlinearity for rate model

# BDNF spatial template: two gaussian peaks
M = 2
A_m = [1.0, 1.0]
mu_m = [np.array([30., 50.]), np.array([70., 50.])]
sigma_m = [15.0, 15.0]

# BDNF time envelope (critical period)
t_pk = T_grow_steps / 3.0
sigma_B = T_grow_steps / 6.0

# Trauma schedule (cortisol pulse during critical period)
C_baseline = 0.01
A_c = 1.5
t_tr = t_pk            # trauma centered at BDNF peak
sigma_c = T_grow_steps / 30.0

# Local BDNF holes (trauma-induced)
n_holes = 3
hole_centers = [np.array([40., 40.]), np.array([60., 60.]), np.array([50., 30.])]
hole_rho = 6.0
hole_H = 0.9  # magnitude of negative blob (relative)

# Memory test params
dt_rate = 0.1
T_cue = 50.0
T_test = 400.0
cue_amplitude = 2.0
pattern_sparsity = 0.1  # fraction of neurons in pattern
S_thresh = 0.5

# E/I balance options
APPLY_EI = True
FRAC_INH = 0.1
INH_STRENGTH = 0.2
ROW_NORM_DESIRED = 1.0

SPEC_SCALING = 0.9  # scale spectral radius to this before dynamics

# --------------------------- Helper functions ---------------------------
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

def Xi_func(x, t, trauma_active=False, rng_local=None):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    if rng_local is None:
        rng_local = rng
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
        rng_local = rng
    S_B_t = np.exp(-((t_step - t_pk)**2) / (2.0 * sigma_B**2))
    b0_vals = B0_func(x)
    grad_b0 = grad_B0(x)
    xi_vals = Xi_func(x, t_step, trauma_active=trauma_active, rng_local=rng_local)
    B_vals = b0_vals * S_B_t + xi_vals
    grad_B = grad_b0 * S_B_t
    if trauma_active:
        # add hole gradients (repulsive)
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
    else:
        return C_baseline

def apply_ei_balance(W, seed, frac_inh=FRAC_INH, inh_strength=INH_STRENGTH, row_norm=ROW_NORM_DESIRED):
    rng_local = default_rng(int(seed) + 99999)
    Nloc = W.shape[0]
    n_inh = max(1, int(np.round(frac_inh * Nloc)))
    inh_idx = rng_local.choice(Nloc, size=n_inh, replace=False)
    Wm = W.copy().astype(float)
    # make inhibitory rows negative proportional to existing absolute weights
    for i in inh_idx:
        Wm[i, :] = -inh_strength * np.abs(Wm[i, :])
    # row-normalize by absolute sums
    row_abs = np.sum(np.abs(Wm), axis=1, keepdims=True)
    row_abs[row_abs == 0.0] = 1.0
    Wm = Wm / row_abs * row_norm
    # ensure inhibitory rows are negative
    for i in inh_idx:
        Wm[i, :] = -np.abs(Wm[i, :])
    return Wm, inh_idx

# --------------------------- Simulation functions ---------------------------
def simulate_growth(somata, targets_idx, trauma_active=False, seed=None):
    """Run growth and synaptogenesis returning W, connect_time, trajectories, somata, targets_idx."""
    if seed is None:
        rng_local = default_rng(RNG_SEED)
    else:
        rng_local = default_rng(int(seed))
    Nloc = somata.shape[0]
    target_centers = np.array([mu_m[idx] for idx in targets_idx])

    # initialize tips
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
        nonzero = dists > 1e-8
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
                if j == i or connected_flag[i,j]:
                    continue
                B_local = B_vals[i] if np.ndim(B_vals) > 0 else B_vals
                B_max_ref = max(A_m)
                B_norm = float(B_local / (B_max_ref + 1e-9))
                delta_ij = np.linalg.norm(tips[i] - somata[j])
                w_ij = w0 * (1.0 + a_B * B_norm) * np.exp(-gamma_dist * delta_ij)
                if apply_weight_cap:
                    w_ij = np.clip(w_ij, 0.0, w_max)
                W[i, j] = w_ij
                connected_flag[i, j] = True
                connect_time[i, j] = n

    return W, connect_time, trajectories

def run_memory_and_metrics(W_raw, targets_idx, pattern_vec, seed_for_ei=0):
    # apply E/I
    W_work = W_raw.copy().astype(float)
    if APPLY_EI:
        W_work, inh_idx = apply_ei_balance(W_work, seed_for_ei, frac_inh=FRAC_INH, inh_strength=INH_STRENGTH, row_norm=ROW_NORM_DESIRED)

    # spectral scaling
    eigs = np.linalg.eigvals(W_work)
    spec = max(np.abs(eigs)) if eigs.size>0 else 0.0
    if spec > 0:
        W_sim = W_work * (SPEC_SCALING / spec)
    else:
        W_sim = W_work.copy()

    # dynamics
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

    # retention
    post_cue = similarity[cue_steps:]
    drop_idxs = np.where(post_cue < S_thresh)[0]
    if drop_idxs.size > 0:
        retention_after_offset = drop_idxs[0] * dt_rate
        retention_absolute = (cue_steps + drop_idxs[0]) * dt_rate
        hit_cap = False
    else:
        retention_after_offset = (T_test - T_cue)
        retention_absolute = T_test
        hit_cap = True

    window_steps = int(min(50.0 / dt_rate, max(1, int(0.2 * time_steps))))
    final_overlap = float(np.mean(similarity[-window_steps:]))
    pat_idx = (pattern_vec > 0)
    energy_in_pattern = np.sum(x[pat_idx]**2) if np.any(pat_idx) else 0.0
    total_energy = np.sum(x**2)
    final_energy_frac = float(energy_in_pattern / total_energy) if total_energy > 1e-12 else 0.0

    # graph metrics (raw W)
    G = nx.DiGraph()
    for i in range(W_raw.shape[0]):
        G.add_node(i)
    rows, cols = np.where(W_raw > 0)
    for i,j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(W_raw[int(i), int(j)]))
    degrees = np.array([d for _, d in G.degree()])
    avg_degree = float(degrees.mean()) if degrees.size>0 else 0.0
    clustering = float(nx.average_clustering(G.to_undirected())) if G.number_of_nodes()>0 else 0.0
    eigvals_full = np.linalg.eigvals(W_raw)
    spec_rad = float(max(np.abs(eigvals_full))) if eigvals_full.size>0 else 0.0

    # within-cluster edge fraction & edge stats
    if rows.size>0:
        same_cluster = np.sum(targets_idx[rows] == targets_idx[cols])
        frac_within_cluster = float(same_cluster / len(rows))
        edge_count = len(rows)
        # compute mean edge length and mean weight
        dists = np.linalg.norm(somata[rows] - somata[cols], axis=1)
        mean_edge_length = float(np.mean(dists))
        mean_weight = float(np.mean(W_raw[rows, cols]))
    else:
        frac_within_cluster = 0.0
        edge_count = 0
        mean_edge_length = np.nan
        mean_weight = np.nan

    # eigenmodes of W_sim for diagnostics
    eigvals_sim, eigvecs_sim = np.linalg.eig(W_sim)
    idxs = np.argsort(-np.abs(eigvals_sim))
    eigvals_sorted = eigvals_sim[idxs]
    eigvecs_sorted = eigvecs_sim[:, idxs]
    # leading eigenvector localization
    v1 = eigvecs_sorted[:, 0].real
    v1_energy_cluster0 = float(np.sum((np.abs(v1)**2) * (targets_idx==0)))

    metrics = {
        'retention_after_offset': retention_after_offset,
        'retention_absolute': retention_absolute,
        'hit_cap': hit_cap,
        'final_overlap': final_overlap,
        'final_energy_frac': final_energy_frac,
        'avg_degree': avg_degree,
        'clustering': clustering,
        'spec_rad': spec_rad,
        'frac_within_cluster': frac_within_cluster,
        'edge_count': edge_count,
        'mean_edge_length': mean_edge_length,
        'mean_weight': mean_weight,
        'v1_energy_cluster0': v1_energy_cluster0,
        'eigvals_sim': eigvals_sorted,      # array
        'eigvecs_sim': eigvecs_sorted       # matrix
    }

    return metrics, similarity

# --------------------------- Main paired run (neurotypical vs trauma) ---------------------------

def main_run(seed=RNG_SEED, save_tables=True):
    rng_master = default_rng(int(seed))
    # fixed somata and targets so NT vs Trauma are comparable
    somata_local = rng_master.random((N,2)) * L

    # targets assign similar to original: repeated centers then shuffle
    targets_idx = np.repeat(np.arange(M), N//M)
    if len(targets_idx) < N:
        extra = list(range(N - len(targets_idx)))
        targets_idx = np.concatenate([targets_idx, rng_master.choice(np.arange(M), size=len(extra))])
    rng_master.shuffle(targets_idx)

    # run growth for both conditions with same somata/targets and same RNG seed for paired growth RNG
    W_nt, ct_nt, tr_nt = simulate_growth(somata_local, targets_idx, trauma_active=False, seed=seed)
    W_tr, ct_tr, tr_tr = simulate_growth(somata_local, targets_idx, trauma_active=True, seed=seed)

    # pick two patterns: random and cluster-local
    k_pattern = max(1, int(np.floor(pattern_sparsity * N)))
    rng_pat = default_rng(seed + 12345)
    rand_idx = rng_pat.choice(N, size=k_pattern, replace=False)
    p_random = np.zeros(N); p_random[rand_idx] = 1.0

    # cluster-local: find largest cluster
    unique, counts = np.unique(targets_idx, return_counts=True)
    cluster0 = unique[np.argmax(counts)]
    cluster_idxs = np.where(targets_idx == cluster0)[0]
    if cluster_idxs.size >= k_pattern:
        pick = rng_pat.choice(cluster_idxs, size=k_pattern, replace=False)
        p_cluster = np.zeros(N); p_cluster[pick] = 1.0
    else:
        p_cluster = np.zeros(N); p_cluster[cluster_idxs] = 1.0

    patterns = {'random': p_random, 'cluster': p_cluster}

    # run memory tests and assemble metrics table rows
    graph_rows = []
    memory_rows = []
    eig_rows = []
    edge_rows = []

    for cond_name, W in [('neurotypical', W_nt), ('trauma', W_tr)]:
        # graph metrics independent of pattern (we compute once)
        rows, cols = np.where(W > 0)
        edge_count = len(rows)
        mean_w = float(np.mean(W[rows, cols])) if edge_count>0 else np.nan
        mean_len = float(np.mean(np.linalg.norm(somata_local[rows]-somata_local[cols], axis=1))) if edge_count>0 else np.nan
        G = nx.DiGraph()
        for i in range(N):
            G.add_node(i)
        for i,j in zip(rows, cols):
            G.add_edge(int(i), int(j), weight=float(W[int(i),int(j)]))
        degrees = np.array([d for _, d in G.degree()])
        avg_deg = float(degrees.mean()) if degrees.size>0 else 0.0
        clustering = float(nx.average_clustering(G.to_undirected())) if G.number_of_nodes()>0 else 0.0
        eigvals_full = np.linalg.eigvals(W)
        spec_rad_unscaled = float(max(np.abs(eigvals_full))) if eigvals_full.size>0 else 0.0

        graph_rows.append({
            'condition': cond_name,
            'N': N,
            'avg_degree': avg_deg,
            'clustering': clustering,
            'spec_rad_unscaled': spec_rad_unscaled,
            'edge_count': edge_count,
            'mean_weight': mean_w,
            'mean_edge_length': mean_len
        })

        for pname, pvec in patterns.items():
            metrics, sim = run_memory_and_metrics(W, targets_idx, pvec, seed_for_ei=seed)
            memory_rows.append({
                'condition': cond_name,
                'pattern': pname,
                'k_pattern': int(np.sum(pvec)),
                'retention_after_offset': metrics['retention_after_offset'],
                'retention_absolute': metrics['retention_absolute'],
                'hit_cap': metrics['hit_cap'],
                'final_overlap': metrics['final_overlap'],
                'final_energy_frac': metrics['final_energy_frac'],
                'avg_degree': metrics['avg_degree'],
                'clustering': metrics['clustering'],
                'spec_rad': metrics['spec_rad'],
                'frac_within_cluster': metrics['frac_within_cluster'],
                'edge_count': metrics['edge_count'],
                'mean_edge_length': metrics['mean_edge_length'],
                'mean_weight': metrics['mean_weight'],
                'v1_energy_cluster0': metrics['v1_energy_cluster0']
            })

            # save similarity traces
            np.savez(os.path.join(OUTDIR, f"similarity_{cond_name}_{pname}.npz"), sim=sim)

            # eigenvalues table (top 10)
            eigvals_sorted = metrics['eigvals_sim']
            for i_e in range(min(10, len(eigvals_sorted))):
                eig_rows.append({
                    'condition': cond_name,
                    'pattern': pname,
                    'mode_idx': i_e+1,
                    'eigval': float(eigvals_sorted[i_e]),
                    'abs_eigval': float(np.abs(eigvals_sorted[i_e]))
                })

        # edge stats row
        edge_rows.append({
            'condition': cond_name,
            'edge_count': edge_count,
            'mean_weight': mean_w,
            'mean_length': mean_len
        })

    # Save CSV tables
    if save_tables:
        pd.DataFrame(graph_rows).to_csv(os.path.join(OUTDIR, "graph_metrics.csv"), index=False)
        pd.DataFrame(memory_rows).to_csv(os.path.join(OUTDIR, "memory_metrics.csv"), index=False)
        pd.DataFrame(eig_rows).to_csv(os.path.join(OUTDIR, "eigenmodes.csv"), index=False)
        pd.DataFrame(edge_rows).to_csv(os.path.join(OUTDIR, "edge_stats.csv"), index=False)
        # save raw networks
        np.savez(os.path.join(OUTDIR, "network_neurotypical.npz"), W=W_nt, somata=somata_local, targets=targets_idx, connect_time=ct_nt)
        np.savez(os.path.join(OUTDIR, "network_trauma.npz"), W=W_tr, somata=somata_local, targets=targets_idx, connect_time=ct_tr)

    # Figures: trajectories, adjacency, memory similarity (neurotypical & trauma)
    def plot_trajectories(trajs, fname):
        plt.figure(figsize=(6,5))
        for ii in range(trajs.shape[1]):
            traj = trajs[:, ii, :]
            plt.plot(traj[:,0], traj[:,1], linewidth=0.8)
            plt.scatter(traj[0,0], traj[0,1], s=10)
        plt.title("Tip trajectories (subset)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.xlim(0, L); plt.ylim(0, L)
        plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()

    plot_trajectories(tr_nt, os.path.join(OUTDIR, "trajectories_neurotypical.png"))
    plot_trajectories(tr_tr, os.path.join(OUTDIR, "trajectories_trauma.png"))

    def plot_network(Wmat, somata_arr, fname):
        plt.figure(figsize=(6,6))
        pos = {i: tuple(somata_arr[i]) for i in range(N)}
        nx.draw_networkx_nodes(nx.DiGraph(), pos, node_size=40)  # only nodes
        rows, cols = np.where(Wmat > 0)
        weights = Wmat[rows, cols] if rows.size>0 else np.array([])
        if weights.size>0:
            wmin = weights.min()
            wrange = np.ptp(weights)
            if wrange < 1e-9:
                wnorm = np.zeros_like(weights)
            else:
                wnorm = (weights - wmin) / wrange
            alphas = 0.2 + 0.8 * wnorm
            for (u,v,w), a in zip(zip(rows,cols, weights), alphas):
                x_coords = [somata_arr[u,0], somata_arr[v,0]]
                y_coords = [somata_arr[u,1], somata_arr[v,1]]
                plt.plot(x_coords, y_coords, linewidth=0.6, alpha=float(a))
        plt.title("Final network (nodes at soma positions)")
        plt.xlim(0, L); plt.ylim(0, L)
        plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()

    plot_network(W_nt, somata_local, os.path.join(OUTDIR, "final_network_neurotypical.png"))
    plot_network(W_tr, somata_local, os.path.join(OUTDIR, "final_network_trauma.png"))

    # memory similarity plots (overlay patterns)
    for pname in patterns:
        data_nt = np.load(os.path.join(OUTDIR, f"similarity_neurotypical_{pname}.npz"))['sim']
        data_tr = np.load(os.path.join(OUTDIR, f"similarity_trauma_{pname}.npz"))['sim']
        t = np.linspace(0, T_test, len(data_nt))
        plt.figure(figsize=(7,4))
        plt.plot(t, data_nt, label='Neurotypical')
        plt.plot(t, data_tr, label='Trauma')
        plt.axvline(T_cue, linestyle='--', color='gray')
        plt.xlabel('Time (s)'); plt.ylabel('Cosine similarity')
        plt.title(f'Similarity — pattern={pname}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"similarity_overlay_{pname}.png")); plt.close()

    print("Saved CSV tables and figures to", OUTDIR)
    return

if __name__ == "__main__":
    main_run(seed=RNG_SEED, save_tables=True)

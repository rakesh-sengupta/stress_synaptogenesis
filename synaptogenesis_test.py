# Corrected chemo-tactic axon-growth + synaptogenesis + memory test script
# Requires: numpy, scipy, networkx, matplotlib
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import default_rng
import os

rng = default_rng(42)  # reproducible

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

# Local BDNF holes (trauma-induced) -- set when trauma=True
n_holes = 3
# keep these fixed per-run (original script uses fixed coords; we keep that)
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

# Flags
trauma = False    # toggle to produce trauma vs neurotypical
apply_weight_cap = True

# E/I balance options (set APPLY_EI = True to apply E/I procedure)
APPLY_EI = True
FRAC_INH = 0.1
INH_STRENGTH = 0.2
ROW_NORM_DESIRED = 1.0

SPEC_SCALING = 0.9  # scale spectral radius to this before dynamics

# --------------------------- Helper functions ---------------------------
def B0_func(x):
    """Base spatial BDNF profile B0(x) as sum of Gaussians."""
    x = np.asarray(x)
    # handle both (n,2) and (2,) inputs
    if x.ndim == 1:
        x = x[None, :]
    val = np.zeros(x.shape[0])
    for A, mu, sig in zip(A_m, mu_m, sigma_m):
        diff = x - mu
        d2 = np.sum(diff**2, axis=-1)
        val += A * np.exp(-d2 / (2.0 * sig**2))
    return val if val.size > 1 else float(val)

def grad_B0(x):
    """Gradient of B0(x) (analytic). Returns shape (n,2)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    grad = np.zeros_like(x)
    for A, mu, sig in zip(A_m, mu_m, sigma_m):
        diff = x - mu
        exp_part = np.exp(-np.sum(diff**2, axis=-1) / (2.0 * sig**2))
        grad += A * exp_part[:, None] * (-(diff) / (sig**2))
    return grad

def Xi_func(x, t, trauma_active=False):
    """Perturbation field Xi(x,t): returns small noise + holes value (no gradient)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    # small spatial noise term (mean zero)
    noise = 0.02 * rng.standard_normal(size=(x.shape[0],))
    val = noise.copy()
    if trauma_active:
        # negative gaussian holes (value)
        for center in hole_centers:
            diff = x - center
            d2 = np.sum(diff**2, axis=-1)
            val += -hole_H * np.exp(-d2 / (2.0 * hole_rho**2))
    return val if val.size > 1 else float(val)

def B_field_and_grad(x, t_step, trauma_active=False):
    """
    Return B(x,t) and gradient at positions x (x: (n,2)).
    IMPORTANT: this version includes the gradient of the hole perturbations,
    so holes *repel* growth cones (biologically plausible).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    S_B_t = np.exp(-((t_step - t_pk)**2) / (2.0 * sigma_B**2))
    # B0 and its gradient
    b0_vals = B0_func(x)                  # shape (n,)
    grad_b0 = grad_B0(x)                  # shape (n,2)
    # xi values (noise + holes value)
    xi_vals = Xi_func(x, t_step, trauma_active=trauma_active)  # shape (n,)
    # combined B value
    B_vals = b0_vals * S_B_t + xi_vals
    # gradient: grad(B0)*S_B_t + grad(Xi) where Xi arises from holes (we compute analytic grad)
    grad_B = grad_b0 * S_B_t  # (n,2)
    if trauma_active:
        # add gradient from fixed hole centers (holes were defined globally)
        for center in hole_centers:
            diff = x - center                 # (n,2)
            d2 = np.sum(diff**2, axis=1)      # (n,)
            exp_part = np.exp(-d2 / (2.0 * hole_rho**2))
            # grad of (-H * exp(-d2/(2 rho^2))) = H * exp_part * (x-center) / rho^2
            grad_h = (hole_H * exp_part)[:, None] * (diff / (hole_rho**2))
            grad_B += grad_h * 1.0            # multiply by S_B_t if holes should be modulated in time (optional)
    return (B_vals if B_vals.size > 1 else float(B_vals)), grad_B

def cortisol(t_step):
    """Cortisol schedule: baseline + trauma pulse (Gaussian)"""
    if trauma:
        return C_baseline + A_c * np.exp(-((t_step - t_tr)**2) / (2.0 * sigma_c**2))
    else:
        return C_baseline

# --------------------------- E/I BALANCE helper ---------------------------
def apply_ei_balance(W, seed, frac_inh=FRAC_INH, inh_strength=INH_STRENGTH, row_norm=ROW_NORM_DESIRED):
    """
    Choose a fraction of neurons as inhibitory (randomized by seed),
    set their outgoing rows negative with magnitude inh_strength, then
    row-normalize absolute outgoing strengths to row_norm.
    Returns modified W and inhibitory indices.
    """
    rng_local = default_rng(int(seed) + 99999)
    Nloc = W.shape[0]
    inh_idx = rng_local.choice(Nloc, size=int(frac_inh * Nloc), replace=False)
    Wm = W.copy()
    for i in inh_idx:
        Wm[i, :] *= -inh_strength
    row_abs = np.sum(np.abs(Wm), axis=1, keepdims=True)
    row_abs[row_abs == 0] = 1.0
    Wm = Wm / row_abs * row_norm
    # ensure inhibitory rows are negative
    for i in inh_idx:
        Wm[i, :] *= -1.0
    return Wm, inh_idx

# --------------------------- Initialize neurons and tips ---------------------------
# Place neuron somata randomly, then assign half to each target center
somata = rng.random((N, 2)) * L
# assign target centers - choose nearest peak centers by index (or random half-half)
targets_idx = np.repeat(np.arange(M), N//M)
if len(targets_idx) < N:
    extra = list(range(N - len(targets_idx)))
    targets_idx = np.concatenate([targets_idx, rng.choice(np.arange(M), size=len(extra))])
rng.shuffle(targets_idx)
target_centers = np.array([mu_m[idx] for idx in targets_idx])

# Initialize growth tips slightly offset from soma
tips = somata + 0.5 * (rng.random((N,2)) - 0.5)

# to record trajectories for a subset of tips for plotting
n_trace = min(12, N)
trace_idx = np.arange(n_trace)
trajectories = np.zeros((T_grow_steps+1, n_trace, 2))
trajectories[0,:,:] = tips[trace_idx]

# adjacency and record arrays
W = np.zeros((N, N))
connect_time = np.full((N, N), np.inf)
connected_flag = np.zeros((N, N), dtype=bool)

# build KDTree for somata (for quick proximity queries)
tree = cKDTree(somata)

# --------------------------- Growth loop ---------------------------
for n in range(T_grow_steps):
    # compute B and gradient at tip positions (vectorized)
    B_vals, grad_B = B_field_and_grad(tips, n, trauma_active=(trauma and True))
    # cortisol at this time
    C_t = cortisol(n)
    # parameters modulated by cortisol
    alpha_t = alpha0 * np.exp(-k_alpha * C_t)
    D_t = D0 * (1.0 + k_D * C_t)
    # compute persistence unit vectors p for each tip
    diff_to_target = target_centers - tips
    dists = np.linalg.norm(diff_to_target, axis=1)
    p = np.zeros_like(diff_to_target)
    nonzero = dists > 1e-8
    p[nonzero] = (diff_to_target[nonzero] / dists[nonzero][:,None])
    # stochastic increments
    xi = rng.standard_normal(size=(N,2))
    tips = tips + dt_g * (alpha_t * grad_B + beta * p) + np.sqrt(2.0 * D_t * dt_g) * xi
    # keep tips inside the domain (reflecting boundary)
    tips = np.minimum(np.maximum(tips, 0.0), L)
    # record trajectories for subset
    trajectories[n+1,:,:] = tips[trace_idx]
    # proximity check for synapse formation
    neighbors_list = tree.query_ball_point(tips, r_syn)
    for i, neighs in enumerate(neighbors_list):
        if not neighs:
            continue
        # exclude self-connections and process each neighbor explicitly
        for j in neighs:
            if j == i:
                continue
            if connected_flag[i, j]:
                continue
            # assign weight based on local BDNF at tip i
            # B_vals may be scalar if single tip; handle that
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

# Build directed network object for metrics/plotting
G = nx.DiGraph()
for i in range(N):
    G.add_node(i, pos=tuple(somata[i]))

# add edges weighted
for i in range(N):
    for j in range(N):
        if W[i, j] > 0:
            G.add_edge(i, j, weight=W[i,j])

# --------------------------- Report simple graph metrics ---------------------------
degrees = np.array([d for _, d in G.degree()])
avg_degree = degrees.mean()
clustering = nx.average_clustering(G.to_undirected())
# spectral radius
eigvals = np.linalg.eigvals(W)
spec_rad = max(np.abs(eigvals)) if eigvals.size>0 else 0.0
print("Graph metrics: N =", N)
print("Average degree:", avg_degree)
print("Average clustering (undirected):", clustering)
print("Spectral radius of W:", spec_rad)

# --------------------------- Memory retention test ---------------------------
# Create a random sparse pattern p
k_pattern = max(1, int(np.floor(pattern_sparsity * N)))
pattern_idx = rng.choice(np.arange(N), size=k_pattern, replace=False)
p_vec = np.zeros(N)
p_vec[pattern_idx] = 1.0

# Integrate rate model using simple Euler method, but first apply optional E/I balance
W_for_dynamics = W.copy()
if APPLY_EI:
    W_for_dynamics, inh_idx = apply_ei_balance(W_for_dynamics, seed=42, frac_inh=FRAC_INH, inh_strength=INH_STRENGTH, row_norm=ROW_NORM_DESIRED)
# scale spectral radius to SPEC_SCALING
eigvals_dyn = np.linalg.eigvals(W_for_dynamics)
spec_rad_dyn = max(np.abs(eigvals_dyn)) if eigvals_dyn.size>0 else 0.0
if spec_rad_dyn > 0:
    W_sim = W_for_dynamics * (SPEC_SCALING / spec_rad_dyn)
else:
    W_sim = W_for_dynamics.copy()

time_steps = int(np.ceil(T_test / dt_rate))
x = np.zeros(N)
similarity = np.zeros(time_steps)
time_arr = np.linspace(0, T_test, time_steps)
cue_steps = int(np.ceil(T_cue / dt_rate))

for t_idx in range(time_steps):
    Icue = cue_amplitude * p_vec if t_idx < cue_steps else np.zeros(N)
    dx = (-x + W_sim.dot(phi(x)) + Icue) * (dt_rate / tau)
    x = x + dx
    # compute similarity
    norm_x = np.linalg.norm(x) + 1e-12
    norm_p = np.linalg.norm(p_vec) + 1e-12
    similarity[t_idx] = np.dot(x, p_vec) / (norm_x * norm_p)

# compute retention times: time after cue offset, absolute time, and hit_cap
post_cue = similarity[cue_steps:]
drop_idxs = np.where(post_cue < S_thresh)[0]
if drop_idxs.size > 0:
    retention_after_offset = drop_idxs[0] * dt_rate
    retention_absolute = time_arr[cue_steps + drop_idxs[0]]
    hit_cap = False
else:
    retention_after_offset = (T_test - T_cue)
    retention_absolute = T_test
    hit_cap = True

# final-window averaging: last window is min(50s or 20% of time)
window_steps = int(min(50.0 / dt_rate, max(1, int(0.2 * time_steps))))
final_overlap = float(np.mean(similarity[-window_steps:]))
# compute energy fraction in pattern (use x stored at last step)
pat_idx = (p_vec > 0)
energy_in_pattern = np.sum(x[pat_idx]**2) if np.any(pat_idx) else 0.0
total_energy = np.sum(x**2)
final_energy_frac = float(energy_in_pattern / total_energy) if total_energy > 1e-12 else 0.0

print(f"Pattern size: {k_pattern} neurons.")
print(f"Retention (after cue offset): {retention_after_offset:.3f} s")
print(f"Retention (absolute): {retention_absolute:.3f} s")
print(f"Hit cap (survived to T_test): {hit_cap}")
print(f"Final overlap (mean last window): {final_overlap:.4f}")
print(f"Final energy fraction (pattern): {final_energy_frac:.4f}")

# --------------------------- Diagnostics (suggested) ---------------------------
# within-cluster edge fraction
rows, cols = np.where(W > 0)
if len(rows) > 0:
    same_cluster = np.sum(targets_idx[rows] == targets_idx[cols])
    frac_within_cluster = same_cluster / len(rows)
else:
    frac_within_cluster = 0.0
print(f"Within-cluster edge fraction: {frac_within_cluster:.3f}")

# leading eigenvector localization (of signed/scaled W used in dynamics)
eigvals_s, eigvecs_s = np.linalg.eig(W_sim)
idx = np.argsort(-np.abs(eigvals_s))
v1 = eigvecs_s[:, idx[0]].real
v1_energy_cluster0 = np.sum((np.abs(v1)**2) * (targets_idx==0))
print(f"v1 energy on cluster 0 (sum of squared coeffs): {v1_energy_cluster0:.4f}")

# --------------------------- Plots ---------------------------
fig_dir = "./data"
os.makedirs(fig_dir, exist_ok=True)

# 1) Trajectories for subset of tips
plt.figure(figsize=(6,5))
for ii in range(n_trace):
    traj = trajectories[:, ii, :]
    plt.plot(traj[:,0], traj[:,1], linewidth=0.8)
    plt.scatter(traj[0,0], traj[0,1], s=10)
plt.title("Tip trajectories (subset)")
plt.xlabel("x"); plt.ylabel("y")
plt.xlim(0, L); plt.ylim(0, L)
traj_path = os.path.join(fig_dir, "trajectories_neurotypical.png")
plt.savefig(traj_path, dpi=150, bbox_inches="tight")
plt.close()

# 2) Final network adjacency on soma positions
plt.figure(figsize=(6,6))
pos = {i: tuple(somata[i]) for i in range(N)}
nx.draw_networkx_nodes(G, pos, node_size=40)
weights = np.array([d['weight'] for _,_,d in G.edges(data=True)])
if weights.size > 0:
    wmin = weights.min()
    wrange = np.ptp(weights)
    if wrange < 1e-9:
        wnorm = np.zeros_like(weights)
    else:
        wnorm = (weights - wmin) / wrange
    alphas = 0.2 + 0.8 * wnorm
    for (u, v, d), a in zip(G.edges(data=True), alphas):
        x_coords = [somata[u, 0], somata[v, 0]]
        y_coords = [somata[u, 1], somata[v, 1]]
        plt.plot(x_coords, y_coords, linewidth=0.6, alpha=float(a))
plt.title("Final network (nodes at soma positions)")
plt.xlim(0, L); plt.ylim(0, L)
adj_path = os.path.join(fig_dir, "final_network_neurotypical.png")
plt.savefig(adj_path, dpi=150, bbox_inches="tight")
plt.close()

# 3) Memory similarity vs time
plt.figure(figsize=(6,4))
plt.plot(time_arr, similarity, linewidth=1.2)
plt.axvline(T_cue, linestyle='--')
plt.xlabel("Time"); plt.ylabel("Cosine similarity to pattern")
plt.title("Memory retention (similarity)")
sim_path = os.path.join(fig_dir, "memory_similarity_neurotypical.png")
plt.savefig(sim_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved figures:")
print(traj_path)
print(adj_path)
print(sim_path)

# Save a copy of the adjacency matrix and weights for later analysis
np.savez(os.path.join(fig_dir, "w_and_somata_neurotypical.npz"), W=W, somata=somata, connect_time=connect_time, targets=target_centers)

# Save the script for convenience
with open(os.path.join(fig_dir, "growth_model_readme_neurotypical.txt"), "w") as f:
    f.write("Output files: trajectories_neurotypical.png, final_network_trauma.png, memory_similarity_trauma.png, w_and_somata_trauma.npz\n")

print("All done. Figures and data saved to ./data")

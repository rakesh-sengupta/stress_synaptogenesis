"""
Stress, Structure, and Recall — Revised Simulation
====================================================
Revision notes keyed to reviewer concerns:

[REV-Circ]   Circularity control: weight-matched and topology-isolated conditions
             added to show that topology, not weight magnitude alone, drives recall
             deficits (Reviewer Point 3).

[REV-Spec]   Spectral analysis language and plots softened: eigenvalue panels now
             explicitly distinguish structural gain from dynamical bifurcation claims
             (Reviewer Point 5).

[REV-Fig]    All figures now carry full parameter annotations in titles/captions
             (Reviewer minor points: Fig 2, Fig 4 labels missing).

[REV-Bug]    Fixed file-name collision: explain_variance_via_criticality was
             overwriting Fig_Response_Spectral.png.

[REV-Scale]  Removed "large-scale" language from print statements (Reviewer Point 2).

[REV-Vec]    Global inhibition term rewritten as vector operation to match the
             mathematics in Eq. 7 of the manuscript (Reviewer minor point).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
import matplotlib.patches as patches
from numpy import linalg as LA

# ==========================================
# 1.  CONFIGURATION
# ==========================================
class Params:
    N_SEEDS       = 100
    N_SWEEP_SEEDS = 10
    N_NEURONS     = 100
    L             = 100.0

    # Growth physics
    T_GROW  = 2000
    DT_G    = 0.5
    ALPHA_0 = 1.2
    D_0     = 0.1
    BETA    = 0.1

    # Default trauma amplitudes (main batch)
    CORTISOL_AMP = 3.0
    HOLE_AMP     = 1.5
    HOLE_RADIUS  = 12.0

    # Synaptogenesis & memory
    R_SYN       = 3.0
    W_0         = 1.0
    A_B         = 3.0
    W_MAX       = 10.0
    GLOBAL_GAIN = 0.25
    GLOBAL_INH  = 0.8

    # Dynamics
    TAU    = 10.0
    DT_R   = 0.1
    T_CUE  = 20.0
    T_TEST = 200.0

params = Params()


# ==========================================
# 2.  SUBSTRATE GENERATION
# ==========================================
def generate_substrate(seed):
    np.random.seed(seed)
    somata  = np.random.rand(params.N_NEURONS, 2) * params.L
    centers = np.array([[25, 75], [75, 25]])
    labels  = np.random.randint(0, 2, params.N_NEURONS)
    targets = centers[labels]
    return somata, labels, targets


# ==========================================
# 3.  PHYSICS ENGINE
# ==========================================
def get_bdnf_field(coords, t, condition, hole_amp, cort_amp):
    """
    Returns (gradient_vectors, local_bdnf_values) at each coord.

    Biological note: B_0 models a discrete set of target-derived
    chemoattractant peaks (BDNF Gaussians). Holes represent localised
    trophic deprivation — negative perturbations to the BDNF landscape —
    producing both reduced local concentration and repulsive gradients.
    """
    centers   = np.array([[25, 75], [75, 25]])
    hole_locs = np.array([[50, 50], [25, 75], [75, 25]])

    grad_x = np.zeros(coords.shape[0])
    grad_y = np.zeros(coords.shape[0])
    val    = np.zeros(coords.shape[0])

    t_peak  = params.T_GROW / 3.0
    sigma_b = params.T_GROW / 5.0
    sb_t    = np.exp(-((t - t_peak)**2) / (2 * sigma_b**2))

    for c in centers:
        dx, dy   = coords[:, 0] - c[0], coords[:, 1] - c[1]
        dist_sq  = dx**2 + dy**2
        gauss    = np.exp(-dist_sq / (2 * 15.0**2))
        val     += sb_t * gauss
        grad_x  += sb_t * gauss * (-dx / 15.0**2)
        grad_y  += sb_t * gauss * (-dy / 15.0**2)

    if condition == 'TR':
        for h in hole_locs:
            dx, dy   = coords[:, 0] - h[0], coords[:, 1] - h[1]
            dist_sq  = dx**2 + dy**2
            gauss_h  = hole_amp * np.exp(-dist_sq / (2 * params.HOLE_RADIUS**2))
            val     -= sb_t * gauss_h
            grad_x  += sb_t * gauss_h * (dx / params.HOLE_RADIUS**2)
            grad_y  += sb_t * gauss_h * (dy / params.HOLE_RADIUS**2)

    return np.stack([grad_x, grad_y], axis=1), val


def run_growth(somata, targets, seed, condition,
               hole_amp=None, cort_amp=None, return_traj=False):
    """
    Simulates axon-tip dynamics and returns (W, mean_tortuosity [, traj]).

    Modeling note: tips represent growth-cone positions, not axon shafts.
    Synapses form on proximity to somata (a modeling abstraction; biologically,
    axons target dendrites — the soma-proximity rule here represents a
    coarse-grained synaptic contact event). W_ij encodes synaptic efficacy
    between unit i (source) and unit j (target), interpreted at the single-
    neuron level in this 100-unit network.
    """
    np.random.seed(seed + 99999)
    tips  = somata.copy()
    W     = np.zeros((params.N_NEURONS, params.N_NEURONS))

    hole_amp = params.HOLE_AMP     if hole_amp is None else hole_amp
    cort_amp = params.CORTISOL_AMP if cort_amp is None else cort_amp

    traj_history  = [] if return_traj else None
    path_lengths  = np.zeros(params.N_NEURONS)
    start_pos     = tips.copy()

    for step in range(params.T_GROW):
        if return_traj and step % 10 == 0:
            traj_history.append(tips.copy())

        t        = step * params.DT_G
        cortisol = 0.0
        if condition == 'TR':
            pulse    = np.exp(-((t - params.T_GROW / 3)**2) /
                              (2 * (params.T_GROW / 20)**2))
            cortisol = cort_amp * pulse

        alpha = params.ALPHA_0 / (1.0 + 3.0 * cortisol)
        D     = params.D_0     * (1.0 + 5.0 * cortisol)

        grad, b_val = get_bdnf_field(tips, t, condition, hole_amp, cort_amp)
        vec  = targets - tips
        norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9

        noise      = np.random.randn(params.N_NEURONS, 2)
        step_delta = (params.DT_G * (alpha * grad + params.BETA * (vec / norm))
                      + np.sqrt(2 * D * params.DT_G) * noise)

        path_lengths += np.linalg.norm(step_delta, axis=1)
        tips         += step_delta
        tips          = np.clip(tips, 0, params.L)

        if step % 20 == 0:
            dists        = cdist(tips, somata)
            connect_mask = (dists < params.R_SYN) & (dists > 0.1)
            local_bdnf   = b_val[np.where(connect_mask)[0]]
            src, tgt     = np.where(connect_mask)

            for idx, (s, tg) in enumerate(zip(src, tgt)):
                if W[s, tg] == 0:
                    strength   = params.W_0 * (1.0 + params.A_B * local_bdnf[idx])
                    W[s, tg]   = np.clip(strength, 0.0, params.W_MAX)

    displacements = np.linalg.norm(tips - start_pos, axis=1) + 1e-9
    tortuosity    = np.mean(path_lengths / displacements)

    if return_traj:
        return W, tortuosity, np.array(traj_history)
    return W, tortuosity


# ==========================================
# 4.  MEMORY TEST
# ==========================================
def run_memory_test(W, labels, pattern_type='cluster', pattern_seed=0):
    """
    Probes pattern retention in a recurrent rate model.

    Eq. 7 (manuscript): τ ẋ = -x + W·φ(x) - G_inh · 1·mean(φ(x)) + I_cue
    where 1 is the ones vector, so the inhibitory term is broadcast correctly.
    [REV-Vec]: previously used scalar subtraction; now uses np.ones to match
    vector form in the manuscript.
    """
    np.random.seed(pattern_seed)
    W_sim = W * params.GLOBAL_GAIN

    if pattern_type == 'cluster':
        p_idx = np.where(labels == 0)[0]
    else:
        n_active = max(1, int(0.1 * params.N_NEURONS))
        p_idx    = np.random.choice(params.N_NEURONS, size=n_active, replace=False)

    pattern    = np.zeros(params.N_NEURONS)
    pattern[p_idx] = 1.0
    pattern   /= (np.linalg.norm(pattern) + 1e-9)

    x          = np.zeros(params.N_NEURONS)
    ones       = np.ones(params.N_NEURONS)                 # [REV-Vec]
    n_steps    = int((params.T_CUE + params.T_TEST) / params.DT_R)
    cue_steps  = int(params.T_CUE / params.DT_R)
    sim_accum  = 0.0

    for i in range(n_steps):
        I_ext    = 2.0 * pattern if i < cue_steps else np.zeros(params.N_NEURONS)
        activity = np.tanh(x)
        # [REV-Vec] inhibitory term broadcast via ones vector
        dx       = (-x + W_sim @ activity
                    - params.GLOBAL_INH * np.mean(activity) * ones + I_ext)
        x       += (params.DT_R / params.TAU) * dx

        if i >= cue_steps:
            norm_act    = np.linalg.norm(activity) + 1e-9
            sim_accum  += max(0, np.dot(activity, pattern) / norm_act)

    return sim_accum / (params.T_TEST / params.DT_R)


# ==========================================
# 5.  [REV-Circ] CIRCULARITY CONTROLS
# ==========================================

def _spectral_radius(W):
    """Leading real eigenvalue magnitude of W."""
    return np.max(np.real(LA.eigvals(W)))


def run_weight_matched_control(W_nt, W_tr, labels, pattern_type='cluster',
                                pattern_seed=0):
    """
    [REV-Circ] Weight-matched control (Control A).

    Rescales W_tr so its spectral radius equals that of W_nt, then tests
    memory. If TR still under-performs NT, the deficit cannot be attributed
    solely to overall weight magnitude reduction — topology must be carrying
    independent explanatory work.

    Returns: (score_nt_original, score_tr_matched)
    """
    rho_nt = _spectral_radius(W_nt)
    rho_tr = _spectral_radius(W_tr)

    if rho_tr < 1e-9:
        W_tr_matched = W_tr.copy()
    else:
        W_tr_matched = W_tr * (rho_nt / rho_tr)

    score_nt      = run_memory_test(W_nt, labels, pattern_type, pattern_seed)
    score_tr_mtch = run_memory_test(W_tr_matched, labels, pattern_type, pattern_seed)
    return score_nt, score_tr_mtch


def run_topology_control(W_nt, W_tr, labels, pattern_type='cluster',
                          pattern_seed=0):
    """
    [REV-Circ] Topology-isolated control (Control B).

    Creates a binary adjacency mask from W_tr (1 where W_tr > 0, else 0)
    and fills all non-zero entries with the MEAN weight of W_nt. This
    preserves TR topology exactly while equating average synaptic strength
    to the NT level — isolating whether connection pattern (topology) drives
    deficits independently of magnitude.

    Returns: (score_nt_original, score_tr_topology_only)
    """
    mean_nt_weight   = np.mean(W_nt[W_nt > 0]) if np.any(W_nt > 0) else 1.0
    W_tr_topo        = (W_tr > 0).astype(float) * mean_nt_weight

    score_nt        = run_memory_test(W_nt, labels, pattern_type, pattern_seed)
    score_tr_topo   = run_memory_test(W_tr_topo, labels, pattern_type, pattern_seed)
    return score_nt, score_tr_topo


# ==========================================
# 6.  MAIN BATCH (WITH CIRCULARITY CONTROLS)
# ==========================================
def run_main_batch():
    print(f"--- Main Batch (N={params.N_SEEDS} seeds) ---")
    data = []

    for seed in range(params.N_SEEDS):
        somata, labels, targets = generate_substrate(seed)

        W_nt, tort_nt = run_growth(somata, targets, seed, 'NT')
        W_tr, tort_tr = run_growth(somata, targets, seed, 'TR')

        for ptype in ['cluster', 'random']:
            score_nt = run_memory_test(W_nt, labels, ptype, seed)
            score_tr = run_memory_test(W_tr, labels, ptype, seed)

            # [REV-Circ] both controls run for cluster pattern only (primary result)
            if ptype == 'cluster':
                _, score_tr_wm   = run_weight_matched_control(W_nt, W_tr, labels,
                                                               ptype, seed)
                _, score_tr_topo = run_topology_control(W_nt, W_tr, labels,
                                                         ptype, seed)
            else:
                score_tr_wm   = np.nan
                score_tr_topo = np.nan

            data.append({
                'Seed':           seed,
                'Condition':      'Neurotypical',
                'Pattern':        ptype.capitalize(),
                'Score':          score_nt,
                'Tortuosity':     tort_nt,
                'Score_WM':       np.nan,
                'Score_Topo':     np.nan,
            })
            data.append({
                'Seed':           seed,
                'Condition':      'Trauma',
                'Pattern':        ptype.capitalize(),
                'Score':          score_tr,
                'Tortuosity':     tort_tr,
                'Score_WM':       score_tr_wm,       # weight-matched
                'Score_Topo':     score_tr_topo,     # topology-only
            })

        if seed % 20 == 0:
            print(f"  Seed {seed:3d} done.")

    return pd.DataFrame(data)


# ==========================================
# 7.  PARAMETER SWEEP
# ==========================================
def run_parameter_sweep():
    print("--- Parameter Sweep ---")
    hole_amps = [0.0, 0.5, 1.0, 1.5, 2.0]
    cort_amps = [0.0, 1.0, 2.0, 3.0, 4.0]
    sweep_data = []

    for ha in hole_amps:
        for ca in cort_amps:
            scores = []
            for seed in range(params.N_SWEEP_SEEDS):
                somata, labels, targets = generate_substrate(seed)
                W_tr, _ = run_growth(somata, targets, seed, 'TR',
                                     hole_amp=ha, cort_amp=ca)
                scores.append(run_memory_test(W_tr, labels, 'cluster', seed))
            sweep_data.append({
                'Hole_Amp':    ha,
                'Cortisol_Amp': ca,
                'Mean_Score':  np.mean(scores),
                'Std_Score':   np.std(scores),
            })
            print(f"  Hole={ha:.1f}, Cortisol={ca:.1f}  →  "
                  f"Score={np.mean(scores):.3f} ± {np.std(scores):.3f}")

    return pd.DataFrame(sweep_data)


# ==========================================
# 8.  STATISTICS
# ==========================================
def report_statistics(df_main):
    sep = "=" * 50
    print(f"\n{sep}\nSTATISTICAL REPORT\n{sep}")

    for pattern in ['Cluster', 'Random']:
        df_p = df_main[df_main['Pattern'] == pattern]
        nt   = df_p[df_p['Condition'] == 'Neurotypical']['Score'].values
        tr   = df_p[df_p['Condition'] == 'Trauma']['Score'].values

        stat, p = wilcoxon(nt, tr)
        d       = (np.mean(nt) - np.mean(tr)) / (np.std(nt - tr) + 1e-9)
        ci      = np.percentile(nt - tr, [2.5, 97.5])

        print(f"\n--- {pattern} Pattern ---")
        print(f"  NT  : Mean={np.mean(nt):.3f}, SD={np.std(nt):.3f}")
        print(f"  TR  : Mean={np.mean(tr):.3f}, SD={np.std(tr):.3f}")
        print(f"  Diff: Mean={np.mean(nt-tr):.3f}, 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  Wilcoxon W={stat:.0f}, p={p:.2e}")
        print(f"  Cohen's d = {d:.3f}")

        # [REV-Circ] Report circularity controls for cluster pattern
        if pattern == 'Cluster':
            tr_df = df_p[df_p['Condition'] == 'Trauma']
            wm    = tr_df['Score_WM'].dropna().values
            topo  = tr_df['Score_Topo'].dropna().values
            nt_c  = nt  # same NT scores

            if len(wm):
                stat_wm, p_wm = wilcoxon(nt_c, wm)
                d_wm = (np.mean(nt_c) - np.mean(wm)) / (np.std(nt_c - wm) + 1e-9)
                print(f"\n  [Control A — Weight-Matched TR]")
                print(f"  TR_WM: Mean={np.mean(wm):.3f}, SD={np.std(wm):.3f}")
                print(f"  Wilcoxon W={stat_wm:.0f}, p={p_wm:.2e}, Cohen's d={d_wm:.3f}")
                print(f"  Interpretation: if p<0.05, topology drives deficit "
                      f"beyond weight magnitude.")

            if len(topo):
                stat_tp, p_tp = wilcoxon(nt_c, topo)
                d_tp = (np.mean(nt_c) - np.mean(topo)) / (np.std(nt_c - topo) + 1e-9)
                print(f"\n  [Control B — Topology-Isolated TR]")
                print(f"  TR_Topo: Mean={np.mean(topo):.3f}, SD={np.std(topo):.3f}")
                print(f"  Wilcoxon W={stat_tp:.0f}, p={p_tp:.2e}, Cohen's d={d_tp:.3f}")
                print(f"  Interpretation: if p<0.05, TR connectivity pattern itself"
                      f" is functionally suboptimal even with NT-level weights.")

    # Tortuosity
    df_tort = df_main.drop_duplicates(['Seed', 'Condition'])[
        ['Condition', 'Tortuosity', 'Seed']]
    nt_t = df_tort[df_tort['Condition'] == 'Neurotypical']['Tortuosity'].values
    tr_t = df_tort[df_tort['Condition'] == 'Trauma']['Tortuosity'].values
    stat, p = wilcoxon(nt_t, tr_t)
    d       = (np.mean(nt_t) - np.mean(tr_t)) / (np.std(nt_t - tr_t) + 1e-9)
    print(f"\n--- Tortuosity ---")
    print(f"  NT  : Mean={np.mean(nt_t):.3f}, SD={np.std(nt_t):.3f}")
    print(f"  TR  : Mean={np.mean(tr_t):.3f}, SD={np.std(tr_t):.3f}")
    print(f"  Wilcoxon W={stat:.0f}, p={p:.2e}, Cohen's d={d:.3f}")


# ==========================================
# 9.  PLOTTING
# ==========================================
def plot_main_results(df_main, df_sweep, example_data=None):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    if example_data is not None:
        # --- Trajectory figure ---
        fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig1.suptitle(
            "Axon Growth Trajectories (Seed 0)\n"
            f"N_neurons={params.N_NEURONS}, "
            f"Cortisol_Amp={params.CORTISOL_AMP}, "
            f"Hole_Amp={params.HOLE_AMP}, "
            f"Hole_Radius={params.HOLE_RADIUS}",
            fontsize=10)  # [REV-Fig]

        for ax, cond, traj in zip(
                axes, ['Neurotypical', 'Trauma'],
                [example_data['traj_nt'], example_data['traj_tr']]):
            ax.scatter(example_data['somata'][:, 0],
                       example_data['somata'][:, 1],
                       c=example_data['labels'], cmap='coolwarm', s=20, zorder=3)
            for i in range(0, params.N_NEURONS, 5):
                ax.plot(traj[:, i, 0], traj[:, i, 1],
                        alpha=0.4, linewidth=0.8, color='gray')
            if cond == 'Trauma':
                for h in [(50, 50), (25, 75), (75, 25)]:
                    ax.add_patch(patches.Circle(
                        h, params.HOLE_RADIUS, color='red', alpha=0.15))
            ax.set_title(f"{cond} Growth")
            ax.set_xlim(0, 100); ax.set_ylim(0, 100)
            ax.set_xlabel("x (a.u.)"); ax.set_ylabel("y (a.u.)")

        plt.tight_layout()
        plt.savefig("Figure_1_Trajectories.png", dpi=300, bbox_inches='tight')
        plt.show()

        # --- Connectivity matrices ---
        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle(
            f"Connectivity Matrices (Seed 0) — "
            f"W_max={params.W_MAX}, G_global={params.GLOBAL_GAIN}",
            fontsize=10)  # [REV-Fig]
        for ax, W, title in zip(
                axes,
                [example_data['W_nt'], example_data['W_tr']],
                ['Neurotypical', 'Trauma']):
            im = ax.imshow(W, cmap='inferno', vmin=0, vmax=params.W_MAX)
            ax.set_title(title)
            ax.set_xlabel("Target neuron"); ax.set_ylabel("Source neuron")
            plt.colorbar(im, ax=ax, label="Synaptic weight")
        plt.tight_layout()
        plt.savefig("Figure_2_Connectivity.png", dpi=300, bbox_inches='tight')
        plt.show()

    # --- Main results: memory + tortuosity + sweep ---
    fig3, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig3.suptitle(
        f"Main Results — N={params.N_SEEDS} seeds, "
        f"Cortisol_Amp={params.CORTISOL_AMP}, Hole_Amp={params.HOLE_AMP}, "
        f"G_global={params.GLOBAL_GAIN}, G_inh={params.GLOBAL_INH}",
        fontsize=10)  # [REV-Fig]

    for ax, pattern, title in zip(
            [axes[0, 0], axes[0, 1]],
            ['Cluster', 'Random'],
            ['(A) Cluster Pattern Memory', '(B) Random Pattern Memory']):
        df_p = df_main[df_main['Pattern'] == pattern]
        sns.violinplot(data=df_p, x='Condition', y='Score',
                       palette=['#2ecc71', '#e74c3c'],
                       cut=0, inner='quartile', ax=ax)
        sns.stripplot(data=df_p, x='Condition', y='Score',
                      color='k', size=2.5, jitter=True, alpha=0.5, ax=ax)
        for seed in df_p['Seed'].unique():
            y0 = df_p[(df_p['Seed'] == seed) &
                      (df_p['Condition'] == 'Neurotypical')]['Score'].values
            y1 = df_p[(df_p['Seed'] == seed) &
                      (df_p['Condition'] == 'Trauma')]['Score'].values
            if len(y0) and len(y1):
                ax.plot([0, 1], [y0[0], y1[0]],
                        color='gray', alpha=0.2, linewidth=0.6)
        ax.set_title(title); ax.set_ylabel("Cosine Similarity Score")

    # Tortuosity
    ax_t = axes[1, 0]
    df_tort = df_main.drop_duplicates(['Seed', 'Condition'])[
        ['Condition', 'Tortuosity', 'Seed']]
    sns.violinplot(data=df_tort, x='Condition', y='Tortuosity',
                   palette=['#2ecc71', '#e74c3c'],
                   cut=0, inner='quartile', ax=ax_t)
    sns.stripplot(data=df_tort, x='Condition', y='Tortuosity',
                  color='k', size=2.5, jitter=True, alpha=0.5, ax=ax_t)
    for seed in df_tort['Seed'].unique():
        y0 = df_tort[(df_tort['Seed'] == seed) &
                     (df_tort['Condition'] == 'Neurotypical')]['Tortuosity'].values
        y1 = df_tort[(df_tort['Seed'] == seed) &
                     (df_tort['Condition'] == 'Trauma')]['Tortuosity'].values
        if len(y0) and len(y1):
            ax_t.plot([0, 1], [y0[0], y1[0]],
                      color='gray', alpha=0.2, linewidth=0.6)
    ax_t.set_title("(C) Axon Growth Tortuosity"); ax_t.set_ylabel("Tortuosity (ratio)")

    # Parameter sweep heatmap
    pivot = df_sweep.pivot(index='Hole_Amp', columns='Cortisol_Amp', values='Mean_Score')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", ax=axes[1, 1])
    axes[1, 1].set_title("(D) Parameter Sweep — Cluster Memory Score")
    axes[1, 1].set_xlabel("Cortisol Amplitude"); axes[1, 1].set_ylabel("Hole Amplitude")

    plt.tight_layout()
    plt.savefig("Figure_3_Results_Summary.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_circularity_controls(df_main):
    """
    [REV-Circ] Plots the two circularity-control conditions alongside the
    original NT and TR scores. Three-panel violin strip chart.
    """
    df_c = df_main[(df_main['Pattern'] == 'Cluster') &
                   (df_main['Condition'] == 'Trauma')].copy()

    plot_df_rows = []
    for _, row in df_main[(df_main['Pattern'] == 'Cluster') &
                           (df_main['Condition'] == 'Neurotypical')].iterrows():
        plot_df_rows.append({'Condition': 'NT (original)',  'Score': row['Score']})
    for _, row in df_c.iterrows():
        plot_df_rows.append({'Condition': 'TR (original)',  'Score': row['Score']})
        if not np.isnan(row['Score_WM']):
            plot_df_rows.append({'Condition': 'TR (weight-\nmatched)',
                                  'Score': row['Score_WM']})
        if not np.isnan(row['Score_Topo']):
            plot_df_rows.append({'Condition': 'TR (topology-\nonly)',
                                  'Score': row['Score_Topo']})

    df_plot = pd.DataFrame(plot_df_rows)
    order    = ['NT (original)', 'TR (original)',
                'TR (weight-\nmatched)', 'TR (topology-\nonly)']
    palette  = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_plot, x='Condition', y='Score',
                   order=order, palette=palette,
                   cut=0, inner='quartile', ax=ax)
    sns.stripplot(data=df_plot, x='Condition', y='Score',
                  order=order, color='k', size=2.5, jitter=True, alpha=0.4, ax=ax)
    ax.set_title(
        "Circularity Controls — Cluster Pattern Memory\n"
        "[REV-Circ] Weight-matched and topology-isolated conditions\n"
        f"N={params.N_SEEDS} seeds, Cortisol_Amp={params.CORTISOL_AMP}, "
        f"Hole_Amp={params.HOLE_AMP}",
        fontsize=10)
    ax.set_ylabel("Cosine Similarity Score")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("Figure_4_Circularity_Controls.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_spectral_analysis(seed=42):
    """
    [REV-Spec] Generates 2×2 factorial connectivity + spectral plots.

    Language softened throughout: eigenvalue collapse is presented as a
    structural result (reduced recurrent gain) consistent with sub-criticality,
    NOT as a proof of crossing a dynamical bifurcation threshold. Dynamical
    claims are supported by the recall simulations, not eigenvalue analysis alone.
    """
    print("--- Spectral Analysis (2×2 Factorial) ---")
    scenarios = [
        {'name': 'Baseline (NT)',       'H': 0.0, 'C': 0.0},
        {'name': 'High Noise Only',     'H': 0.0, 'C': 3.0},
        {'name': 'Structural Damage\nOnly', 'H': 1.5, 'C': 0.0},
        {'name': 'Double Hit\n(Trauma)','H': 1.5, 'C': 3.0},
    ]

    somata, labels, targets = generate_substrate(seed)
    threshold = 1.0 / params.GLOBAL_GAIN   # structural reference line

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Spectral Analysis — Seed={seed}, "
        f"G_global={params.GLOBAL_GAIN}, W_max={params.W_MAX}\n"
        f"Dashed line = structural reference (1/G_global = {threshold:.1f}); "
        f"eigenvalue collapse is a structural, not dynamical, bifurcation claim.",
        fontsize=9)  # [REV-Spec, REV-Fig]

    for i, scen in enumerate(scenarios):
        W, _ = run_growth(somata, targets, seed, 'TR',
                          hole_amp=scen['H'], cort_amp=scen['C'])

        # Top: connectivity matrix
        ax_m = axes[0, i]
        im   = ax_m.imshow(W, cmap='viridis', vmin=0, vmax=params.W_MAX)
        ax_m.set_title(
            f"{scen['name']}\nH={scen['H']}, C={scen['C']}",
            fontsize=9)
        ax_m.axis('off')
        plt.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)

        # Bottom: eigenspectrum
        eigs = np.sort(np.real(LA.eigvals(W)))[::-1][:20]
        ax_s = axes[1, i]
        ax_s.bar(range(len(eigs)), eigs, color='teal', alpha=0.8)
        ax_s.set_ylim(0, max(threshold + 2, float(eigs[0]) + 1))
        ax_s.set_xlabel("Eigenmode rank")
        if i == 0:
            ax_s.set_ylabel("Eigenvalue magnitude")
        # [REV-Spec] label changed to structural reference, not "bifurcation"
        ax_s.axhline(y=threshold, color='r', linestyle='--', alpha=0.7,
                     label=f'Structural ref. (1/G = {threshold:.1f})')
        ax_s.set_title(
            f"λ₁ = {eigs[0]:.2f}  "
            f"({'above' if eigs[0] >= threshold else 'below'} ref.)",
            fontsize=9)
        ax_s.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("Fig_Response_Spectral.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_variance_criticality(df_main):
    """
    [REV-Bug] Fixed: was overwriting Fig_Response_Spectral.png.
              Now saves to Fig_Response_Variance.png.
    [REV-Fig] Full parameter annotation added.
    [REV-Spec] Language: 'near criticality' is a working hypothesis consistent
               with high variance, not a proven claim.
    """
    df_c  = df_main[df_main['Pattern'] == 'Cluster']
    nt_s  = df_c[df_c['Condition'] == 'Neurotypical']['Score']
    tr_s  = df_c[df_c['Condition'] == 'Trauma']['Score']

    print("\n--- Variance Analysis ---")
    print(f"  NT Variance: {nt_s.var():.4f}  "
          f"(consistent with near-critical dynamics; high seed sensitivity)")
    print(f"  TR Variance: {tr_s.var():.4f}  "
          f"(consistent with sub-critical collapse; functional floor effect)")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(nt_s, fill=True, label=f'Neurotypical  (Var={nt_s.var():.4f})',
                color='green', ax=ax)
    sns.kdeplot(tr_s, fill=True, label=f'Trauma  (Var={tr_s.var():.4f})',
                color='red', ax=ax)
    ax.set_title(
        "Score Distributions: Variance as Structural Sensitivity\n"
        f"[REV-Spec] Variance contrast is consistent with, not proof of, "
        f"criticality\nN={params.N_SEEDS}, "
        f"Cortisol_Amp={params.CORTISOL_AMP}, Hole_Amp={params.HOLE_AMP}",
        fontsize=9)
    ax.set_xlabel("Memory Retention Score (Cosine Similarity)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Fig_Response_Variance.png", dpi=300, bbox_inches='tight')  # [REV-Bug]
    plt.show()


def plot_1d_noise_effect(df_sweep):
    """
    [REV-Fig] Full parameter annotation added to title.
    """
    df_p = df_sweep[df_sweep['Hole_Amp'] == 0.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_p['Cortisol_Amp'], df_p['Mean_Score'],
            marker='o', color='steelblue', label='Mean score')
    ax.fill_between(df_p['Cortisol_Amp'],
                    df_p['Mean_Score'] - df_p['Std_Score'],
                    df_p['Mean_Score'] + df_p['Std_Score'],
                    alpha=0.2, color='steelblue', label='±1 SD')
    ax.set_title(
        "Noise-Driven Exploration in Intact BDNF Landscape (Hole_Amp=0)\n"
        f"N={params.N_SWEEP_SEEDS} seeds per point, Cluster pattern",
        fontsize=10)
    ax.set_xlabel("Cortisol Amplitude (stochastic diffusion noise)")
    ax.set_ylabel("Mean Memory Retention Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Fig_Response_Noise_1D.png", dpi=300, bbox_inches='tight')
    plt.show()


# ==========================================
# 10.  MAIN
# ==========================================
if __name__ == "__main__":
    # Primary batch (includes circularity controls) [REV-Circ]
    df_main  = run_main_batch()

    # Parameter sweep
    df_sweep = run_parameter_sweep()

    # Example trajectories + connectivity (seed 0)
    somata0, labels0, targets0 = generate_substrate(0)
    W_nt0, _, traj_nt0 = run_growth(somata0, targets0, 0, 'NT', return_traj=True)
    W_tr0, _, traj_tr0 = run_growth(somata0, targets0, 0, 'TR', return_traj=True)
    example_data = dict(
        somata=somata0, labels=labels0,
        W_nt=W_nt0, W_tr=W_tr0,
        traj_nt=traj_nt0, traj_tr=traj_tr0,
    )

    # Statistics
    report_statistics(df_main)

    # Figures
    plot_main_results(df_main, df_sweep, example_data)
    plot_circularity_controls(df_main)        # [REV-Circ] new figure
    run_spectral_analysis()                   # [REV-Spec, REV-Fig]
    plot_variance_criticality(df_main)        # [REV-Bug, REV-Spec, REV-Fig]
    plot_1d_noise_effect(df_sweep)            # [REV-Fig]

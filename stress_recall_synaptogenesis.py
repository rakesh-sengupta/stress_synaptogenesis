import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
import matplotlib.patches as patches
from numpy import linalg as LA  # <--- Added missing import

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Params:
    N_SEEDS = 100       # Main batch seeds
    N_SWEEP_SEEDS = 10  # Seeds for parameter sweep (for speed)
    N_NEURONS = 100
    L = 100.0

    # Growth Physics
    T_GROW = 2000
    DT_G = 0.5
    ALPHA_0 = 1.2
    D_0 = 0.1
    BETA = 0.1

    # Trauma Settings (Default)
    CORTISOL_AMP = 3.0   # Default for main batch
    HOLE_AMP = 1.5       # Default for main batch
    HOLE_RADIUS = 12.0

    # Synaptogenesis & Memory
    R_SYN = 3.0
    W_0 = 1.0
    A_B = 3.0
    W_MAX = 10.0
    GLOBAL_GAIN = 0.25
    GLOBAL_INH = 0.8

    # Dynamics
    TAU = 10.0
    DT_RATE = 0.1
    T_CUE = 20.0
    T_TEST = 200.0

params = Params()

# ==========================================
# 2. SUBSTRATE GENERATION
# ==========================================
def generate_substrate(seed):
    np.random.seed(seed)
    somata = np.random.rand(params.N_NEURONS, 2) * params.L
    cluster_centers = np.array([[25, 75], [75, 25]])
    labels = np.random.randint(0, 2, params.N_NEURONS)
    targets = cluster_centers[labels]
    return somata, labels, targets

# ==========================================
# 3. PHYSICS ENGINE (With Tortuosity Tracking)
# ==========================================
def get_bdnf_field(coords, t, condition, hole_amp, cort_amp):
    centers = np.array([[25, 75], [75, 25]])
    hole_locs = np.array([[50, 50], [25, 75], [75, 25]])

    grad_x = np.zeros(coords.shape[0])
    grad_y = np.zeros(coords.shape[0])
    val = np.zeros(coords.shape[0])

    t_peak = params.T_GROW / 3.0
    sigma_b = params.T_GROW / 5.0
    sb_t = np.exp(-((t - t_peak)**2) / (2 * sigma_b**2))

    for i in range(len(centers)):
        dx = coords[:, 0] - centers[i, 0]
        dy = coords[:, 1] - centers[i, 1]
        dist_sq = dx**2 + dy**2
        gauss = 1.0 * np.exp(-dist_sq / (2 * 15.0**2))
        val += sb_t * gauss
        grad_x += sb_t * gauss * (-dx / 15.0**2)
        grad_y += sb_t * gauss * (-dy / 15.0**2)

    if condition == 'TR':
        for h in hole_locs:
            dx = coords[:, 0] - h[0]
            dy = coords[:, 1] - h[1]
            dist_sq = dx**2 + dy**2
            gauss_h = hole_amp * np.exp(-dist_sq / (2 * params.HOLE_RADIUS**2))
            val -= sb_t * gauss_h
            grad_x += sb_t * gauss_h * (dx / params.HOLE_RADIUS**2)
            grad_y += sb_t * gauss_h * (dy / params.HOLE_RADIUS**2)

    return np.stack([grad_x, grad_y], axis=1), val

def run_growth(somata, targets, seed, condition, hole_amp=None, cort_amp=None, return_traj=False):
    np.random.seed(seed + 99999)
    tips = somata.copy()
    W = np.zeros((params.N_NEURONS, params.N_NEURONS))

    # Use provided parameters or defaults
    if hole_amp is None:
        hole_amp = params.HOLE_AMP
    if cort_amp is None:
        cort_amp = params.CORTISOL_AMP

    traj_history = [] if return_traj else None
    path_lengths = np.zeros(params.N_NEURONS)
    start_pos = tips.copy()

    for step in range(params.T_GROW):
        if return_traj and step % 10 == 0:
            traj_history.append(tips.copy())

        t = step * params.DT_G

        cortisol = 0.0
        if condition == 'TR':
            pulse = np.exp(-((t - params.T_GROW/3)**2)/(2*(params.T_GROW/20)**2))
            cortisol = cort_amp * pulse

        alpha = params.ALPHA_0 / (1.0 + 3.0 * cortisol)
        D = params.D_0 * (1.0 + 5.0 * cortisol)

        grad, b_val = get_bdnf_field(tips, t, condition, hole_amp, cort_amp)
        vec = targets - tips
        norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9

        noise = np.random.randn(params.N_NEURONS, 2)
        step_delta = params.DT_G * (alpha * grad + params.BETA * (vec/norm)) + np.sqrt(2*D*params.DT_G)*noise

        # Accumulate path length for tortuosity
        path_lengths += np.linalg.norm(step_delta, axis=1)

        tips += step_delta
        tips = np.clip(tips, 0, params.L)

        if step % 20 == 0:
            dists = cdist(tips, somata)
            connect_mask = (dists < params.R_SYN) & (dists > 0.1)
            local_bdnf = b_val[np.where(connect_mask)[0]]
            sources, targets_idx = np.where(connect_mask)

            for i, (s, tg) in enumerate(zip(sources, targets_idx)):
                if W[s, tg] == 0:
                    strength = params.W_0 * (1.0 + params.A_B * local_bdnf[i])
                    strength = max(0.0, min(strength, params.W_MAX))
                    W[s, tg] = strength

    # Calculate tortuosity
    displacements = np.linalg.norm(tips - start_pos, axis=1) + 1e-9
    tortuosity = np.mean(path_lengths / displacements)

    if return_traj:
        return W, tortuosity, np.array(traj_history)
    else:
        return W, tortuosity

# ==========================================
# 4. MEMORY TEST FOR TWO PATTERN TYPES
# ==========================================
def run_memory_test(W, labels, pattern_type='cluster', pattern_seed=0):
    np.random.seed(pattern_seed)  # For reproducibility of random pattern
    W_sim = W * params.GLOBAL_GAIN

    if pattern_type == 'cluster':
        # Cluster 0 pattern
        p_indices = np.where(labels == 0)[0]
    else:  # random
        # Randomly select 10% of neurons
        n_active = max(1, int(0.1 * params.N_NEURONS))
        p_indices = np.random.choice(params.N_NEURONS, size=n_active, replace=False)

    pattern = np.zeros(params.N_NEURONS)
    pattern[p_indices] = 1.0
    pattern /= (np.linalg.norm(pattern) + 1e-9)

    x = np.zeros(params.N_NEURONS)
    n_steps = int((params.T_CUE + params.T_TEST) / params.DT_RATE)
    cue_steps = int(params.T_CUE / params.DT_RATE)

    similarity_accum = 0.0

    for i in range(n_steps):
        I_ext = np.zeros(params.N_NEURONS)
        if i < cue_steps:
            I_ext = 2.0 * pattern
        activity = np.tanh(x)
        dx = -x + np.dot(W_sim, activity) - params.GLOBAL_INH * np.mean(activity) + I_ext
        x += (params.DT_RATE / params.TAU) * dx

        if i >= cue_steps:
            norm_act = np.linalg.norm(activity) + 1e-9
            sim = np.dot(activity, pattern) / norm_act
            similarity_accum += max(0, sim)

    return similarity_accum / (params.T_TEST / params.DT_RATE)

# ==========================================
# 5. MAIN BATCH EXPERIMENT
# ==========================================
def run_main_batch():
    print(f"--- Main Batch Experiment (N={params.N_SEEDS}) ---")
    data = []

    for seed in range(params.N_SEEDS):
        somata, labels, targets = generate_substrate(seed)

        # Neurotypical
        W_nt, tort_nt = run_growth(somata, targets, seed, 'NT')
        score_nt_cluster = run_memory_test(W_nt, labels, 'cluster', seed)
        score_nt_random = run_memory_test(W_nt, labels, 'random', seed)

        # Trauma (with default parameters)
        W_tr, tort_tr = run_growth(somata, targets, seed, 'TR')
        score_tr_cluster = run_memory_test(W_tr, labels, 'cluster', seed)
        score_tr_random = run_memory_test(W_tr, labels, 'random', seed)

        # Store
        data.append({'Seed': seed, 'Condition': 'Neurotypical', 'Pattern': 'Cluster',
                     'Score': score_nt_cluster, 'Tortuosity': tort_nt})
        data.append({'Seed': seed, 'Condition': 'Neurotypical', 'Pattern': 'Random',
                     'Score': score_nt_random, 'Tortuosity': tort_nt})
        data.append({'Seed': seed, 'Condition': 'Trauma', 'Pattern': 'Cluster',
                     'Score': score_tr_cluster, 'Tortuosity': tort_tr})
        data.append({'Seed': seed, 'Condition': 'Trauma', 'Pattern': 'Random',
                     'Score': score_tr_random, 'Tortuosity': tort_tr})

        if seed % 20 == 0:
            print(f"Seed {seed} done...")

    return pd.DataFrame(data)

# ==========================================
# 6. PARAMETER SWEEP
# ==========================================
def run_parameter_sweep():
    print(f"--- Parameter Sweep ---")
    hole_amps = [0.0, 0.5, 1.0, 1.5, 2.0]
    cort_amps = [0.0, 1.0, 2.0, 3.0, 4.0]

    sweep_data = []

    for ha in hole_amps:
        for ca in cort_amps:
            scores = []
            for seed in range(params.N_SWEEP_SEEDS):
                somata, labels, targets = generate_substrate(seed)
                # For the sweep, we only run the trauma condition with the given parameters
                W_tr, _ = run_growth(somata, targets, seed, 'TR', hole_amp=ha, cort_amp=ca)
                # Test with cluster pattern (same as main batch)
                score = run_memory_test(W_tr, labels, 'cluster', seed)
                scores.append(score)
            sweep_data.append({'Hole_Amp': ha, 'Cortisol_Amp': ca,
                               'Mean_Score': np.mean(scores), 'Std_Score': np.std(scores)})
            print(f"Hole {ha}, Cort {ca}: Score {np.mean(scores):.3f}")

    return pd.DataFrame(sweep_data)

# ==========================================
# 7. PLOTTING
# ==========================================
def plot_figures(df_main, df_sweep, example_data=None):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # If we have example data (from seed 0) for trajectories and connectivity
    if example_data is not None:
        # Plot trajectories
        fig1, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, cond, traj in zip(axes, ['Neurotypical', 'Trauma'],
                                   [example_data['traj_nt'], example_data['traj_tr']]):
            ax.scatter(example_data['somata'][:,0], example_data['somata'][:,1],
                       c=example_data['labels'], cmap='coolwarm', s=20, zorder=3)
            for i in range(0, params.N_NEURONS, 5):
                ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.4, linewidth=0.8, color='gray')
            if cond == 'Trauma':
                holes = [(50, 50), (25, 75), (75, 25)]
                for h in holes:
                    circ = patches.Circle(h, params.HOLE_RADIUS, color='red', alpha=0.1)
                    ax.add_patch(circ)
            ax.set_title(f"{cond} Growth")
            ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.show()

        # Plot connectivity matrices
        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        vmax = params.W_MAX
        axes[0].imshow(example_data['W_nt'], cmap='inferno', vmin=0, vmax=vmax)
        axes[0].set_title("Neurotypical Connectivity")
        axes[1].imshow(example_data['W_tr'], cmap='inferno', vmin=0, vmax=vmax)
        axes[1].set_title("Trauma Connectivity")
        plt.tight_layout()
        plt.show()

    # Main results: memory scores and tortuosity (use violins)
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Memory scores for cluster pattern
    df_cluster = df_main[df_main['Pattern'] == 'Cluster']
    sns.violinplot(data=df_cluster, x='Condition', y='Score',
                   palette=['#2ecc71', '#e74c3c'],
                   cut=0, inner='quartile', scale='width', ax=axes[0, 0])
    sns.stripplot(data=df_cluster, x='Condition', y='Score', ax=axes[0, 0],
                  color='k', size=3, jitter=True, dodge=True, alpha=0.6)
    # paired lines connecting NT->TR for each seed
    for seed in np.unique(df_cluster['Seed'].values):
        row_nt = df_cluster[(df_cluster['Seed'] == seed) & (df_cluster['Condition'] == 'Neurotypical')]
        row_tr = df_cluster[(df_cluster['Seed'] == seed) & (df_cluster['Condition'] == 'Trauma')]
        if len(row_nt) and len(row_tr):
            y0 = float(row_nt['Score'].values[0])
            y1 = float(row_tr['Score'].values[0])
            axes[0, 0].plot([0, 1], [y0, y1], color='gray', alpha=0.25, linewidth=0.7)
    axes[0, 0].set_title("Cluster Pattern Memory")

    # Memory scores for random pattern
    df_random = df_main[df_main['Pattern'] == 'Random']
    sns.violinplot(data=df_random, x='Condition', y='Score',
                   palette=['#2ecc71', '#e74c3c'],
                   cut=0, inner='quartile', scale='width', ax=axes[0, 1])
    sns.stripplot(data=df_random, x='Condition', y='Score', ax=axes[0, 1],
                  color='k', size=3, jitter=True, dodge=True, alpha=0.6)
    for seed in np.unique(df_random['Seed'].values):
        row_nt = df_random[(df_random['Seed'] == seed) & (df_random['Condition'] == 'Neurotypical')]
        row_tr = df_random[(df_random['Seed'] == seed) & (df_random['Condition'] == 'Trauma')]
        if len(row_nt) and len(row_tr):
            y0 = float(row_nt['Score'].values[0])
            y1 = float(row_tr['Score'].values[0])
            axes[0, 1].plot([0, 1], [y0, y1], color='gray', alpha=0.25, linewidth=0.7)
    axes[0, 1].set_title("Random Pattern Memory")

    # Tortuosity: take unique condition values per seed
    df_tort = df_main.drop_duplicates(['Seed', 'Condition'])[['Condition', 'Tortuosity', 'Seed']]
    sns.violinplot(data=df_tort, x='Condition', y='Tortuosity',
                   palette=['#2ecc71', '#e74c3c'],
                   cut=0, inner='quartile', scale='width', ax=axes[1, 0])
    sns.stripplot(data=df_tort, x='Condition', y='Tortuosity', ax=axes[1, 0],
                  color='k', size=3, jitter=True, dodge=True, alpha=0.6)
    for seed in np.unique(df_tort['Seed'].values):
        row_nt = df_tort[(df_tort['Seed'] == seed) & (df_tort['Condition'] == 'Neurotypical')]
        row_tr = df_tort[(df_tort['Seed'] == seed) & (df_tort['Condition'] == 'Trauma')]
        if len(row_nt) and len(row_tr):
            y0 = float(row_nt['Tortuosity'].values[0])
            y1 = float(row_tr['Tortuosity'].values[0])
            axes[1, 0].plot([0, 1], [y0, y1], color='gray', alpha=0.25, linewidth=0.7)
    axes[1, 0].set_title("Axon Growth Tortuosity")

    # Parameter sweep heatmap
    pivot = df_sweep.pivot(index='Hole_Amp', columns='Cortisol_Amp', values='Mean_Score')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", ax=axes[1, 1])
    axes[1, 1].set_title("Parameter Sweep (Trauma Memory Score)")
    axes[1, 1].set_xlabel("Cortisol Amplitude")
    axes[1, 1].set_ylabel("Hole Amplitude")

    plt.tight_layout()
    plt.show()

# ==========================================
# 8. STATISTICAL REPORTING
# ==========================================
def report_statistics(df_main):
    print("\n" + "="*40)
    print("STATISTICAL REPORT")
    print("="*40)

    # For each pattern type, compare NT and TR
    for pattern in ['Cluster', 'Random']:
        df_pattern = df_main[df_main['Pattern']==pattern]
        nt = df_pattern[df_pattern['Condition']=='Neurotypical']['Score'].values
        tr = df_pattern[df_pattern['Condition']=='Trauma']['Score'].values

        stat, p = wilcoxon(nt, tr)
        d = (np.mean(nt) - np.mean(tr)) / np.std(nt - tr)
        ci_low, ci_high = np.percentile(nt - tr, [2.5, 97.5])

        print(f"\n--- {pattern} Pattern ---")
        print(f"Neurotypical: Mean={np.mean(nt):.3f}, SD={np.std(nt):.3f}")
        print(f"Trauma:       Mean={np.mean(tr):.3f}, SD={np.std(tr):.3f}")
        print(f"Difference:   Mean={np.mean(nt-tr):.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}]")
        print(f"Wilcoxon p:   {p:.2e}")
        print(f"Cohen's d:    {d:.3f}")

    # Tortuosity
    df_tort = df_main.drop_duplicates(['Seed', 'Condition'])[['Condition', 'Tortuosity']]
    nt_tort = df_tort[df_tort['Condition']=='Neurotypical']['Tortuosity'].values
    tr_tort = df_tort[df_tort['Condition']=='Trauma']['Tortuosity'].values
    stat, p = wilcoxon(nt_tort, tr_tort)
    d = (np.mean(nt_tort) - np.mean(tr_tort)) / np.std(nt_tort - tr_tort)
    print(f"\n--- Tortuosity ---")
    print(f"Neurotypical: Mean={np.mean(nt_tort):.3f}, SD={np.std(nt_tort):.3f}")
    print(f"Trauma:       Mean={np.mean(tr_tort):.3f}, SD={np.std(tr_tort):.3f}")
    print(f"Wilcoxon p:   {p:.2e}")
    print(f"Cohen's d:    {d:.3f}")

def run_spectral_analysis(seed=42):
    """
    Generates the 2x2 Factorial Plot (Reviewer Request) AND Spectral Analysis (Deep Math)
    """
    print("Running 2x2 Spectral Analysis...")

    # The 4 Corners of the Parameter Space
    scenarios = [
        {'name': 'Baseline (NT)',      'H': 0.0, 'C': 0.0}, # Low C, Low H
        {'name': 'High Noise',         'H': 0.0, 'C': 3.0}, # High C, Low H
        {'name': 'Structural Damage',  'H': 1.5, 'C': 0.0}, # Low C, High H
        {'name': 'Trauma (Double Hit)','H': 1.5, 'C': 3.0}  # High C, High H
    ]

    somata, labels, targets = generate_substrate(seed)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, scen in enumerate(scenarios):
        # 1. Run Growth
        W, _ = run_growth(somata, targets, seed, 'TR', hole_amp=scen['H'], cort_amp=scen['C'])

        # 2. Plot Matrix (Top Row) - Visualizing Connectivity
        ax_mat = axes[0, i]
        im = ax_mat.imshow(W, cmap='viridis', vmin=0, vmax=10)
        ax_mat.set_title(f"{scen['name']}\nH={scen['H']}, C={scen['C']}")
        ax_mat.axis('off')

        # 3. Spectral Analysis (Bottom Row) - Deep Math
        # Calculate Eigenvalues
        eigenvalues = np.real(LA.eigvals(W))
        # Sort and take top 20
        eigenvalues = np.sort(eigenvalues)[::-1][:20]

        ax_spec = axes[1, i]
        ax_spec.bar(range(20), eigenvalues, color='teal')
        ax_spec.set_ylim(0, max(12, eigenvalues[0]+1))
        ax_spec.set_title(f"Spectral Radius: $\lambda_1={eigenvalues[0]:.2f}$")
        ax_spec.set_xlabel("Eigenmode Rank")
        if i == 0: ax_spec.set_ylabel("Eigenvalue Magnitude")

        # Overlay the "Critical Threshold" (Heuristic)
        ax_spec.axhline(y=1.0/params.GLOBAL_GAIN, color='r', linestyle='--', alpha=0.5, label='Bifurcation')

    plt.tight_layout()
    plt.savefig('Fig_Response_Spectral.png')
    plt.show()

def explain_variance_via_criticality(df_main):
    """
    Mathematical explanation for Reviewer's variance question.
    """
    nt_scores = df_main[df_main['Condition']=='Neurotypical']['Score']
    tr_scores = df_main[df_main['Condition']=='Trauma']['Score']

    print("\n--- Variance Analysis (Criticality Hypothesis) ---")
    print(f"NT Variance: {nt_scores.var():.4f} (Near Bifurcation = High Sensitivity)")
    print(f"TR Variance: {tr_scores.var():.4f} (Sub-critical = Consistently Dead)")

    # Plotting the Distributions
    plt.figure(figsize=(8,5))
    sns.kdeplot(nt_scores, fill=True, label='Neurotypical (Critical)', color='green')
    sns.kdeplot(tr_scores, fill=True, label='Trauma (Sub-critical)', color='red')
    plt.title("Score Distributions: Criticality vs. Collapse")
    plt.xlabel("Memory Retention Score")
    plt.legend()
    plt.savefig('Fig_Response_Spectral.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_1d_noise_effect(df_sweep):
    # Filter for NO structural damage
    df_pristine = df_sweep[df_sweep['Hole_Amp'] == 0.0]
    
    plt.figure(figsize=(6,4))
    plt.plot(df_pristine['Cortisol_Amp'], df_pristine['Mean_Score'], marker='o', color='blue')
    plt.fill_between(df_pristine['Cortisol_Amp'], 
                     df_pristine['Mean_Score'] - df_pristine['Std_Score'],
                     df_pristine['Mean_Score'] + df_pristine['Std_Score'], alpha=0.2)
    plt.title("Noise-Driven Exploration (H=0)")
    plt.xlabel("Cortisol Amplitude (Noise)")
    plt.ylabel("Memory Retention Score")
    plt.savefig('Fig_Response_Noise_1D.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 9. MAIN
# ==========================================
if __name__ == "__main__":
    # Run main batch
    df_main = run_main_batch()

    # Run parameter sweep
    df_sweep = run_parameter_sweep()

    # Get example data (seed 0) for trajectories and connectivity
    somata0, labels0, targets0 = generate_substrate(0)
    W_nt0, tort_nt0, traj_nt0 = run_growth(somata0, targets0, 0, 'NT', return_traj=True)
    W_tr0, tort_tr0, traj_tr0 = run_growth(somata0, targets0, 0, 'TR', return_traj=True)
    example_data = {
        'somata': somata0, 'labels': labels0,
        'W_nt': W_nt0, 'W_tr': W_tr0,
        'traj_nt': traj_nt0, 'traj_tr': traj_tr0
    }

    # Report statistics
    report_statistics(df_main)

    # Plot figures
    plot_figures(df_main, df_sweep, example_data)

    # Run new analyses
    run_spectral_analysis()
    explain_variance_via_criticality(df_main)
    plot_1d_noise_effect(df_sweep)

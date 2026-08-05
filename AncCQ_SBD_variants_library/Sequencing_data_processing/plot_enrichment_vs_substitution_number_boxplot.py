import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sn

# --- Configuration ---
color_dict = {
    "LPPVK": "#f2b330",
    "p5": "#05d70b",
}

bin_width_dict = {
    "LPPVK":0.15,
    "p5":0.1
}

yticks_dict = {
    "LPPVK":(0,1,2,3,4),
    "p5": (0,1,2,3,4),
}

axis_labels_font_size = 16

marker_map = {
    "5": "#000000",           # Black line - AncCQ F462S
    "00000000000000": "#FF0000", # Red line - AncCQ
    "11111111101110": "#0000FF", # Blue line - AncCQ+12
}


# --- 1. Load Data ---
df_full = pl.read_parquet("Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")

#Exclude AncCQ F462S and variants containing T494K and S540N substitutions from plotting
df_plot = df_full.filter(
    (pl.col("codon_string") != "5") &
    (pl.col("codon_string").str.contains(r".........0...0"))
)

# --- 3. Setup Plotting ---
conditions = ["LPPVK", "p5"]

fig = plt.figure(figsize=(5, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], wspace=0.1, hspace=0.1)
first_ax_dist = None

for i, cond in enumerate(conditions):
    # Define axes
    if i == 0:
        ax_dist = fig.add_subplot(gs[i, 1])
        first_ax_dist = ax_dist
    else:
        ax_dist =  fig.add_subplot(gs[i, 1], sharex=first_ax_dist)
    ax_box = fig.add_subplot(gs[i, 0], sharey=ax_dist)

    # --- A. Plot General Data (Box & Scatter) ---
    df_subset = df_plot.filter(pl.col("selection") == cond)

    unique_dists = []
    if df_subset.height > 0:
        dist = df_subset["substitution_number"].to_numpy()
        score = df_subset["linear_score"].to_numpy()
        unique_dists = np.sort(np.unique(dist))

        # 1. Distribution (Right Panel)
        ax_dist.hist(
            score, bins=np.arange(min(score), max(score) + bin_width_dict[cond], bin_width_dict[cond]), orientation='horizontal',
            color=color_dict.get(cond, "gray"), alpha=0.8, edgecolor="#000000",linewidth=0.5
        )

        # 2. Scatter (Left Panel - Jittered Background)
        jitter = np.random.uniform(-0.15, 0.15, size=len(dist))
        ax_box.scatter(
            dist + jitter, score, s=8, alpha=0.2,
            c=color_dict.get(cond, "gray"), edgecolors='none', zorder=1
        )

        # 3. Box Plot (Left Panel - Overlay)
        data_for_boxplot = [score[dist == d] for d in unique_dists]
        ax_box.boxplot(
            data_for_boxplot, positions=unique_dists, widths=0.6,
            showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
            medianprops=dict(color='black', linewidth=2),
            zorder=2
        )

    # --- B. Add marker lines to distributions ---
    for codon_str, color in marker_map.items():
        target_row = df_full.filter(
            (pl.col("selection") == cond) &
            (pl.col("codon_string") == codon_str)
        )

        if target_row.height > 0:
            target_score = target_row.select(pl.col("linear_score")).item()
            target_sub = target_row.select(pl.col("substitution_number")).item()

            if target_score is not None:
                ax_dist.axhline(target_score, color=color, linestyle='--', linewidth=2, alpha=1.0)


    # --- Styling ---
    # Distribution
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.set_yticks(yticks_dict.get(cond, np.arange(-0.5, 2.0, 0.5)))
    ax_dist.tick_params(axis='x', labelsize=axis_labels_font_size)
    ax_dist.tick_params(axis='y', labelsize=axis_labels_font_size)
    ax_dist.set_ylim([-0.2,4.2])
    plt.setp(ax_dist.get_yticklabels(), visible=False) # Hide Y ticks (shared)

    # Boxplot
    ax_box.set_yticks(yticks_dict.get(cond, np.arange(-0.5, 2.0, 0.5)))
    plt.setp(ax_box.get_yticklabels(), visible=True) # Hide Y ticks (shared)
    ax_box.tick_params(axis='x', labelsize=axis_labels_font_size)
    ax_box.tick_params(axis='y', labelsize=axis_labels_font_size)
    ax_box.set_ylim([-0.2,4.2])

    # X-Axis Labels
    if i == len(conditions) - 1: 
        ax_box.set_xlabel("Substitution number", fontsize=15)
        if len(unique_dists) > 0:
            ax_box.set_xticks(unique_dists)
            ax_box.set_xticklabels([val if i % 2 == 0 else "" for i, val in enumerate(unique_dists)])

        ax_dist.set_xlabel("Count", fontsize=15)
    else:
        plt.setp(ax_box.get_xticklabels(), visible=False)
        plt.setp(ax_dist.get_xticklabels(), visible=False)

    for spine in ax_box.spines.values():
        spine.set_linewidth(1.5)
    for spine in ax_dist.spines.values():
        spine.set_linewidth(1.5)
    ax_box.tick_params(width=1.5,length=6)
    ax_dist.tick_params(width=1.5,length=6)

sn.despine()
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.savefig("per_substitution_number_boxplot_no_T494K_S540N.png", dpi=300)
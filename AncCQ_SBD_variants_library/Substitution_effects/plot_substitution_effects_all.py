import pandas as pd
import polars as pl
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap

df_En = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")\
    .sort("codon_string")
df_En_no_F462S = df_En.filter(pl.col("codon_string") != "5")

df_variants = df_En_no_F462S.group_by("codon_string").agg(
    pl.col("linear_score").filter(pl.col("selection")=="LPPVK").first().alias("LPPVK_enrichment"),
    pl.col("linear_score").filter(pl.col("selection")=="p5").first().alias("p5_enrichment"),
).sort("codon_string")

substitution_list = ["L440Y", "T445S", "R446P", "I454V", "T456V", "S459T", "A466V", "S471G", "E482G", "T494K", "T511S", "A528T", "D532E", "S540N"]
Pareto_ordered_substitution_list = ["L440Y", "S471G","A466V","R446P","D532E","T445S","S459T","T456V","A528T","T511S","I454V","E482G","T494K","S540N"]
selections = ["LPPVK_enrichment", "p5_enrichment"]
peptides = ["LPPVK", "p5"]
peptide_colorbar_ticks_dict = {"LPPVK":[-1,-0.5,0,0.5,1],"p5":[-0.3,-0.15,0,0.15,0.3]}

sn.set_context("paper")
sn.set_style("ticks")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), sharex=True)

fig.subplots_adjust(left=0.15, right=0.85, bottom=0.25, top=0.95, hspace=0.15)
yticks = [[-2,-1,0,1,2,3],[-2,-1,0,1,2,3]]
ylim = [-2.5,3.1]

for ax, selection, peptide, yticks in zip(axes, selections, peptides, yticks):
    dfs_to_concat = []

    for i, mutation_name in enumerate(substitution_list):
        regex_1 = f"^.{{{i}}}1.*"
        regex_0 = f"^.{{{i}}}0.*"

        vals_1 = df_variants.filter(pl.col('codon_string').str.contains(regex_1))[selection]
        vals_0 = df_variants.filter(pl.col('codon_string').str.contains(regex_0))[selection]

        diffs = vals_1.to_numpy() - vals_0.to_numpy()

        temp_df = pd.DataFrame({
            "Mutation": mutation_name,
            "Effect_Size": diffs
        })
        dfs_to_concat.append(temp_df)

    plot_df = pd.concat(dfs_to_concat, ignore_index=True)

    #Colors
    medians = plot_df.groupby("Mutation")["Effect_Size"].median()
    colors_purple_green = ["#a956b7", "#FFFFFF", "#40a66c"]
    cmap = LinearSegmentedColormap.from_list("PurpleGreen", colors_purple_green, N=256)
    max_abs_val = np.max(np.abs(medians)) * 1.3
    norm = mcolors.Normalize(vmin=-max_abs_val, vmax=max_abs_val)
    palette_dict = {mut: cmap(norm(medians[mut])) for mut in Pareto_ordered_substitution_list}

    #Filter out values below 1st and above 99th percentile

    q_low = plot_df.groupby("Mutation")["Effect_Size"].transform(lambda x: x.quantile(0.01))
    q_high = plot_df.groupby("Mutation")["Effect_Size"].transform(lambda x: x.quantile(0.99))
    filtered_df = plot_df[(plot_df["Effect_Size"] >= q_low) & (plot_df["Effect_Size"] <= q_high)]

    ax.axhline(0, color='gray', linestyle='--', alpha=1, linewidth=1, zorder=0)

    sn.violinplot(
        ax=ax,
        data=filtered_df,
        x="Mutation",
        y="Effect_Size",
        hue="Mutation",
        palette=palette_dict,
        inner="quartile",
        gridsize=200,
        common_norm=True,
        legend=False,
        cut=0,
        order=Pareto_ordered_substitution_list,
        linewidth=0.6,
        linecolor='k'
    )

    ax.set_title("")
    ax.set_ylabel("$\\Delta E_{\\mathrm{N}}$, " + peptide)
    ax.set_xlabel("")
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)

    cax = inset_axes(ax,
                     width="2.5%",
                     height="50%",
                     loc='upper left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)

    cbar.outline.set_linewidth(1)
    cbar.set_label("Median $\\Delta E_{\\mathrm{N}}$", rotation=270, labelpad=12, fontsize=8)
    cax.tick_params(labelsize=8, width=1)
    cax.set_yticks(peptide_colorbar_ticks_dict[peptide])

#Format shared x-axis 
axes[1].tick_params(axis='x', rotation=90)
tick_labels = axes[1].get_xticklabels()
indices_to_italicize = [12,13]
indices_to_bold = [0, 1, 2, 3, 4, 5]

for i in indices_to_italicize:
    tick_labels[i].set_style('italic')

for i in indices_to_bold:
    tick_labels[i].set_fontweight('bold')

plt.savefig("substitution_effects_all_Pareto_order.png", dpi=600, bbox_inches='tight')
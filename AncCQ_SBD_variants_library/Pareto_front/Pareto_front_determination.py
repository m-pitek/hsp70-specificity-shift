import numpy as np
import polars as pl
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sn
from collections import Counter


def get_pareto_front_polars(df: pl.DataFrame, x_col: str, y_col: str, max_x=True, max_y=True):
    """
    Finds the Pareto front in a Polars DataFrame using vectorized operations.
    
    df - polars dataframe listing enrichments for all AncCQ SBD substitution variants in a short format
    x_col - column name for the first enrichment score (LPPVK binding)
    y_col - column name for the second enrichment score (p5 binding)
    max_x - should target Pareto front maximize value contained in the x_col?
    max_y - should target Pareto front maximize value contained in the y_col?

    Returns: A dataframe containing all AncCQ SBD variants that lie on the Pareto front
    """

    # 1. Sort based on X
    # If maximizing X: Sort Descending
    # If minimizing X: Sort Ascending
    df_sorted = df.sort(x_col, descending=max_x)

    # 2. Create the Filter Mask using Cumulative functions
    # Compare the current Y against the best Y seen in previous rows
    if max_y:
        # Find rows where Y is strictly GREATER than the max Y of all previous rows
        # shift(1) to compare with the "history" of the column
        # fill_null with -infinity so the first row is always selected
        mask = pl.col(y_col) > pl.col(y_col).cum_max().shift(1).fill_null(float('-inf'))
    else:
        # Find rows where Y is strictly LOWER than the min Y of all previous rows
        # fill_null with +infinity so the first row is always selected
        mask = pl.col(y_col) < pl.col(y_col).cum_min().shift(1).fill_null(float('inf'))

    # 3. Apply the filter
    return df_sorted.filter(mask)

def calculate_position_counts(df, col_name="codon_string",frequency=False):
    """
    Calculates the substitution count (1's) at each of the 14 positions

    df - polars dataframe listing AncCQ SBD substitution variants
    col_name - name of the column containing binary string informing about absence (0) or presence (1) of substitution
    frequency - should the count be converted to substitution frequency?

    Returns: A DataFrame with 1 row and 14 columns listing counts/frequencies of each substitution
    """

    if frequency:
        expressions = [
            pl.col(col_name).str.slice(i, 1).cast(pl.UInt8).sum().truediv(len(df)).alias(f"pos_{i}")
            for i in range(14)
        ]
    else:
        expressions = [
            pl.col(col_name).str.slice(i, 1).cast(pl.UInt8).sum().alias(f"pos_{i}")
            for i in range(14)
        ]

    return df.select(expressions)

def calculate_front_variants_number(front_variants_counts, threshold):
    """
    Calculates number of unique AncCQ SBD substitution variants that occupy certain fraction of the
    total pool of the Pareto front variants

    front_item_counts - collections.Counter containing counts of Pareto front AncCQ SBD variants
    threshold - fractional threshold used to calculate minimal number of unique AncCQ SBD variants needed to reach it

    Returns: Number of unique AncCQ SBD variants needed to reach given threshold fraction
    """

    variant_count = 0
    unique_variant_count = 0
    total_variant_count = np.sum(list(front_variants_counts.values()))
    target_count = threshold*total_variant_count

    for item in sorted(front_variants_counts.items(),
                       key=lambda item: item[1],
                       reverse=True):
        variant_count += item[1]
        unique_variant_count +=1
        if variant_count >= target_count:
            return unique_variant_count

def plot_variant_point(df,variant,color,ax,edgecolor="black",linewidth=1.0):
    """
    Marks individual AncCQ SBD variant in the scatter plot

    df - polars dataframe listing enrichments for all AncCQ SBD substitution variants in a long format
    variant - binary string identifying SBD AncCQ variant to plot
    color - hex specifying the color to use for plotting the point
    ax - axes used to plot the variant
    edgecolor - color of the point rim
    linewidth - edge line width

    Returns: None
    """

    sn.scatterplot(x=df.filter((pl.col("codon_string")==variant) & (pl.col("selection")=="LPPVK"))["linear_score"],
                    y=df.filter((pl.col("codon_string")==variant) & (pl.col("selection")=="p5"))["linear_score"],
                    s=50,
                    color=color,
                    ax=ax,
                    edgecolor=edgecolor,
                    linewidth=linewidth, 
                    zorder=2)


substitution_list = ["L440Y", "T445S", "R446P", "I454V", "T456V", "S459T", "A466V", "S471G", "E482G", "T494K", "T511S", "A528T", "D532E", "S540N"]

df_En = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet").sort("codon_string")

#Select everything except the negative control F462S and variants containing T494K or S540N substitutions
#Ensure that rows are sorted by the codon string
df_En_filtered = df_En.filter(pl.col("codon_string")!="5").filter((pl.col("codon_string").str.contains(r".........0...0"))).sort("codon_string")

SAVE_SCATTERPLOT = True

df_variants = df_En_filtered.group_by("codon_string").agg(
    pl.col("linear_score").filter(pl.col("selection")=="LPPVK").first().alias("LPPVK_enrichment"),
    pl.col("linear_score").filter(pl.col("selection")=="p5").first().alias("p5_enrichment"),
    pl.col("substitution_number").first()
)

# ============= Pareto front Monte-Carlo sampling ==============================================================================
sample_number = 5000
np.random.seed(1001)

En_LPPVK_mean_arr = df_En_filtered.filter(pl.col("selection")=="LPPVK")["linear_score"].to_numpy()
En_LPPVK_stderr_arr = df_En_filtered.filter(pl.col("selection")=="LPPVK")["linear_SE"].to_numpy()
En_p5_mean_arr = df_En_filtered.filter(pl.col("selection")=="p5")["linear_score"].to_numpy()
En_p5_stderr_arr = df_En_filtered.filter(pl.col("selection")=="p5")["linear_SE"].to_numpy()

fronts = []
front_frequencies= []

for i in range(sample_number):
    df_En_sample = pl.DataFrame({
        "FoldChange_LPPVK":np.random.normal(En_LPPVK_mean_arr, En_LPPVK_stderr_arr),
        "FoldChange_p5":np.random.normal(En_p5_mean_arr, En_p5_stderr_arr),
        "codon_string":df_En_filtered.filter(pl.col("selection")=="LPPVK")["codon_string"]
    })
    front = get_pareto_front_polars(df_En_sample,"FoldChange_LPPVK","FoldChange_p5",True,False)
    fronts.append(front)
    front_frequencies.append(calculate_position_counts(front,frequency=True))

front_variants = pl.concat(fronts)
total_variant_count = len(front_variants)
front_counts = Counter(front_variants['codon_string'])
print(f"""Number of unique Pareto front variants constituting 90% of all variants found on the Pareto front: 
      {calculate_front_variants_number(front_counts,0.9)}""")

front_frequencies = pl.concat(front_frequencies)

mean_front_frequencies = {
    s:(fmean[0], fstd[0]) 
    for s, fmean, fstd in zip (
        substitution_list,
        front_frequencies.mean(),
        front_frequencies.std()) 
    if s not in ["T494K","S540N"]
}

sorted_mean_front_frequencies = {
    k:v 
    for k, v in sorted(
        mean_front_frequencies.items(), 
        key=lambda item: item[1][0],
        reverse=True)
}

print("Sorted mean substitution frequencies within Pareto front variants:")
print(sorted_mean_front_frequencies)

# ========================== PLOTTING =========================
plt.clf()
sn.set_context("talk")
sn.set_style("ticks")

sn.barplot(x=[k for k in sorted_mean_front_frequencies.keys()],
           y=[f[0] for f in sorted_mean_front_frequencies.values()],
           edgecolor='k',
           width=0.6,
           color="#CACACA",
           linewidth=1)
ax = plt.gca()
ax.axhline(2/3, color="#000000", linestyle='--', linewidth=2.5, alpha=0.9)
ax.errorbar(x=[k for k in sorted_mean_front_frequencies.keys()],
            y=[f[0] for f in sorted_mean_front_frequencies.values()],
            yerr=[f[1] for f in sorted_mean_front_frequencies.values()],
            linestyle='none',
            fmt='none',
            elinewidth=1.5,
            ecolor="#292929")

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.tick_params(width=1.5,colors="#000000")
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=15)
ax.margins(x=0.05)
plt.xticks(rotation=90)
tick_labels = ax.get_xticklabels()

for i in range(6):
    tick_labels[i].set_fontweight('bold')

ax.set_yticks([0,0.5,1.0])
fig = plt.gcf()
fig.set_size_inches(5,3.2)
plt.tight_layout()
plt.savefig("pareto_front_substitution_frequency_MC_noT494K_S540N_v3.png",dpi=300,transparent=True)


plt.clf()
sn.set_context("paper")
sn.set_style("ticks")
fig,ax= plt.subplots(nrows=1, ncols=1, figsize=(6,5))

variants_color="#0d89e8"
color_map_name = "turbo_r"
alpha = 1.0
n_colors = 13

# Prepare data for scatter plot with color mapping
df_scatter = pl.DataFrame({
    "LPPVK_enrichment": df_En_filtered.filter(pl.col("selection")=="LPPVK")["linear_score"],
    "p5_enrichment": df_En_filtered.filter(pl.col("selection")=="p5")["linear_score"],
    "substitution_number": df_En_filtered.filter(pl.col("selection")=="p5")["substitution_number"]
}).sample(shuffle=True,fraction=1.0)

# Generate a colormap that spans the full range from 0 to 12 substitutions
colors = sn.color_palette(color_map_name, n_colors=n_colors)
discrete_cmap = mpl.colors.ListedColormap(colors)
bounds = np.arange(-0.5, n_colors + 0.5, 1)
norm = mpl.colors.BoundaryNorm(bounds, discrete_cmap.N)

# Generate the scatterplot
scatter = ax.scatter(df_scatter["LPPVK_enrichment"].to_numpy(), df_scatter["p5_enrichment"].to_numpy(),
                    c=df_scatter["substitution_number"].to_numpy(), cmap=discrete_cmap,
                    norm=norm, s=5, alpha = alpha, edgecolors='none', zorder=2)

tick_locs = np.arange(0.5, n_colors + 0.5, 1)
cax = ax.inset_axes((0.95, 0.42, 0.03, 0.55))
cb = plt.colorbar(scatter,cax=cax,orientation="vertical",spacing="proportional")
cb.set_ticks(np.arange(0, n_colors),labels=np.arange(0, n_colors),size=12)
cb.ax.minorticks_off()

#Mark selected variants on the scatter plot
plot_variant_point(df_En,"00000000000000",colors[0],ax) #AncCQ
plot_variant_point(df_En,"5",colors[1],ax) #AncCQ F462S
plot_variant_point(df_En, "10000000000000",colors[1],ax) #AncCQ L440Y
plot_variant_point(df_En, "00000001000000",colors[1],ax) #AncCQ S471G
plot_variant_point(df_En,"10000001000000",colors[2],ax) #AncCQ L440Y S471G
plot_variant_point(df_En, "11100011000010",colors[6],ax) #AncCQ+6
plot_variant_point(df_En,"10000011000000",colors[3],ax) #AncCQ+3

#Draw Pareto front

df_En_filtered_short = pl.DataFrame({
                        "FoldChange_LPPVK":df_En_filtered.filter(pl.col("selection")=="LPPVK")["linear_score"],
                        "FoldChange_p5":df_En_filtered.filter(pl.col("selection")=="p5")["linear_score"],
                        "variant":df_En_filtered.filter(pl.col("selection")=="LPPVK")["codon_string"]
                        })

front = get_pareto_front_polars(df_En_filtered_short,"FoldChange_LPPVK","FoldChange_p5",True,False)
sn.lineplot(x=front["FoldChange_LPPVK"],y=front["FoldChange_p5"],ax=ax,color="k",linestyle="--",linewidth=1, zorder=1)

for name, spine in ax.spines.items():
    spine.set_linewidth(1.25)
    if name in ['top', 'right']:
        spine.set_visible(False)
ax.tick_params(width=1.5,length=6)

ax.set_xlim((-0.3,4.5))
ax.set_ylim((-0.1,1.55))
ax.set_xticks([0,1,2,3,4],labels=['0','1','2','3','4'],size=14)
ax.set_yticks([0,0.5,1.0,1.5],labels=['0','0.5','1.0','1.5'],size=14)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()

if SAVE_SCATTERPLOT:
    plt.savefig("AncCQ_variant_distribution_pareto_front_no_T494K_S540N_v3.png",dpi=300)
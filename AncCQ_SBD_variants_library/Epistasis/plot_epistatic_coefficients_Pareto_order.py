import polars as pl
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import io
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker


plt.rcParams.update({'font.size': 17})

substitutions = ["L440Y", "T445S", "R446P", "I454V", "T456V", "S459T", "A466V", "S471G", "E482G","T511S", "A528T", "D532E"]
Pareto_ordered_substitution_list = ["L440Y", "S471G","A466V","R446P","D532E","T445S","S459T","T456V","A528T","T511S","I454V","E482G"]

# Define file paths for the two datasets
lppvk_path = "AncCQ_LPPVK_3order_stat_epistasis_model_coefficients.txt"
p5_path = "AncCQ_p5_3order_stat_epistasis_model_coefficients.txt"

def load_and_process_data(file_path):
    """
    Loads coefficients of fitted epistasis model

    file_path - file path of the txt file storing coefficient values of fitted epistasis model

    Returns: Polars DataFrame listing epistasis model coefficients 
    with added values for coefficient order and statistical significance.
    """
    coeff_df = pl.read_csv(file_path, skip_lines=2, separator="\t")
    bonferroni_p_threshold = 0.05 / len(coeff_df)
    # Add the Order column based on comma count
    return coeff_df.with_columns(
        pl.when(pl.col("Term") == "Intercept")
        .then(pl.lit(0))
        .otherwise(pl.col("Term").str.count_matches(",").cast(pl.Int64) + 1)
        .alias("Order"),
        pl.when(pl.col("p-value")<bonferroni_p_threshold)
        .then(True)
        .otherwise(False)
        .alias("Significant")
    )

def generate_second_order_matrix(df, substitutions):
    """
    Generate a second order coefficients matrix from a DataFrame, 
    with non-significant values set to NaN.

    df - polars DataFrame listing second-order epistatic coefficients 
    substitutions - substitution list

    Returns: Second-order epistatic coefficients matrix
    """
    num_substitutions = len(substitutions)
    matrix = np.full((num_substitutions, num_substitutions), np.nan)

    for row in df.iter_rows(named=True):
        if row["Significant"]:
            terms = [int(x) for x in row["Term"].split(',')]
            i, j = terms[0] - 1, terms[1] - 1

            matrix[i, j] = row["Coefficient"]
            matrix[j, i] = row["Coefficient"] #Matrix is symmetric

    return matrix

def reorder_matrix(matrix, old_order, new_order, _type="matrix"):
    """
    Reorder rows and columns of second-order epistatic coefficients matrix

    matrix - second-order epistatic coefficients matrix
    old_order - current matrix order of matrix rows and columns (specified as list of substitutions)
    new_order - new matrix order
    _type - type of data structure to reorder (vector or matrix)

    Returns: Reordered matrix/vector
    """
    index_map = [old_order.index(label) for label in new_order]
    if _type == "vector":
        return matrix[index_map] 
    elif _type == "matrix":
        return matrix[np.ix_(index_map, index_map)]


def prepare_first_order_matrix(df):
    """
    Prepare a 1xN matrix for first-order coefficients

    df - polars DataFrame listing epistatic coefficients 

    Returns: A 1xN matrix of first-order coefficients
    """
    first_order_df = df.filter(pl.col("Order") == 1)
    first_order_matrix = np.full(len(substitutions), np.nan)

    for row in first_order_df.iter_rows(named=True):
        if row["Significant"]:
            term = int(row["Term"])
            first_order_matrix[term - 1] = row["Coefficient"]

    first_order_matrix = np.nan_to_num(first_order_matrix, nan=0)
    return first_order_matrix

def prepare_higher_order_matrix(df):
    """
    Prepare a 1xN matrix containing order normalized sums of higher order (3rd) coefficients,
    partitioning each equally between substitutions participating in given epistatic interaction

    df - polars DataFrame listing epistatic coefficients 

    Returns: A 1xN matrix of higher-order coefficients order-normalized sums
    """
    higher_order_df = df.filter(pl.col("Order")>2)
    matrix = np.zeros((1,len(substitutions)))
    for row in higher_order_df.iter_rows(named=True):
        if row["Significant"]:
            terms = [int(i) for i in row["Term"].split(',')]
            for term in terms:
                matrix[0,term-1] += row["Coefficient"]/len(terms)
    return matrix

def turn_on_spines(ax):
    """
    Turn on all spines of given axes

    ax - matplotlib axes

    Returns: None
    """
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

def map_cmap_to_bars(values,min_val,max_val,cmap):
    """
    Determine colors of the bar plot bars using the heatmap colormap 

    values - first-order epistatic coefficients
    min_val - minimum value used for normalization
    max_val - maximal value used for normalization
    cmap - matplotlib colormap

    Returns: Color list to be applied to a barplot
    """
    norm = Normalize(vmin=min_val, vmax=max_val)
    return [cmap(norm(val)) for val in values]

def prepare_first_order_errors(df, substitutions):
    """
    Prepare a list of first order coefficients standard errors 

    df - a polars dataframe listing model coefficients
    substitutions - substitution list

    Returns: List of first order coefficients standard errors
    """
    err = np.zeros(len(substitutions))
    for row in df.filter(pl.col("Order") == 1).iter_rows(named=True):
        if row["Significant"]:
            err[int(row["Term"]) - 1] = row["Standard Error"]
    return err

# Load and process data from both files
full_df_lppvk = load_and_process_data(lppvk_path)
full_df_p5 = load_and_process_data(p5_path)

# Prepare second-order coefficient matrices for each dataset
second_order_df_lppvk = full_df_lppvk.filter(pl.col("Order") == 2)
second_order_df_p5 = full_df_p5.filter(pl.col("Order") == 2)

heatmap_matrix_lppvk = reorder_matrix(
    generate_second_order_matrix(
        second_order_df_lppvk, 
        substitutions
        ),
    substitutions,
    Pareto_ordered_substitution_list
    )
heatmap_matrix_p5 = reorder_matrix(
    generate_second_order_matrix(
        second_order_df_p5,
        substitutions
        ),
    substitutions,
    Pareto_ordered_substitution_list
    )

# Prepare first-order coefficients 1xN matrices
first_order_matrix_lppvk = reorder_matrix(prepare_first_order_matrix(full_df_lppvk),substitutions,Pareto_ordered_substitution_list,"vector")
first_order_matrix_p5 = reorder_matrix(prepare_first_order_matrix(full_df_p5),substitutions,Pareto_ordered_substitution_list,"vector")

# Prepare higher order normalized sums
higher_order_matrix_lppvk = np.array([reorder_matrix(prepare_higher_order_matrix(full_df_lppvk)[0],substitutions,Pareto_ordered_substitution_list,"vector")])
higher_order_matrix_p5 = np.array([reorder_matrix(prepare_higher_order_matrix(full_df_p5)[0],substitutions,Pareto_ordered_substitution_list,"vector")])

# Extract all significant values for a shared color scale
all_values_lppvk = np.concatenate([
    first_order_matrix_lppvk[~np.isnan(first_order_matrix_lppvk)],
    heatmap_matrix_lppvk[~np.isnan(heatmap_matrix_lppvk)],
    higher_order_matrix_lppvk[~np.isnan(higher_order_matrix_lppvk)]
])

all_values_p5 = np.concatenate([
    first_order_matrix_p5[~np.isnan(first_order_matrix_p5)],
    heatmap_matrix_p5[~np.isnan(heatmap_matrix_p5)],
    higher_order_matrix_p5[~np.isnan(higher_order_matrix_p5)]
])

vmax_lppvk = np.max(np.abs(all_values_lppvk))
vmax_p5 = np.max(np.abs(all_values_p5))
vmin_lppvk = -vmax_lppvk
vmin_p5 = -vmax_p5

# Prepare mask for the diagonal
mask_diag = np.identity(len(Pareto_ordered_substitution_list), dtype=bool)
diagonal_matrix = np.full_like(mask_diag, 0.5, dtype=float)

# --- Plotting the second-order coefficients heatmaps ---

N = len(Pareto_ordered_substitution_list)   

fig = plt.figure(figsize=(12, 12))
gs_fig = gridspec.GridSpec(72, 12)

cmap_diag = ListedColormap(["#9E9E9E"])

colors = ["#0011FC", "#FFFFFF", "#FC0F0F"]
blue_red_cmap = LinearSegmentedColormap.from_list("BlueRed", colors, N=256)

# --- Middle row: Second-Order Heatmaps ---

#LPPVK

ax_lppvk_heatmap = fig.add_subplot(gs_fig[14:54, 0:5])
im_lppvk = sn.heatmap(
    heatmap_matrix_lppvk,
    annot=False, fmt=".2f", cmap=blue_red_cmap,
    xticklabels=Pareto_ordered_substitution_list, yticklabels=Pareto_ordered_substitution_list,
    linewidths=.5, linecolor='black',
    vmin=vmin_lppvk, vmax=vmax_lppvk,
    mask=mask_diag, ax=ax_lppvk_heatmap, cbar=False
)
sn.heatmap(
    diagonal_matrix, cmap=cmap_diag, annot=False,
    xticklabels=Pareto_ordered_substitution_list, yticklabels=Pareto_ordered_substitution_list,
    cbar=False, linewidths=.5, linecolor='black',
    ax=ax_lppvk_heatmap, mask=~mask_diag
)
ax_lppvk_heatmap.set_aspect('equal', adjustable='box')
ax_lppvk_heatmap.set_yticklabels(Pareto_ordered_substitution_list, rotation=0)
ax_lppvk_heatmap.set_xticklabels(["" for s in Pareto_ordered_substitution_list], rotation=90)
ax_lppvk_heatmap.xaxis.tick_top()

#p5
ax_p5_heatmap = fig.add_subplot(gs_fig[14:54, 6:11])

im_p5 = sn.heatmap(
    heatmap_matrix_p5,
    annot=False, fmt=".2f", cmap=blue_red_cmap,
    xticklabels=Pareto_ordered_substitution_list, yticklabels=Pareto_ordered_substitution_list,
    linewidths=.5, linecolor='black',
    vmin=vmin_p5, vmax=vmax_p5,
    mask=mask_diag, ax=ax_p5_heatmap, cbar=False
)
sn.heatmap(
    diagonal_matrix, cmap=cmap_diag, annot=False,
    xticklabels=Pareto_ordered_substitution_list, yticklabels=Pareto_ordered_substitution_list,
    cbar=False, linewidths=.5, linecolor='black',
    ax=ax_p5_heatmap, mask=~mask_diag
)
ax_p5_heatmap.set_aspect('equal', adjustable='box')
ax_p5_heatmap.set_yticklabels(Pareto_ordered_substitution_list, rotation=0)
ax_p5_heatmap.set_xticklabels(["" for s in Pareto_ordered_substitution_list], rotation=90)
ax_p5_heatmap.xaxis.tick_top()

# --- Top Row: First-Order barplots ---
ax_lppvk_first_order = fig.add_subplot(gs_fig[0:12, 0:5])
color_mapping_lppvk = {sub: color for sub, color in zip(Pareto_ordered_substitution_list,map_cmap_to_bars(first_order_matrix_lppvk,vmin_lppvk,vmax_lppvk,blue_red_cmap))}
lppvk_first_order_df = {
    "substitution":Pareto_ordered_substitution_list,
    "coefficient":first_order_matrix_lppvk
}
sn.barplot(lppvk_first_order_df, x="substitution",
           y="coefficient",
           ax=ax_lppvk_first_order,
           palette=color_mapping_lppvk,
           hue="substitution",
           legend=False,
           edgecolor="black",
           linewidth=1,
           width=0.6,
           )

lppvk_errors = reorder_matrix(prepare_first_order_errors(full_df_lppvk, substitutions),
                              substitutions, Pareto_ordered_substitution_list, "vector")

ax_lppvk_first_order.errorbar(
    x=range(len(Pareto_ordered_substitution_list)), 
    y=first_order_matrix_lppvk, 
    yerr=lppvk_errors, 
    fmt='none',
    color='black',
    capsize=2,
    linewidth=1
)

ax_lppvk_first_order.axhline(0, color='black', linewidth=1)

ax_lppvk_first_order.xaxis.set_major_locator(ticker.FixedLocator(range(len(Pareto_ordered_substitution_list))))
ax_lppvk_first_order.set_xticklabels([s for s in Pareto_ordered_substitution_list], rotation=90)
ax_lppvk_first_order.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_lppvk_first_order.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax_lppvk_first_order.tick_params(axis='y', which='major', length=8, width=1.5)
ax_lppvk_first_order.tick_params(axis='y', which='minor', length=4, width=1.5)
ax_lppvk_first_order.set_ylim((-0.25,0.45))
ax_lppvk_first_order.set_xlabel("")
ax_lppvk_first_order.set_ylabel("")

ax_p5_first_order = fig.add_subplot(gs_fig[0:12, 6:11])

color_mapping_p5 = {sub: color for sub, color in zip(Pareto_ordered_substitution_list,map_cmap_to_bars(first_order_matrix_p5, vmin_p5, vmax_p5, blue_red_cmap))}
p5_first_order_df = {
    "substitution":Pareto_ordered_substitution_list,
    "coefficient":first_order_matrix_p5
}
sn.barplot(p5_first_order_df,
           x="substitution",
           y="coefficient",
           ax=ax_p5_first_order,
           palette=color_mapping_p5,
           hue="substitution",
           legend=False,
           edgecolor="black",
           linewidth=1,
           width=0.6
           )

p5_errors = reorder_matrix(prepare_first_order_errors(full_df_p5, substitutions),
                              substitutions, Pareto_ordered_substitution_list, "vector")

ax_p5_first_order.errorbar(
    x=range(len(Pareto_ordered_substitution_list)), 
    y=first_order_matrix_p5, 
    yerr=p5_errors, 
    fmt='none',     
    color='black',    
    capsize=2,
    linewidth=1
)
ax_p5_first_order.axhline(0, color='black', linewidth=1)

ax_p5_first_order.xaxis.set_major_locator(ticker.FixedLocator(range(len(Pareto_ordered_substitution_list))))
ax_p5_first_order.set_xticklabels([s for s in Pareto_ordered_substitution_list], rotation=90)
ax_p5_first_order.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_p5_first_order.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax_p5_first_order.tick_params(axis='y', which='major', length=8, width=1.5)
ax_p5_first_order.tick_params(axis='y', which='minor', length=4, width=1.5)
ax_p5_first_order.set_ylim((-0.25,0.45))
ax_p5_first_order.set_xlabel("")
ax_p5_first_order.set_ylabel("")

# --- Bottom Row: Higher-Order Heatmaps
ax_lppvk_higher_order = fig.add_subplot(gs_fig[51:55, 0:5])
sn.heatmap(
    higher_order_matrix_lppvk,
    annot=False, fmt=".2f", cmap=blue_red_cmap,
    xticklabels=[], yticklabels=[""],
    linewidths=.5, linecolor='black',
    vmin=vmin_lppvk, vmax=vmax_lppvk,
    ax=ax_lppvk_higher_order, cbar=False
)

ax_lppvk_higher_order.set_aspect('equal', adjustable='box')
ax_lppvk_higher_order.set_yticks([])

ax_p5_higher_order = fig.add_subplot(gs_fig[51:55, 6:11])
sn.heatmap(
    higher_order_matrix_p5,
    annot=False, fmt=".2f", cmap=blue_red_cmap,
    xticklabels=[], yticklabels=[""],
    linewidths=.5, linecolor='black',
    vmin=vmin_p5, vmax=vmax_p5,
    ax=ax_p5_higher_order, cbar=False
)
ax_p5_higher_order.set_aspect('equal', adjustable='box')
ax_p5_higher_order.set_yticks([])


turn_on_spines(ax_lppvk_first_order)
turn_on_spines(ax_p5_first_order)
turn_on_spines(ax_lppvk_heatmap)
turn_on_spines(ax_p5_heatmap)
turn_on_spines(ax_lppvk_higher_order)
turn_on_spines(ax_p5_higher_order)

ax_lppvk_cbar = fig.add_subplot(gs_fig[55:59, 0:3])
ax_p5_cbar = fig.add_subplot(gs_fig[55:59, 6:9])
ax_lppvk_cbar.tick_params(width=1.5,length=6)
ax_p5_cbar.tick_params(width=1.5,length=6)
cbar_lppvk = fig.colorbar(ax_lppvk_heatmap.collections[0], cax=ax_lppvk_cbar,orientation='horizontal')
cbar_p5 = fig.colorbar(im_p5.get_children()[0], cax=ax_p5_cbar,orientation='horizontal')
cbar_lppvk.outline.set_linewidth(1.5)
cbar_p5.outline.set_linewidth(1.5)
ax_lppvk_cbar.set_box_aspect(0.08)
ax_p5_cbar.set_box_aspect(0.08)
ax_lppvk_cbar.set_xticks((-0.3,0.0,0.3))
ax_p5_cbar.set_xticks((-0.1,0.0,0.1))

axes = [ax_lppvk_first_order,ax_p5_first_order, ax_lppvk_heatmap, ax_p5_heatmap, ax_lppvk_higher_order, ax_p5_higher_order]

for ax in axes:
    ax.tick_params(width=1.5,length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


ax_lppvk_first_order.set_xlim(-0.5, len(Pareto_ordered_substitution_list) - 0.5)
ax_p5_first_order.set_xlim(-0.5, len(Pareto_ordered_substitution_list) - 0.5)

lppvk_barplot_ticklabels = ax_lppvk_first_order.get_xticklabels()
lppvk_heatmap_yticklabels = ax_lppvk_heatmap.get_yticklabels()
p5_barplot_ticklabels = ax_p5_first_order.get_xticklabels()
p5_heatmap_yticklabels = ax_p5_heatmap.get_yticklabels()

#Mark Pareto-enriched substitutions
for i in range(6):
    lppvk_barplot_ticklabels[i].set_fontweight('bold')
    lppvk_heatmap_yticklabels[i].set_fontweight('bold')
    p5_barplot_ticklabels[i].set_fontweight('bold')
    p5_heatmap_yticklabels[i].set_fontweight('bold')

fig.subplots_adjust(wspace=0.2, hspace=0.2)
fig.align_ylabels()

plt.savefig("epistatic_coefficients_heatmaps_Pareto_ordered.png",dpi=600,transparent=True)
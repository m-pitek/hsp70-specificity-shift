import polars as pl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import linregress
from scipy.special import factorial
from collections import Counter
import seaborn as sn

def get_constant_gap_pos(n_bits=6, block_gap=4.0, row_gap=2.2, node_spacing=1.8):
    """
    Calculate coordinates for each graph node

    n_bits - number of substituted positions considered within graph
    block_gap - spacing between node blocks (differing by the number of substituted positions)
    row_gap - spacing between rows within the same block
    node_spacing - horizontal spacing between nodes

    Returns: An array listing positions of each node and vertical center of each block
    """
    pos = {}
    nodes_by_layer = {i: [] for i in range(n_bits + 1)}
    MAX_WIDTH = 8
    for i in range(2**n_bits):
        layer = bin(i).count('1')
        nodes_by_layer[layer].append(i)

    current_y = 0
    layer_centers = {}
    for layer_idx in range(n_bits + 1):
        nodes = sorted(nodes_by_layer[layer_idx], reverse=True)
        n_total = len(nodes)
        num_rows = int(np.ceil(n_total / MAX_WIDTH))
        block_height = (num_rows - 1) * row_gap
        layer_centers[layer_idx] = current_y - (block_height / 2)

        for i, node_id in enumerate(nodes):
            subrow_idx = i // MAX_WIDTH
            idx_in_subrow = i % MAX_WIDTH
            row_len = min(MAX_WIDTH, n_total - (subrow_idx * MAX_WIDTH))
            x_start = -(row_len - 1) * node_spacing / 2
            x = x_start + (idx_in_subrow * node_spacing)
            y = current_y - (subrow_idx * row_gap)
            pos[node_id] = np.array([x, y])

        current_y -= (block_height + block_gap)
    return pos, layer_centers

def draw_node(ax, pos, color, node_id, size=0.48):
    """
    Draw a node with a 6-bit mutation dot code

    ax - matplotlib axes
    pos - node position
    color - node color
    node_id - node id (based on binary encoded substitutions)
    size - size of the node

    Returns: None
    """
    x, y = pos
    ax.add_patch(Circle((x, y), size, facecolor=color, edgecolor='black', lw=1.0, zorder=4))
    # Extract bits from most significant (index 0) to least significant (index 5)
    mask = [(node_id >> i) & 1 for i in range(5, -1, -1)]

    r_placement = size * 0.62
    dot_radius = size * 0.22

# Start at pi/2 (Top) and move clockwise (-2*pi)
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, 6, endpoint=False)
    for i, angle in enumerate(angles):
        dx, dy = r_placement * np.cos(angle), r_placement * np.sin(angle)
        dot_color = 'black' if mask[i] else 'white'
        ax.add_patch(Circle((x + dx, y + dy), dot_radius, facecolor=dot_color,
                            edgecolor='black', lw=0.4, zorder=5))

def codon_to_id(s):
    """
    Convert the 14-element codon string to a 6-element codon string
    Extracted indices of the 6 substitutions: 0, 1, 2, 6, 7, 12

    s - 14-element codon string

    Returns: 6-element codon string
    """
    bits = [s[0], s[1], s[2], s[6], s[7], s[12]]

    return int("".join(bits), 2)

def draw_curved_arrows(ax, G, pos, node_radius=0.48, arrow_size=10):
    """
    Draw curved arrows marking accessible edges between nodes

    ax - matplotlib axes
    G - networkx directed graph
    pos - array listing graph node positions
    node_radius - radii of the drawn graph nodes
    arrow_size - size of the arrow

    Returns: None
    """
    for u, v in G.edges():
        p1, p2 = pos[u], pos[v]
        diff = p2 - p1
        dist = np.linalg.norm(diff)

        if dist == 0: continue
        unit_vec = diff / dist

        start = p1 + (unit_vec * (node_radius + 0.05))
        end = p2 - (unit_vec * (node_radius + 0.05))

        # Dynamic Curvature Logic:
        # If moving right, curve outward to the right. If left, curve left.
        # This prevents edges from bunching up in the center column.
        x_shift = p2[0] - p1[0]
        if abs(x_shift) < 0.1:
            rad = 0.15  # Bow slightly out for straight-down connections
        else:
            # Curve strength scales slightly with horizontal distance
            rad = 0.1 * np.sign(x_shift)

        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="gray",
                        alpha=0.8,
                        linewidth=0.9,
                        mutation_scale=arrow_size,
                        # This is the magic line that curves the edge
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0,
                        shrinkB=0,
                        zorder=1
                    ))


def id_to_codon_string(variant_id):
    """
    Maps an integer (0-63) to a 14-character string based on the template:
    (0)(1)(2) 0 0 0 (3)(4) 0 0 0 0 (5) 0

    variant_id - number in 0-63 range indicating id of given substitution variant

    Returns: 14-element codon string
    """
    #Convert to 6-bit binary string
    bits = format(variant_id, '06b')

    res = list("00000000000000")

    res[0]  = bits[0]
    res[1]  = bits[1]
    res[2]  = bits[2]
    res[6]  = bits[3]
    res[7]  = bits[4]
    res[12] = bits[5]

    return "".join(res)

def estimate_number_of_accessible_paths_by_Monte_Carlo_sampling(df_6_enrichment, number_of_samples, ddg_tolerance):
    """
    Estimate number of accessible paths within the network representing all allowable transitions between AncCQ SBD variants

    df_6_enrichment - dataframe containing information about all possible combinations of 6 Pareto-enriched substitutions
    number_of_samples - number of Monte Carlo samples
    ddg_tolerance - the tolerance for defining an allowable transition (neutral substitutions) in kcal/mol

    Returns: Edge counts of each edge within the graph observed during Monte Carlo sampling
    """
    edge_counts = Counter()
    accessible_path_fractions = []
    LPPVK_scores = df_6_enrichment["linear_score_LPPVK"].to_numpy()
    LPPVK_SE = df_6_enrichment["linear_SE_LPPVK"].to_numpy()
    p5_scores = df_6_enrichment["linear_score_p5"].to_numpy()
    p5_SE = df_6_enrichment["linear_SE_p5"].to_numpy()
    for s in range(number_of_samples):
        #Draw a sample from the normal distribution
        LPPVK_En = np.random.normal(LPPVK_scores,LPPVK_SE)
        p5_En = np.random.normal(p5_scores,p5_SE)
        ddg_p5_LPPVK = ((p5_En*slope_p5)+intercept_p5) - ((LPPVK_En*slope_LPPVK)+intercept_LPPVK)
        ddg_dict = {k:v for k,v in zip(df_6_enrichment["codon_string"].to_numpy(),ddg_p5_LPPVK)}
        G = nx.DiGraph()
        for i in range(2**N_BITS):
            for bit in range(N_BITS):
                neighbor = i | (1 << ((N_BITS-1)-bit))
                i_ddG = ddg_dict[id_to_codon_string(i)]
                neighbor_ddG = ddg_dict[id_to_codon_string(neighbor)]
                if (neighbor != i) and (neighbor_ddG > i_ddG-ddg_tolerance):
                    G.add_edge(i, neighbor)
                    edge_counts[(i,neighbor)] += 1
        accessible_path_fractions.append(len(list(nx.all_simple_paths(G,0,63)))/factorial(N_BITS))

    sn.set_context("paper")
    sn.set_style("ticks")
    plt.clf()
    plt.hist(accessible_path_fractions,bins=np.linspace(0,1,21), weights=np.ones_like(accessible_path_fractions) / number_of_samples,color="#5c96cc",edgecolor="k",linewidth=0.75)
    plt.axvline(np.mean(accessible_path_fractions),color="k",linestyle="--")
    ax = plt.gca()
    ax.set_xlabel("Fraction of accessible paths")
    ax.set_ylabel("Sample fraction")
    ax.set_xlim([0,1])
    ax.set_ylim([0,0.3])
    plt.gcf().set_size_inches(2.75,1.9)
    plt.tight_layout()
    plt.savefig(f"Fraction_of_accessible_paths_ddg_tol{ddg_tolerance}.png",dpi=600)
    print(f"Mean fraction of accessible paths: {np.mean(accessible_path_fractions)}")
    print(f"SD fraction of accessible paths: {np.std(accessible_path_fractions)}")
    return edge_counts

# --- DATA LOADING AND PROCESSING ---

df_kd = pl.read_csv("AncCQ_SBD_variants_affinities.csv",encoding="utf-8",schema_overrides={"codon string (Ancestral only)": pl.Utf8})
df_kd_12 = df_kd.filter(pl.col("codon string (Ancestral only)").str.contains(r"^.........0...0$"))
df_kd_12 = df_kd_12.rename({'codon string (Ancestral only)':"codon_string"})

df_enrichment = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")

#Perform linear regression between dG and enrichment for both peptides
df_joint_kd_en = df_kd_12.join(df_enrichment.pivot(index="codon_string",on="selection"), on="codon_string")
slope_LPPVK, intercept_LPPVK, r_LPPVK, p_LPPVK, se_LPPVK = linregress(df_joint_kd_en["linear_score_LPPVK"], df_joint_kd_en["LPPVK dG"])
slope_p5, intercept_p5, r_p5, p_p5, se_p5 = linregress(df_joint_kd_en["linear_score_p5"], df_joint_kd_en["p5 dG"])


df_6_enrichment = df_enrichment.filter(
    (pl.col("codon_string") != "5") &
    (pl.col("codon_string").str.contains(r"^...000..0000.0$"))
).pivot(index="codon_string",on="selection")

df_6_enrichment = df_6_enrichment.with_columns(
        (((pl.col("linear_score_p5")*slope_p5)+intercept_p5)-((pl.col("linear_score_LPPVK")*slope_LPPVK)+intercept_LPPVK))
        .alias("ddG_p5_LPPVK")
    )

# Prepare lookup {node_id: score}
nodes_data = (
    df_6_enrichment.with_columns(
        pl.col("codon_string").map_elements(codon_to_id, return_dtype=pl.Int64)
        .alias("node_id")
    )
    .group_by("node_id")
    .agg(
        pl.col("ddG_p5_LPPVK")
        .mean()
    )
    .to_dicts()
)
score_lookup = {d["node_id"]: d["ddG_p5_LPPVK"] for d in nodes_data}

# --- GRAPH CONFIGURATION ---

N_BITS = 6
NODE_RAD = 0.55
pos, label_y = get_constant_gap_pos(N_BITS, block_gap=3.0, row_gap=1.5, node_spacing=1.5)


AncCQ_ddG = df_6_enrichment.filter(pl.col("codon_string")==id_to_codon_string(0))["ddG_p5_LPPVK"][0]
#Calculate number of accessible paths under the assumption that each step produces better selectivity than the previous one
#Use Monte Carlo sampling to assess uncertainty of the estimate
np.random.seed(1001)
number_of_samples = 5000

estimate_number_of_accessible_paths_by_Monte_Carlo_sampling(df_6_enrichment, number_of_samples, 0.0)
#Allow acceptance of substitutions that result in a small decrease in the ddG (0.1 kcal/mol)
edge_counts = estimate_number_of_accessible_paths_by_Monte_Carlo_sampling(df_6_enrichment, number_of_samples, 0.1)

edge_count_threshold = 0.8 #Show only connections that appear in more than 80% of sampled graphs

G = nx.DiGraph()

for edge in edge_counts.items():
    if edge[1]/number_of_samples > edge_count_threshold:
        G.add_edge(edge[0][0],edge[0][1])

# --- GRAPH PLOTTING ---

fig, ax = plt.subplots(figsize=(12, 20))
all_scores = list(score_lookup.values()) if score_lookup else [0, 1]
norm = Normalize(vmin=0, vmax=max(all_scores))
cmap = LinearSegmentedColormap.from_list("WhiteOrange", ["white", "#ff6600"])

draw_curved_arrows(ax, G, pos, node_radius=NODE_RAD, arrow_size=12)

for node_id in range(2**N_BITS):
    score = score_lookup.get(node_id)
    node_color = cmap(norm(score))
    draw_node(ax, pos[node_id], node_color, node_id, size=NODE_RAD)

for layer, y_mid in label_y.items():
    ax.text(-10.0, y_mid, f"Order {layer}", va='center', ha='right', fontweight='bold', fontsize=16)

#Add cbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar to the figure
cbar = fig.colorbar(sm, ax=ax, shrink=0.25, aspect=20, pad=0.05)
cbar.outline.set_linewidth(2)  # Increase for a thicker border
cbar.ax.tick_params(labelsize=20, width=2)

ax.relim()
ax.autoscale_view()
ax.margins(0.1)
ax.set_aspect('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig("AncCQ_AncCQ_plus6_graph_cutoff_0_8_relaxed_accessibility_condition.svg")

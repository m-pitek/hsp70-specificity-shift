import polars as pl
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt

substitutions = ["L440Y", "T445S", "R446P", "I454V", "T456V", "S459T", "A466V", "S471G", "E482G","T511S", "A528T", "D532E"]

# Define file paths for the two datasets
lppvk_path = "AncCQ_LPPVK_3order_stat_epistasis_model_coefficients.txt"
p5_path = "AncCQ_p5_3order_stat_epistasis_model_coefficients.txt"

def load_and_process_data(file_path):
    """
    Load epistatic model coefficients

    file_path - name of the file to load

    Returns: Polars dataframe listing epistatic model coefficients
    """
    coeff_df = pl.read_csv(file_path, skip_lines=2, separator="\t")
    bonferroni_p_threshold = 0.05 / len(coeff_df)
    # Add the Order column based on comma count
    return coeff_df.with_columns(
        pl.when(pl.col("Term") == "Intercept")
        .then(pl.lit(0))
        .otherwise(pl.col("Term").str.count_matches(",").cast(pl.Int64) + 1)
        .alias("Order"),
        pl.when(pl.col("p-value") < bonferroni_p_threshold)
        .then(True)
        .otherwise(False)
        .alias("Significant")
    )

df_epistasis_coeffs_lppvk = load_and_process_data(lppvk_path)
df_epistasis_coeffs_p5 = load_and_process_data(p5_path)

df_measurements = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")
df_measurements = df_measurements.filter((pl.col("codon_string")!="5")&(pl.col("codon_string").str.contains(r".........0...0")))

#Extract 12 positions and cast to statistical (-1,1) coding
indices_to_keep = [i for i, char in enumerate(".........0...0") if char != '0']
df_measurements_with_float_arrays = df_measurements.with_columns(
    (pl.col("codon_string")
    .str.split("")
    .list.gather(indices_to_keep) 
    .cast(pl.List(pl.Float64))
    .alias("codon_array")*2.0)-1.0
)
df_measurements_with_float_arrays = df_measurements_with_float_arrays.pivot(
    on="selection",
    index=["codon_string","codon_array","substitution_number"]
    )

first_order_lppvk_coeffs = df_epistasis_coeffs_lppvk.filter(pl.col("Order")==1).select("Coefficient").to_numpy().flatten().tolist()
first_order_p5_coeffs = df_epistasis_coeffs_p5.filter(pl.col("Order")==1).select("Coefficient").to_numpy().flatten().tolist()
lppvk_model_intercept =  df_epistasis_coeffs_lppvk.filter(pl.col("Term")=="Intercept").select("Coefficient").to_numpy()[0,0].item()
p5_model_intercept =  df_epistasis_coeffs_p5.filter(pl.col("Term")=="Intercept").select("Coefficient").to_numpy()[0,0].item()

df_measurements_with_float_arrays = df_measurements_with_float_arrays.with_columns(
    (
        pl.lit(lppvk_model_intercept) + 
        (pl.col("codon_array") * first_order_lppvk_coeffs).list.sum()    
    ).alias("score_prediction_LPPVK")

)

df_measurements_with_float_arrays = df_measurements_with_float_arrays.with_columns(
    (
        pl.lit(p5_model_intercept) + 
        (pl.col("codon_array") * first_order_p5_coeffs).list.sum()    
    ).alias("score_prediction_p5")

)

#LPPVK
ax_range_LPPVK = (0,4.5)
fig = plt.gcf()
sn.set_theme(style="ticks")
sn.set_context("talk", font_scale=1.3)
g = sn.jointplot(df_measurements_with_float_arrays
                 ,y="linear_score_LPPVK",
                 x="score_prediction_LPPVK",
                 alpha=1,
                 color="#e67300",
                 kind="hist",
                 cbar=True,
                 space=0.3,
                 cbar_kws={'shrink': 0.6, 'label': '', 'pad': -0.05},
                 marginal_ticks=True,
                 joint_kws={'binwidth': (0.1, 0.1)},
                 marginal_kws={'binwidth': 0.1}
                 )
g.figure.set_size_inches(6, 5)
g.ax_joint.set_xlim(ax_range_LPPVK)
g.ax_joint.set_ylim(ax_range_LPPVK)
g.ax_joint.set_yticks([0,1,2,3,4])
g.ax_joint.set_xticks([0,1,2,3,4])
sn.lineplot(x=np.linspace(0,4.5,100),y=np.linspace(0,4.5,100),color="k",linestyle="--",linewidth=1.5)

plt.savefig("LPPVK_selection_measurement_vs_first_order_prediction.png",dpi=300)

plt.clf()

ax_range_p5=(0,1.75)
fig = plt.gcf()

sn.set_theme(style="ticks")
sn.set_context("talk", font_scale=1.3)
g = sn.jointplot(
    data=df_measurements_with_float_arrays,
    y="linear_score_p5",
    x="score_prediction_p5",
    alpha=1,
    color="#28c700",
    kind="hist",
    binrange=(ax_range_p5, ax_range_p5),
    cbar=True,
    marginal_ticks=True,
    ratio=4,
    space=0.3,
    cbar_kws={'shrink': 0.6, 'label': '', 'pad': -0.05},
    joint_kws={'binwidth': (0.04, 0.04)},
    marginal_kws={'binwidth': 0.04}
)

g.figure.set_size_inches(6, 5)
g.ax_joint.set_xlim(ax_range_p5)
g.ax_joint.set_ylim(ax_range_p5)
g.ax_joint.set_xticks([0,0.5,1.0,1.5])
g.ax_joint.set_yticks([0,0.5,1.0,1.5])
cbar_ax = g.figure.axes[-1]

cbar_ax.set_yticks([0, 25,50])

#Add the dashed identity line
line_space = np.linspace(-1, 2.5, 100)
sn.lineplot(x=line_space, y=line_space, color="k", linestyle="--", ax=g.ax_joint,linewidth=1.5)


plt.savefig("p5_selection_measurement_vs_first_order_prediction.png",dpi=300)
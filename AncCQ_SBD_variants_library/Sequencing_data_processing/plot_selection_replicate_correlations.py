import polars as pl
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import math

df = pl.read_parquet("Enrich2/SBD_per_selection_replicate_enrichments_Enrich2.parquet")

#Calculate and plot correlations of enrichments between selection replicates
def plot_enrichment_correlations(df,filename,max_e,axis_interval,alpha=0.1):
    """
    Calculate and plot correlations between AncCQ SBD library selection replicates

    df - polars dataframe listing AncCQ SBD variants enrichments for a given selection
    filename - filename for saving the output plot
    max_e - upper bound for the axes range
    axis_interval - axes tick interval
    alpha - opacity of plotted points

    Returns: None
    """
    plt.clf()
    sn.set_style("ticks")
    sn.set_context("paper")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    replicate_pairs = [(0, 1), (0, 2), (1, 2)]
    for j, i in enumerate(replicate_pairs):
        ax=axes[j]
        x = df.filter(pl.col("selection_replicate")==i[0]).select("linear_score").to_numpy().reshape(-1)
        y = df.filter(pl.col("selection_replicate")==i[1]).select("linear_score").to_numpy().reshape(-1)
        xr = x.reshape(-1,1)
        yr= y.reshape(-1,1)


        reg = LinearRegression().fit(xr, yr)
        r_squared = reg.score(xr,yr)
        pearson_r = np.sqrt(r_squared)
        slope=reg.coef_[0][0]
        intercept=[reg.intercept_[0]]
        print(f"{filename},{i},Pearson r: {pearson_r} Slope: {slope}")

        ax.set_aspect('equal', adjustable='box')

        sn.scatterplot(x=x,y=y,s=4,color="#535353",ax=ax,alpha=alpha)
        sn.lineplot(x=x,y=reg.predict(xr).flatten(),color="red",linewidth=1.5,ax=ax)

        ax.set_ylim(-0.3,max_e)
        ax.set_xlim(-0.3,max_e)
        ax.set_xticks(range(0,math.ceil(max_e),axis_interval))
        ax.set_yticks(range(0,math.ceil(max_e),axis_interval))
        ax.set_xlabel(rf"Replicate {i[0]+1} $E_\mathrm{{N}}$")
        ax.set_ylabel(rf"Replicate {i[1]+1} $E_\mathrm{{N}}$")
        ax.text(0.1, 0.9, f"r = {pearson_r:.2f}", transform=ax.transAxes,fontsize=15, color='k', ha='left', va='center')
        ax.text(0.1, 0.82, f"n = {len(df)/len(df["selection_replicate"].unique()):.0f}", transform=ax.transAxes,fontsize=10, color='k', ha='left', va='center')

    plt.subplots_adjust(right=0.88, wspace=0.3)
    plt.savefig(filename+".png",dpi=300)

#LPPVK selection
plot_enrichment_correlations(df.filter(pl.col("selection")=="LPPVK"),f"./LPPVK_selection_replicates_enrichment_correlation",12.2,2)
plot_enrichment_correlations(df.filter((pl.col("selection")=="LPPVK")&((pl.col("codon_string").str.contains(r".........0...0")) | (pl.col("codon_string")=='5'))),f"./LPPVK_selection_replicates_enrichment_correlation_noT494K_S540N",5,1,alpha=0.2)

#p5 selection
plot_enrichment_correlations(df.filter(pl.col("selection")=="p5"),f"./p5_selection_replicates_enrichment_correlation",3.2,1)
plot_enrichment_correlations(df.filter((pl.col("selection")=="p5")&((pl.col("codon_string").str.contains(r".........0...0")) | (pl.col("codon_string")=='5'))),f"./p5_selection_replicates_enrichment_correlation_noT494K_S540N",2,1,alpha=0.2)

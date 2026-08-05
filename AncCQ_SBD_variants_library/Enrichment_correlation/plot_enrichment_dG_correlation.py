import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from scipy import stats
from adjustText import adjust_text
import polars as pl

# Set seaborn style and context
sn.set_style("ticks")
sn.set_context("paper")

# Error bar settings
ERROR_LINEWIDTH = 1

#Load data
df_enrichment = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")

df_kd = pl.read_csv("../Analysis_of_evolutionary_trajectories_accessibility/AncCQ_SBD_variants_affinities.csv",
                    encoding="utf-8",
                    schema_overrides={"codon string (Ancestral only)": pl.Utf8}
                    )
df_kd = df_kd.rename({'codon string (Ancestral only)':"codon_string"})
df_kd = df_kd.with_columns(pl.col("SBD_name").str.strip_chars())
df_kd = df_kd.with_columns(
    variant = pl.coalesce(
        pl.col("SBD_name").str.extract(r"\((.*)\)", 1),
        pl.col("SBD_name")
    )
)

df_joint_kd_en = df_kd.join(df_enrichment.pivot(index="codon_string",on="selection"), on="codon_string")
print(df_joint_kd_en)

def plot_regplot_with_r2(data, x, y, ax=None, point_color="C0", xlabel=None, ylabel=None,
                    excluded_variants=None, xlim=None, ylim=None, filename=None, annotate_variants=False,
                    y_error=None, x_error=None, show_excluded=True, show_regression_for_whole_dataset=True,
                    figsize=None,yticks=None, **kwargs):
    """
    Create a regression plot and annotate it with r-squared value.

    data - polars DataFrame listing AncCQ SBD variants enrichments and Kd derived free energies of binding
    x - name of the DataFrame column containing x-axis values (Kd derived free energies of binding)
    y - name of the DataFrame column containing y-axis values (enrichments)
    ax - matplotlib axes (default: None; optional)
    point_color - scatter plot point color (default: 'C0'; optional)
    xlabel - x-axis label (default: DataFrame column name; optional)
    ylabel - y-axis label (default: DataFrame column name; optional)
    excluded_variants - list of variants to be excluded from regression (default None; optional)
    xlim - x-axis limits passed as a tuple (xmin, xmax) (default: None optional)
    ylim - y-axis limits passed as a tuple (ymin, ymax) (default: None; optional)
    filename - name of the PNG file to save the plot (default: None; optional)
    annotate_variants - annotates each plotted point when True (default: False; optional)
    y_error - name of the DataFrame column containing y-axis error values (default: None; optional)
    x_error - name of the DataFrame column containing x-axis error values (default: None; optional)
    show_excluded - if True, plot excluded variants as gray points (default: True; optional)
    show_regression_for_whole_dataset - if True, show second regression line for the whole dataset (default True; optional)
    figsize  - figure size (width, height) (default: None; optional)
    yticks - list of y-axis ticks to be marked (default None; optional)
    **kwargs - Additional arguments to pass to seaborn.regplot

    Returns: Matplotlib axes
    """
    plt.close('all')
    if ax is None:
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        ax = plt.gca()

    #Separate included and excluded data
    if excluded_variants is not None:
        is_excluded = pl.col("variant").is_in(excluded_variants)
        included_data = data.filter(~is_excluded)
        excluded_data = data.filter(is_excluded)

    else:
        included_data = data
        excluded_data = None

    if show_regression_for_whole_dataset:
        sn.regplot(data=data, x=x, y=y, ax=ax,
        scatter_kws={"color": point_color, "edgecolor": "black", "linewidths": 0.5, "s":0},
        line_kws={"color": "gray", "linewidth": 1, "linestyle":"--"},
        **kwargs)

    #Prepare regplot with included data only
    sn.regplot(data=included_data, x=x, y=y, ax=ax,
               scatter_kws={"color": point_color, "edgecolor": "black", "linewidths": 0.5, "s":40},
               line_kws={"color": "black", "linewidth": 1},
               **kwargs)

    #Add error bars for included data if provided
    if y_error is not None and y_error in included_data.columns and x_error is not None and x_error in included_data.columns:
        ax.errorbar(included_data[x], included_data[y],
                   xerr=included_data[x_error], yerr=included_data[y_error],
                   fmt='none', ecolor="k", elinewidth=ERROR_LINEWIDTH, capsize=3,
                   capthick=1, alpha=0.7, zorder=0)
    elif y_error is not None and y_error in included_data.columns:
        ax.errorbar(included_data[x], included_data[y],
                   yerr=included_data[y_error],
                   fmt='none', ecolor="k", elinewidth=ERROR_LINEWIDTH, capsize=3,
                   capthick=1, alpha=0.7, zorder=0)
    elif x_error is not None and x_error in included_data.columns:
        ax.errorbar(included_data[x], included_data[y],
                   xerr=included_data[x_error],
                   fmt='none', ecolor="k", elinewidth=ERROR_LINEWIDTH, capsize=3,
                   capthick=1, alpha=0.7, zorder=0)

    #Plot excluded data in gray if provided
    if excluded_data is not None and len(excluded_data) > 0 and show_excluded:
        ax.scatter(excluded_data[x], excluded_data[y],
                  color="#525252", edgecolor="black", linewidths=0.5, alpha=0.6,s=40)

        if y_error is not None and y_error in excluded_data.columns and x_error is not None and x_error in excluded_data.columns:
            ax.errorbar(excluded_data[x], excluded_data[y],
                        xerr=excluded_data[x_error], yerr=excluded_data[y_error],
                        fmt='none', ecolor="k", elinewidth=ERROR_LINEWIDTH, capsize=3,
                        capthick=1, alpha=0.7, zorder=0)
        elif y_error is not None and y_error in excluded_data.columns:
            ax.errorbar(excluded_data[x], excluded_data[y],
                        yerr=excluded_data[y_error],
                        fmt='none', ecolor="k", elinewidth=ERROR_LINEWIDTH, capsize=3,
                        capthick=1, alpha=0.7, zorder=0)
        elif x_error is not None and x_error in excluded_data.columns:
            ax.errorbar(excluded_data[x], excluded_data[y],
                        xerr=excluded_data[x_error],
                        fmt='none', ecolor="k", elinewidth=ERROR_LINEWIDTH, capsize=3,
                        capthick=1, alpha=0.7, zorder=0)

    # Calculate r-squared using only included data
    slope, intercept, r_value, p_value, std_err = stats.linregress(included_data[x], included_data[y])
    r_squared = r_value ** 2

    if show_regression_for_whole_dataset:
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[x], data[y])
            r_squared_all = r_value ** 2

    #Annotate with r-squared (included data)
    ax.text(0.7, 0.95, f'$r^2$ = {r_squared:.3f}',
            transform=ax.transAxes,
            verticalalignment='top')

    #Annotate with r-squared (whole dataset)
    ax.text(0.7, 0.88, f'$r^2$ = {r_squared_all:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            color="gray")

    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Annotate points with variant names
    if annotate_variants:
        texts = []
        for row in data.iter_rows(named=True):
            texts.append(ax.text(row[x], row[y], row["variant"], fontsize=12, ha='center'))

        # Generate avoid object for regression line
        if len(ax.get_lines()) > 0:
            regression_line = ax.get_lines()
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                       avoid_objects=regression_line,
                       expand_points=(1.5, 1.5),
                       expand_text=(1.5, 1.5),
                       force_points=(1, 1))
        else:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    if yticks != None:
        ax.set_yticks(yticks)

    plt.tight_layout()

    # Save to file
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight',transparent=True)
    plt.show()

R=0.001987204 #kcal/(K*mol)
T=298.15 #K

#Kd based free energy of binding
df_joint_kd_en = df_joint_kd_en.with_columns((-R*T*np.log(1/(pl.col("LPPVK KD")*1e-9))).alias("LPPVK_dG"))
df_joint_kd_en = df_joint_kd_en.with_columns((-R*T*np.log(1/(pl.col("p5 KD")*1e-9))).alias("p5_dG"))

#Free energy of binding standard errors
df_joint_kd_en = df_joint_kd_en.with_columns((R*T*((pl.col("LPPVK KD SE")/pl.col("LPPVK KD")))).alias("LPPVK_dG_SE"))
df_joint_kd_en = df_joint_kd_en.with_columns((R*T*(pl.col("p5 KD SE")/pl.col("p5 KD"))).alias("p5_dG_SE"))

excluded_variants = ["AncCQ+14","Sv2","Sv11","Sv12","Sv14"]


plot_regplot_with_r2(df_joint_kd_en,
                x="LPPVK_dG",
                y="linear_score_LPPVK",
                xlabel="$\\Delta$$G$ LPPVK, kcal/mol ",
                ylabel="$E_{\\mathrm{N}}$ LPPVK",
                ci=None,
                point_color="#f2b330",
                excluded_variants=excluded_variants,
                annotate_variants=False,
                y_error="linear_SE_LPPVK",
                x_error="LPPVK_dG_SE",
                show_excluded=True,
                figsize=(3,3),
                filename="En_dG_correlation_LPPVK.png"
                )

plot_regplot_with_r2(df_joint_kd_en,
                x="p5_dG",
                y="linear_score_p5",
                xlabel="$\\Delta$$G$ p5, kcal/mol ",
                ylabel="$E_{\\mathrm{N}}$ p5",
                ci=None,
                point_color="#05d70b",
                excluded_variants=excluded_variants,
                annotate_variants=False,
                y_error="linear_SE_p5",
                x_error="p5_dG_SE",
                show_excluded=True,
                figsize=(3,3),
                filename="En_dG_correlation_p5.png",
                yticks=[0,0.5,1.0,1.5]
                )

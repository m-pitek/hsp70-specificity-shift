from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
from lmfit import Model
from scipy.stats import sem

color_dict = {
"FITC-Ahx-LPPVK":"#f2b330",
"FITC-Ahx-p5":"#05d70b",
"FITC-Ahx-cox4":"#ff0330",
"FITC-Ahx-NR":"#1a1ae6",
}

def monoexponential_dissociation_kinetics_function(x, y0, k, ns):
        """
        Calculate signal value at time x using a monoexponential decay

        x - time since dissociation initiation
        y0 - initial signal value
        k - dissociation constant
        ns - signal value after complete dissociation

        Returns: Signal value at time x
        """
        return (y0 - ns) * np.exp(-k * x) + ns

def fit_monoexponential_dissociation_kinetics(x, y):
    """
    Fit monoexponential decay to a time-series

    x - time
    y - signal

    Returns: Fitted parameters (k - dissociation constant, y0 - signal at time 0, ns - signal after complete dissociation)
    """
    fmodel = Model(monoexponential_dissociation_kinetics_function)
    params = fmodel.make_params(y0=np.max(y),k=1/np.median(x),ns=np.min(y))
    result = fmodel.fit(y,params,x=x)
    return {"k":result.params['k'].value, "k_stderr":result.params['k'].stderr,
            "y0":result.params['y0'].value, "y0_stderr":result.params['y0'].stderr,
            "ns":result.params['ns'].value, "ns_stderr":result.params['ns'].stderr,
            "BIC":result.bic
            }

def write_report_kinetics( output_filename,
                           df,
                           params_list):
    """
    Write report summarizing the fit

    output_filename - name of the output plot .png file
    df - a pandas dataframe containing information about dissociation kinetics
    params_list - list of fitted parameters (koff constant, signal bounds, BIC (goodness of fit))

    Returns: None
    """
    k_off_values, ns_values, BIC_values = [], [], []
    for p in params_list:
        k_off_values.append(p["k"])
        ns_values.append(p["ns"])
        BIC_values.append(p["BIC"])
    with open(output_filename.removesuffix(".png")+"_fit.log","a") as f:
        f.write(f"""
----------------------------------------------------------------------------
----Dissociation kinetics fit----
Receptor: {df.iloc[0]["receptor"]}, {df.iloc[0]["receptor_concentration_nM"]} nM
Ligand: {df.iloc[0]["ligand"]}, {df.iloc[0]["ligand_concentration_nM"]} nM
Competitor ({df.iloc[0]["competitor"]}) concentration: {df.iloc[0]["competitor_concentration_nM"]} nM
Equilibration time: {df.iloc[0]["equilibration_time"]}

FIT DETAILS:
fit type: single exponent
koff +/- stderr (1/h): {np.average(k_off_values)*3600:.3f} +/-{sem(k_off_values)*3600:.3f}, ({", ".join([str(round(k*3600,3)) for k in k_off_values])})
non-specific binding level:  {np.average(ns_values):.3f} +/-{sem(ns_values):.4f}, ({", ".join([str(round(n,3)) for n in ns_values])})
BIC (Bayesian Information Criterion statistic) for each fit: {", ".join([str(round(b,1)) for b in BIC_values])}
----------------------------------------------------------------------------
                """)

def fit_and_plot_kinetics(df,
                  output_filename,
                  anisotropy_range=(0.045,0.135)
                  ):
    """
    Fit monoexponential decay equation and plot dissociation kinetics and associate fit

    df - a pandas dataframe, where each row corresponds to a dissociation kinetics replicate
    output_filename - name of the output .png to save the plot
    anisotropy_range - a tuple specifying anisotropy (y-axis) limits for plotting

    Returns: None
    """
    sn.set_context("talk")
    plot_number = len(df)
    fig, axes = plt.subplots(1,plot_number,figsize=(3.5*plot_number,4),sharey=True)
    fitted_params_list = [] 

    min_anisotropy=anisotropy_range[0]
    max_anisotropy=anisotropy_range[1]
    for row,ax in zip(df.iterrows(),axes):
        monoexponential_fit = fit_monoexponential_dissociation_kinetics(row[1]["time_s"],row[1]["anisotropy"]/1000)
        fitted_params_list.append(monoexponential_fit)
        sn.scatterplot(x=row[1]["time_s"],
                       y=row[1]["anisotropy"]/1000,
                       ax=ax,
                       color=color_dict[row[1]["ligand"]],
                       s=15,
                       edgecolor="k",
                       linewidth=0.3)
        sn.lineplot(x=row[1]["time_s"],
                    y=monoexponential_dissociation_kinetics_function(
                        row[1]["time_s"],
                        monoexponential_fit['y0'],
                        monoexponential_fit['k'],
                        monoexponential_fit['ns']),
                    color="k",
                    ax=ax)
        ax.set_ylim([min_anisotropy-0.005,max_anisotropy+0.005])
    plt.tight_layout()
    plt.savefig(output_filename,dpi=300)
    #Reset log file
    with open(output_filename.removesuffix(".png")+"_fit.log","w") as f:
        f.write("")
    write_report_kinetics(output_filename,df,fitted_params_list)
    plt.close('all')


df = pd.read_parquet("LPPVK_dissociation_kinetics_data.parquet")

for sbd in df.receptor.unique():
    fit_and_plot_kinetics(df.loc[df["receptor"]==sbd],f"{sbd.replace(' ','').replace(',','_')}_LPPVK_dissociation_kinetics.png")
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
from lmfit import Model


color_dict = {
"FITC-Ahx-LPPVK":"#f2b330",
"FITC-Ahx-p5":"#05d70b",
"FITC-Ahx-Cox4":"#ff0330",
"FITC-Ahx-NR":"#1a1ae6",
}

curve_order = ["FITC-Ahx-LPPVK", "FITC-Ahx-p5", "FITC-Ahx-Cox4", "FITC-Ahx-NR"]

FITTED_CURVE_RANGE = [0.1,1.0e5] # x-axis Range for plotting fitted curve in nM


def quadratic_fb_function(x, r0, rb, S, Kd):
    """
    Calculate fluorescence anisotropy value using quadratic binding equation

    x - added SBD concentration
    r0 - fluorescence anisotropy of free ligand
    rb - fluorescence anisotropy of bound ligand (full saturation)
    S - concentration of fluorescently labelled ligand (nM)

    Returns: An anisotropy of the ligand under specified conditions
    """
    P = x
    e0 = P+S+Kd
    e1 = np.sqrt(np.power(P+S+Kd,2)-4*P*S)
    e2 = 2*S
    return r0+(((e0-e1)/e2)*(rb-r0))


def fit_quadratic_fb_function(xdata, ydata, ligand_conc):
    """
    Fit quadratic binding function

    xdata - list of receptor concentrations (nM)
    ydata - list of anisotropy values
    ligand_conc - concentration of fluorescently labelled ligand (nM)

    Returns: A dictionary of fitted parameters
    """
    fmodel = Model(quadratic_fb_function)
    params = fmodel.make_params(r0=min(ydata),rb=max(ydata),S=ligand_conc,Kd=1)
    params['S'].vary=False
    params['Kd'].min  = 0
    params['r0'].min  = 0
    params['rb'].min  = 0
    result = fmodel.fit(ydata,params,x=xdata)

    return {"Kd":result.params['Kd'].value,
            "Kd_stderr":result.params['Kd'].stderr,
            "r0":result.params['r0'].value,
            "r0_stderr":result.params['r0'].stderr,
            "rb":result.params['rb'].value,
            "rb_stderr":result.params['rb'].stderr,
            "BIC":result.bic
            }


def plot_fitted_curve(function, params, color, ax, convert_to_fb=False, bmax=None, bmin=None):
    """
    Plot the fitted line

    function - function used for fitting
    params - list of fitted parameters
    color - plotted line color
    ax - matplotlib axes for plotting
    convert_to_fb - if True, convert fitted line values from anisotropy to fraction bound
    bmax - anisotropy of the bound peptide
    bmin - anisotropy of the free peptide

    Returns: None
    """
    xfit = np.logspace(np.log10(FITTED_CURVE_RANGE[0]),np.log10(FITTED_CURVE_RANGE[1]),100)
    yfit = function(xfit,*params)
    if convert_to_fb:
        if bmax != None and bmin != None:
            yfit = (yfit-bmin)/(bmax-bmin)
        else:
            print("WARNING: bmax or bmin not passed for fit plotting")
    else:
        if bmin != None:
            yfit = yfit-bmin
        else:
            print("WARNING: bmin not passed for fit plotting")
    sn.lineplot(x=xfit,y=yfit,markers=False,color=color,linewidth=1.5,ax=ax)



def write_report(output_filename, curve, params, fit_type):
    """
    Write report summarizing the fit

    output_filename - name of the output plot png file
    curve - dictionary providing information about binding curve
    params - dictionary containing fitted parameters
    fit_type - equation type used for curve fitting

    Returns: None
    """
    with open(output_filename.removesuffix(".png")+"_fit.log","a") as f:
        f.write(f"""
----------------------------------------------------------------------------
----Binding curve----
Receptor: {curve["receptor"]}
Ligand: {curve["ligand"]}
Ligand concentration: {curve["ligand_concentration"]} nM
Equilibration time: {curve["incubation_time"]}

FIT DETAILS:
fit type: {fit_type}
excluded measurements: {curve["excluded_measurements_string"] if "excluded_measurements_string" in curve.keys() else 'None'}
Kd (val ± stderr): {params["Kd"]} ±{params["Kd_stderr"]}
r0 (val ± stderr): {params["r0"]} ±{params["r0_stderr"]}
rb (val ± stderr): {params["rb"]} ±{params["rb_stderr"]}
BIC (Bayesian Information Criterion statistic): {params["BIC"]}
----------------------------------------------------------------------------
                """)

def fit_and_plot_binding_curves(
                curve_info,
                curve_dfs,
                output_filename,
                plot_dimensions
                ):
    """
    Fit and plot binding curves

    curve_info - list of dictionaries providing information about each binding curve
    curve_dfs - list of dataframes containing each binding curve
    output_filename - name of .png file to save the plot
    plot_dimensions - dimensions of the matplotlib plot

    Returns: None
    """

    plt.close('all')
    plt.rcdefaults()
    #Reset log file
    with open(output_filename.removesuffix(".png")+"_fit.log","w") as f:
        f.write("")

    sn.set_context("talk")
    plt.rcParams.update({
    'font.family': 'sans-serif',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    })

    fig, (ax1) = plt.subplots(1,1,figsize=plot_dimensions)

    for i in range(len(curve_dfs)):
        df = curve_dfs[i].copy()
        df["receptor_concentration_nM"] = pd.to_numeric(df["receptor_concentration_nM"], errors='coerce')
        df["measured_anisotropy_value"] = df["measured_anisotropy_value"] / 1000 #convert mili-anisotropy to anisotropy
        curve_dfs[i] = df

    #Curve fitting and plotting
    it = 0
    for info,df in zip(curve_info, curve_dfs):
        #Fit quadratic binding function
        params = fit_quadratic_fb_function(df["receptor_concentration_nM"].copy(),df["measured_anisotropy_value"].copy(),info["ligand_concentration"])
        df["fraction_bound"] = (df["measured_anisotropy_value"]-params["r0"]) / (params["rb"]-params["r0"])
        bmin=params["r0"]

        #Plot the fit
        plot_fitted_curve(
            quadratic_fb_function,
            [
                params["r0"],
                params["rb"],
                info["ligand_concentration"
            ],
            params["Kd"]], 
            color_dict[info["ligand"]],
            ax1,
            convert_to_fb=True,
            bmin=params["r0"],
            bmax=params["rb"])
        write_report(output_filename,info,params,"quadratic_binding_function")

        #Plot datapoints and means
        df["receptor_concentration_nM"] = pd.to_numeric(df["receptor_concentration_nM"], errors='coerce')
        df_pos = df[df["receptor_concentration_nM"] > 0]

        kws = {"s": 15, "facecolor": "none", "linewidth": 1}
        sn.scatterplot(data=df_pos, x="receptor_concentration_nM", y="fraction_bound", edgecolor=color_dict[info["ligand"]], zorder=10, ax=ax1, **kws)

        kws = {"s": 30, "facecolor": color_dict[info["ligand"]], "linewidth": 1}
        df_group = df_pos.groupby("receptor_concentration_nM").mean(numeric_only=True).reset_index()
        sn.scatterplot(data=df_group, x="receptor_concentration_nM", y="fraction_bound", edgecolor="#000000", zorder=20, ax=ax1, **kws)

        it += 1

    ax1.set_xscale("log")
    ax1.minorticks_off()
    ax1.set_ylim([-0.15,1.1])
    ax1.set_xticks([0.1,1,10,100,1000,10000,1e5],["10$^{-4}$","10$^{-3}$","0.01","0.1","1","10","100"])
    ax1.set_xlim([0.05,1.5e5])
    ax1.set_xlabel("SBD, μM",fontsize=20)
    ax1.set_ylabel("Fraction bound",fontsize=20)


    plt.tight_layout()
    plt.savefig(output_filename,dpi=300,transparent=True)
    plt.close('all')


df_measurements = pd.read_csv("binding_curve_anisotropy_measurements.csv")
df_metadata = pd.read_csv("binding_curve_list.csv")

for sbd in df_metadata["receptor"].unique():
    curve_info_list = []
    curve_df_list = []
    for curve in df_metadata.loc[df_metadata["receptor"]==sbd].iterrows():
        curve_info = {
            "receptor":sbd,
            "ligand":curve[1]["ligand"],
            "ligand_concentration":curve[1]["ligand_concentration_nM"],
            "incubation_time":curve[1]["equilibration_time"],
        }

        if df_measurements.loc[df_measurements["experiment_id"]==curve[1]["experiment_id"],"outlier"].any():
            curve_info["excluded_measurements_string"] = ",".join(
                [
                    f"({row[1]['receptor_concentration_nM']} nM, rep. {row[1]['replicate']})" 
                    for row in df_measurements.loc[
                        (df_measurements["experiment_id"]==curve[1]["experiment_id"])&
                        (df_measurements["outlier"]==True),:].iterrows()
                ]
            )
        curve_info_list.append(curve_info)
        curve_df_list.append(df_measurements[
            (df_measurements["experiment_id"]==curve[1]["experiment_id"]) & 
            (df_measurements["outlier"]==False)
            ])

    paired = sorted(
        zip(curve_df_list, curve_info_list),
        key=lambda pair: curve_order.index(pair[1]["ligand"])
    )

    sorted_curve_df_list, sorted_curve_info_list = map(list, zip(*paired))
    fit_and_plot_binding_curves(sorted_curve_info_list, sorted_curve_df_list,f"{sbd.replace(' ','').replace(',','_')}_binding_curves.png",(5.5,3.75))
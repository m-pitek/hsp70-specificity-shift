import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
from whittaker_eilers import WhittakerSmoother
from matplotlib import rcParams

kbt_to_kcal = 0.5922

def load_data(file_list, start_step=0, end_step=None):
    """
    Load PLUMED calculated distances between centers of mass (COMs)

    file_list - list of plumed output files to be read and concatenated
    start_step - discard all values before this time step (default: 0) 
    end_step - discard all values after this time_step (inclusive; default: None)

    Returns: a pandas DataFrame listing COM distances
    """
    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep=r"\s+", skiprows=1, skipinitialspace=True, header=None)
        if end_step == None:
            dfs.append(df.loc[df[0] >= start_step, :])
        else:
            dfs.append(df.loc[(df[0] >= start_step) & (df[0] <= end_step), :])
    return pd.concat(dfs, ignore_index=True)

def calc_block_pmfs(data, block_number, bin_width, range_start, range_end):
    """
    Calculate block PMFs (Potential of Mean Force)

    data - pandas DataFrame listing values for probability distribution calculation
    block_number - number of equal blocks to split data into
    bin_width - width of a bin for probability distribution calculation
    range_start - lower limit for the first bin of probability distribution
    range_end - upper limit for the last bin of probability distribution

    Returns: a list of calculated block PMFs and a list of their standard deviations
    """
    pmfs = [] 
    for i in range(0,block_number):
        block_frame_number = int(len(data)/block_number)
        bins = np.arange(range_start, range_end + bin_width, bin_width)
        block_start = int(i*block_frame_number)
        block_end = int((i+1)*block_frame_number)
        if block_end > len(data):
            block_end = int(len(data))
        counts, bin_edges = np.histogram(data[block_start:block_end], bins=bins, density=True)
        pmf = -np.log(counts, where=counts > 1e-3)*kbt_to_kcal
        pmf[counts<1e-3] = np.nan
        #Shift PMF to 0
        pmf = pmf - pmf[np.nanargmin(pmf)]
        pmfs.append(pmf)
    
    return pmfs, np.nanstd(pmfs,axis=0, ddof=1)

def plot_pmf(
        file_list,
        pmf_limits, 
        xlimits, 
        ylimits, 
        xticks, 
        yticks, 
        output_filename, 
        start_step=0,
        end_step=None, 
        block_number=3, 
        bin_width=0.1, 
        height_ratio=2, 
        fig_size=(4,3), 
        color="b",
        ylimits_prob=None):
    """
    Calculate and plot blokc-averaged potential of mean force (PMF) 
    profile based on probability distribution of given variable

    file_list - list of plumed output files to be read and concatenated
    pmf_limits - variable range for profile calculation
    xlimits - shared x-axis range for plotting
    ylimits - PMF y-axis range for plotting
    xticks - shared x-axis ticks
    yticks - PMF y-axis tikcs
    output_filename - filename for saving the plotted profile
    start_step - first frame to be used for PMF calculation (from each file)
    end_step - last frame to be used for PMF calculation (from each file)
    block_number - number of blocks used for calculating profile's SEM

    Returns: None
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 13
    rcParams['axes.labelsize'] = 13
    rcParams['axes.titlesize'] = 13
    plt.rc('axes',labelsize=13)

    com_distances = load_data(file_list, start_step, end_step)
    data = com_distances[1].dropna()
    block_pmfs, block_pmf_std = calc_block_pmfs(data,block_number,bin_width,pmf_limits[0],pmf_limits[1])
    block_pmf_sem = block_pmf_std / np.sqrt(block_number)

    #Calculate and smooth PMF based on the whole dataset, use block-calculated standard deviations
    bins = np.arange(pmf_limits[0], pmf_limits[1] + bin_width, bin_width)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    pmf = -np.log(counts, where=counts > 1e-3) * kbt_to_kcal
    pmf[counts < 1e-3] = np.nan
    weights = np.where(np.isnan(pmf), 0.0, 1.0)
    counts_filled = np.nan_to_num(counts,nan=0.0)
    pmf_filled = np.nan_to_num(pmf, nan=0.0)
    sem_filled = np.nan_to_num(block_pmf_sem, nan=0.0)

    smoother = WhittakerSmoother(lmbda=5, order=2, data_length=len(pmf_filled), weights=weights)

    def clip_extrapolation(arr, counts):
        #Fill extrapolated regions of the array (0 counts) with np.nan
        valid_indices = np.where(counts > 1e-3)[0]
        if len(valid_indices) > 0:
            arr[:valid_indices[0]] = np.nan
            arr[valid_indices[-1]+1:] = np.nan
            return arr
        else:
            return np.array([np.nan for i in range(len(arr))])

    counts_smooth = clip_extrapolation(np.clip(np.array(smoother.smooth(counts_filled)),a_min=0.0, a_max=None),counts)
    pmf_smooth = clip_extrapolation(np.array(smoother.smooth(pmf_filled)),counts)
    sem_smooth = clip_extrapolation(np.abs(np.array(smoother.smooth(sem_filled))),counts)

    #shift PMF to 0
    min_idx = np.nanargmin(pmf_smooth)
    pmf_smooth = pmf_smooth - pmf_smooth[min_idx]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #Plot block PMFs
    for pmf in block_pmfs:
        sn.lineplot(x=bin_centers,y=pmf,linewidth=1)
    
    plt.xlabel("Distance [Å]")
    plt.ylabel("Free energy \nkcal/mol]")
    fig = plt.gcf()
    fig.set_size_inches(5, 3)
    plt.tight_layout()
    plt.savefig(output_filename.replace(".png", "_block_profiles.png"),dpi=300)
    plt.close('all')

    plot,axes=plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, height_ratio], 'hspace': 0.1})
    prob_plot_id=0
    energy_plot_id=1

    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
    plot.set_size_inches(fig_size)
    axes[energy_plot_id].axhline(y = 0, color = 'k', linestyle = '--', linewidth=1, zorder=-100)

    # Plot the smoothed PMF
    axes[energy_plot_id].plot(bin_centers, pmf_smooth, linewidth=1.5, color=color, zorder=0)
    axes[energy_plot_id].fill_between(bin_centers, pmf_smooth - sem_smooth, pmf_smooth + sem_smooth, 
                                      alpha=0.2, edgecolor=color, facecolor=color, linewidth=1, zorder=0)
    axes[prob_plot_id].plot(bin_centers, counts_smooth, linewidth=0.5, color=color, zorder=0)
    axes[prob_plot_id].fill_between(bin_centers, 0, counts_smooth, alpha=0.2, edgecolor=color, 
                                    facecolor=color, linewidth=1, zorder=0)
   
    axes[energy_plot_id].set_xlabel('Distance [Å]', size=11)
    axes[energy_plot_id].set_ylabel('Free energy \n[kcal/mol]' , size=11)
    axes[prob_plot_id].set_ylabel('p', size=11)
    axes[energy_plot_id].set_xlim(xlimits)
    axes[prob_plot_id].set_xlim(xlimits)
    axes[energy_plot_id].set_ylim(ylimits)
    axes[energy_plot_id].set_xticks(xticks)
    axes[prob_plot_id].set_xticks(xticks)
    axes[energy_plot_id].set_yticks(yticks)
    axes[energy_plot_id].label_outer()
    axes[prob_plot_id].label_outer()
    if ylimits_prob != None:
        axes[prob_plot_id].set_ylim(ylimits_prob)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1)
    plt.tight_layout()
    plot.savefig(output_filename,dpi=300,transparent=True)
    plt.close('all')

#AncCQ-LPPVK
plot_pmf(
        file_list=[f"pep_beta_4_com_dist_AncCQ_LPPVK_rep{i}.txt" for i in range(1,4)],
        pmf_limits=(4,10),
        xlimits=(4,10),
        ylimits=(-0.4,4.8),
        xticks=(4,6,8,10),
        yticks=(0,1,2,3,4),
        output_filename="AncCQ_LPPVK_Nter_beta4_dist_PMF.png",
        start_step=1e5, 
        end_step=1.01e7, 
        color="#1ba887",
        ylimits_prob=[0,1.5]
    )

#AncCQ+14-LPPVK
plot_pmf(
        file_list=[f"pep_beta_4_com_dist_AncCQ_plus14_LPPVK_rep{i}.txt" for i in range(1,4)],
        pmf_limits=(4,10),
        xlimits=(4,10),
        ylimits=(-0.4,4.8),
        xticks=(4,6,8,10),
        yticks=(0,1,2,3,4), 
        output_filename="AncCQ_plus14_LPPVK_Nter_beta4_dist_PMF.png", 
        start_step=1e5, 
        end_step=1.01e7, 
        color="#3a73ad",
        ylimits_prob=[0,1.5]
    )
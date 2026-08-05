import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from pathlib import Path

def moving_average_numpy(data, window=500):
    """
    Compute moving average using NumPy convolution.

    data - 1-dimensional numpy array containing RMSD values
    window - window length used for average calculation

    Returns: numpy array containing the moving average
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def plot_RMSD(results_dir, ma_window_length=500, traj_range=None):
    """
    Plot RMSD traces

    results_dir - path to the directory storing saved RMSD traces
    ma_window_length - length of moving average window to be applied to loaded RMSD traces
    traj_range - range of simulations steps to plot (0.1 ns/step)

    Returns: None
    """
    simulations = ["AncCQ_SBD_LPPVK", "AncCQ_plus14_SBD_LPPVK"]
    replicas = ["rep1", "rep2", "rep3"]
    groups = ["complex","SBDβ","peptide"]
    
    group_to_col_dict = {
        "SBDβ": 2,
        "SBD": 3,
        "complex": 4,
        "peptide": 5
    }

    processed_data = {g: [[] for _ in range(len(simulations))] for g in groups}
    time_axes = [[None for _ in range(len(replicas))] for _ in range(len(simulations))]

    sn.set_context("talk")
    for i, sim in enumerate(simulations):
        for j, rep in enumerate(replicas):
            file_path = Path(results_dir) / f"{sim}_{rep}_rmsd_results.npy"
            arr = np.load(file_path)
            if traj_range != None:
                arr = arr[traj_range[0]:traj_range[1],:]
            
            #Handle time series alignment for a convolution based moving average
            trim_total = ma_window_length - 1
            trim_start = trim_total // 2
            trim_end = trim_total - trim_start
            
            raw_time = arr[:, 1] / 1000000.0 # ps to μs
            
            if trim_end == 0:
                time_axes[i][j] = raw_time[trim_start:]
            else:
                time_axes[i][j] = raw_time[trim_start:-trim_end]

            for g in groups:
                raw_val = arr[:, group_to_col_dict[g]]
                smoothed = moving_average_numpy(raw_val, window=ma_window_length)
                processed_data[g][i].append(smoothed)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey='row')
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    colors = ["#3498db", "#e74c3c", "#2ecc71"] # Colors for rep1, rep2, rep3

    global_ymax = 0
    for row_idx in range(len(simulations)):
        for group_name in groups:
            for rep_smooth in processed_data[group_name][row_idx]:
                if len(rep_smooth) > 0:
                    global_ymax = max(global_ymax, np.max(rep_smooth))

    y_max = global_ymax * 1.1
    for row_idx in range(len(simulations)):
        for col_idx, group_name in enumerate(groups):
            ax = axes[row_idx, col_idx]
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            for rep_idx in range(len(replicas)):
                y = processed_data[group_name][row_idx][rep_idx]
                x = time_axes[row_idx][rep_idx]
                
                if x is not None and len(x) == len(y):
                    ax.plot(x, y, color=colors[rep_idx], 
                            label=f"rep{rep_idx+1}", alpha=0.8, linewidth=1.5
                            )
            ax.set_xlim(0, 10.5)
            ax.set_ylim(0, y_max)
    
    output_plot = Path(results_dir) / "RMSD_SBD_LPPVK_simulations.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    RESULTS_DIR = Path(".")
    plot_RMSD(RESULTS_DIR, ma_window_length=500, traj_range=(0,101000))
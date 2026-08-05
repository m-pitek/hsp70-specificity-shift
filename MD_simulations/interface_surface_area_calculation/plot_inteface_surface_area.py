import seaborn as sn 
from matplotlib import pyplot as plt
import polars as pl
import numpy as np


data_files = [
    "AncCQ_SBD_LPPVK_interface_area.parquet",
    "AncCQ_plus14_SBD_LPPVK_interface_area.parquet",
]

frame_timestep = 100 #ps
time_range = (1e5,1.01e7)#0.1-10.1 μs

data = []

for f in data_files:
    df = pl.read_parquet(f).\
    filter(
        (pl.col("frame") > (time_range[0]/frame_timestep)) & 
        (pl.col("frame") <= (time_range[1]/frame_timestep)) 
        )
    data.append(df)


interface_area_mean = [d.mean()["interface_area_A2"][0] for d in data]
interface_area_SEM = [d.group_by("traj_file").mean().std()["interface_area_A2"][0]/np.sqrt(3) for d in data]

sn.set_style("ticks")
sn.set_context("talk")

xlabels = ["AncCQ-\nLPPVK", "AncCQ+14-\nLPPVK"]

ax = plt.gca()
sn.barplot(
    x=xlabels, 
    y=interface_area_mean, 
    palette=["#aad28e", "#8eaadc"],
    linewidth=1.5,      
    edgecolor="black",
    width=0.6
)
x_pos = np.arange(len(xlabels))
ax.errorbar(
    x=x_pos, 
    y=interface_area_mean, 
    yerr=interface_area_SEM, 
    fmt='none',      
    c='black',       
    capsize=0,
    linewidth=4
)

for spine in ax.spines.values():
    spine.set_linewidth(2.5)

ax.tick_params(width=2.5)

ax.tick_params("x", rotation=90)
ax.tick_params("y",labelsize=20)
ax.set_ylim([600,760])
ax.set_ylabel("Mean interface area, Å$^2$")
fig = plt.gcf()
fig.set_size_inches((3.4,6))
ax.margins(0.2)
plt.tight_layout()
plt.savefig("SBD-LPPVK_interface_area_MD.png",dpi=300)
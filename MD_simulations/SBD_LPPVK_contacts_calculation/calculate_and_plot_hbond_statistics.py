import polars as pl
import pickle
import MDAnalysis as mda
import numpy as np
import seaborn as sn 
from matplotlib import pyplot as plt
import re
import matplotlib 
matplotlib.use("Agg")

replicate_list = range(1,4)
fingerprint_files = [
    [f"AncCQ_SBD_LPPVK_rep{r}_prolif_interaction_fingerprint.pickle" for r in replicate_list],
    [f"AncCQ_plus14_SBD_LPPVK_rep{r}_prolif_interaction_fingerprint.pickle" for r in replicate_list],
]
pdb_files = [
    f"AncCQ_LPPVK_ions_minim3.part0001.pdb",
    f"AncCQ_plus14_LPPVK_ions_minim3.part0001.pdb",
]

peptide_res_selection = "resid 130 to 139" #whole peptide

load_precomputed_counts = True
load_precomputed_counts_SC_aggregated = True

xlabels = ["AncCQ-LPPVK", "AncCQ+14-LPPVK"]

def load_and_merge_ifps(filenames, start_step=0, end_step=None):
    """
    Load multiple ProLIF ifp dictionaries, filter them by a common frame range
    and merge them into a single dictionary

    filenames - list of paths to the .pickle files
    start_step - first frame of each ifp to be included in the merged one
    end_step - last frame of each ifp to be included in the merged one (inclusive)

    Returns: merged ifp dictionary
    """
    merged_ifp = {}
    global_frame_idx = 0

    for filename in filenames:
        print(f"Loading interaction fingerprint from {filename}...")
        with open(filename, "rb") as f:
            loaded_ifp = pickle.load(f)

        frame_keys = sorted(loaded_ifp.keys())
        frames_added_from_file = 0

        for key in frame_keys:

            if key < start_step:
                continue
            if end_step is not None and key > end_step:
                break

            merged_ifp[global_frame_idx] = loaded_ifp[key]
            global_frame_idx += 1
            frames_added_from_file += 1

        print(f"-> Added {frames_added_from_file} frames. (Global total: {global_frame_idx})")

    print("All files loaded and merged successfully")
    return merged_ifp

def count_hydrogen_bonds(
        ifp,
        pdb,
        res_selection,
        counting_reference="peptide",
        extract_in_blocks=False,
        block_number=3, 
        group_aggregation=False, 
        frame_mask=None
        ):
    """ 
    Count hydrogen bonds from a ProLIF interaction fingerprint
    This function reads hydrogen bond interactions from an interaction fingerprint
    dictionary, and returns mean counts either across all trajectories
    or averaged per block. The counting is performed using eihter absolute numbers of
    detected hbonds or peptide chemical group aggregation (backbone carbonyl, amide and 
    sidechain if capable of hydrogen bonding).

    ifp - ProLIF interaction fingerprint dictionary
    pdb - path to the .pdb/.tpr file used to define chemical groups of the peptide
    res_selection - MDAnalysis selection string for residues used for hydrogen bond counting
    counting_reference - choose whether counting should performed with respect to peptide or sbd
    extract_in_blocks - if True, return a separate list for each block/trajectory
    block_number - number of equal-sized blocks to split the trajectory into
    group_aggregation - if True, aggregate hydrogen bonds by chemical group
    frame_mask - if not None, count hbonds only within frames specified by the boolean mask

    Returns:
      dict or list: If extract_in_blocks is False, returns a dict mapping peptide
      atom/group keys to mean hydrogen bond counts per frame. If True, returns a
      list of block-average dicts.
    """
    u = mda.Universe(topology=pdb)
    ref_atoms = u.select_atoms(res_selection)
    hbond_types = ["HBAcceptor","HBDonor"]
    hbond_counts = {}
    counts_list = []

    atom_type_mapping = {
                    "N":['N'],
                    "O":["O","OT1","OT2"],
                    "SC":["OG","NZ","ND1","NE2","SG","OE1","OE2","OG1", "NE","NH1","NH2","OD1","OD2"]
                    }

    if extract_in_blocks:
        block_length = int(len(ifp)/block_number)
        n_frames = len(ifp)
        base_length = n_frames // block_number
        rest = n_frames % block_number
        block_sizes = [base_length + 1 if i < rest else base_length for i in range(block_number)]
        block_boundaries = np.cumsum(block_sizes).tolist()
        
    if type(frame_mask) == type(None):
        frame_mask = [True for i in range(len(ifp))]
    if counting_reference == "peptide":
        interaction_indices = "ligand"
    else:
        interaction_indices = "protein"
    
    analyzed_frames = 0
    for frame, mask_val in zip(list(ifp.items()), frame_mask):
        hbond_set = set()
        if mask_val:
            for res_pair in list(frame[1].keys()):
                for interaction in list(frame[1][res_pair].keys()):
                    if interaction in hbond_types:
                        group_key = ""
                        if counting_reference == "peptide":
                            for index in frame[1][res_pair][interaction][0]['parent_indices'][interaction_indices]:
                                atom = ref_atoms[index]
                                atom_name = ""
                                if group_aggregation:
                                    for item in atom_type_mapping.items():
                                        if atom.name in item[1]:
                                            atom_name = item[0]
                                    if atom.type != 'H':
                                        group_key = f"{atom.resname}{atom.resid}-{atom_name}"
                                        hbond_set.add(group_key)
                                        break        
                                else:
                                    if atom.type != 'H':
                                        group_key = f"{atom.resname}{atom.resid}-{atom_name}"
                                        hbond_counts[group_key] = hbond_counts.get(group_key,0)+1
                                        break  
                        else:
                            sbd_resid = (res_pair[1].number)-1
                            for index in  frame[1][res_pair][interaction][0]['indices'][interaction_indices]:
                                atom = u.residues[sbd_resid].atoms[index]
                                atom_name = ""
                                if group_aggregation:
                                    for item in atom_type_mapping.items():
                                        if atom.name in item[1]:
                                            atom_name = item[0]
                                    if atom.type != 'H':
                                        group_key = f"{atom.resname}{atom.resid}-{atom_name}"
                                        hbond_set.add(group_key)
                                        break        
                                else:
                                    if atom.type != 'H':
                                        group_key = f"{atom.resname}{atom.resid}-{atom_name}"
                                        hbond_counts[group_key] = hbond_counts.get(group_key,0)+1
                                        break  
            analyzed_frames+=1
        if group_aggregation:
            for group_key in hbond_set:
                hbond_counts[group_key] = hbond_counts.get(group_key,0)+1

        if extract_in_blocks:
            if (frame[0]+1) in block_boundaries:
                counts_list.append({k:v/analyzed_frames for k,v in hbond_counts.items()})
                hbond_counts = {}
                analyzed_frames = 0
    if not extract_in_blocks:
        return {k:v/analyzed_frames for k,v in hbond_counts.items()}
    else:
        return counts_list

def plot_hbond_counts_by_chemical_group(hbond_count_dict, peptide, filename):
    """
    Plot the mean number of hydrogen bonds per peptide chemical group.
    
    hbond_count_dict - dictionary storing per system dictionaries of hbond counts aggregated per chemical group
    peptide - name of the peptide present in the complex (e.g. LPPVK)
    filename - name of the .png file to save the output plot

    Returns: None
    """
    all_systems_dfs = []

    for system_name, blocks in hbond_count_dict.items():
        if not blocks:
            continue
        
        df_sys = pl.from_dicts(blocks)
        df_sys = df_sys.unpivot(variable_name="Group", value_name="Count")
        df_sys = df_sys.drop_nulls("Count")

        df_sys = df_sys.with_columns([
            pl.lit(system_name).alias("System"),
            pl.lit(peptide).alias("Peptide")
        ])
        all_systems_dfs.append(df_sys)

    full_df = pl.concat(all_systems_dfs)

    palette = {
        "AncCQ-LPPVK": "#aad28e",
        "AncCQ+14-LPPVK": "#8eaadc",
    }

    peptide_data = full_df.filter(pl.col("Peptide") == peptide).to_pandas()
    
    unique_groups = peptide_data["Group"].unique().tolist()
    
    # Define a sorting key function
    def extract_res_num(group_name):
        # Find the sequence of digits in the string (e.g., "134" from "PRO134-O")
        match = re.search(r'\d+', group_name)
        res_num = int(match.group()) if match else 0
        # Return tuple: sort by residue number first, then alphabetically by the full name
        return (res_num, group_name) 
        
    ordered_groups = sorted(unique_groups, key=extract_res_num)
    
    plt.figure(figsize=(12, 6))
    
    sn.barplot(
        data=peptide_data, x="Group", y="Count", hue="System",
        order=ordered_groups, palette=palette, edgecolor="black", 
        linewidth=1.2, errorbar="se", capsize=0.0, err_kws={'color': 'black'}
    )
    
    plt.ylabel("Mean H-Bonds per Frame", fontsize=12)
    plt.xlabel("Peptide donor/acceptor (N to C Terminus)", fontsize=12)
    
    plt.xticks(rotation=45, ha='right') 
    ax=plt.gca()
    for line in ax.get_lines():
        line.set_solid_capstyle('butt')
    ax.legend_.remove()
    ax.margins(0.03)
    ax.tick_params(axis="y",labelsize=20)
    plt.tight_layout() 
    plt.savefig(filename,dpi=300)

hbond_count_dict = {}

ifps = {}

if not (load_precomputed_counts and load_precomputed_counts_SC_aggregated):
    for fingerprint_file_list, label in zip(fingerprint_files, xlabels):
        print(f"Loading IFP for {label}")
        ifps[label] = load_and_merge_ifps(fingerprint_file_list,start_step=1001,end_step=101000)

if not load_precomputed_counts:
    for fingerprint_file_list, pdb, system_key in zip(fingerprint_files, pdb_files, xlabels):
        print(f"Processing {system_key}")
        hbond_counts = count_hydrogen_bonds(ifps[system_key],pdb,peptide_res_selection, extract_in_blocks=True, block_number=3)
        hbond_count_dict[system_key] = hbond_counts

    pickle.dump(hbond_count_dict,open("all_systems_hbond_counts_dict.pickle","wb"))
else:
    hbond_count_dict = pickle.load(open("all_systems_hbond_counts_dict.pickle","rb"))

#Load hydrogen bond counts with sidechain hbond aggregation
hbond_count_dict_SC_aggregated = {}
ifp = None

if not load_precomputed_counts_SC_aggregated:
    for fingerprint_file_list, pdb, system_key in zip(fingerprint_files, pdb_files, xlabels):
        print(f"Processing {system_key}")
        hbond_counts = count_hydrogen_bonds(ifps[system_key],pdb,peptide_res_selection, extract_in_blocks=True, block_number=3, group_aggregation=True)
        hbond_count_dict_SC_aggregated[system_key] = hbond_counts

    pickle.dump(hbond_count_dict_SC_aggregated,open("all_systems_hbond_counts_SC_aggregated_dict.pickle","wb"))
else:
    hbond_count_dict_SC_aggregated = pickle.load(open("all_systems_hbond_counts_SC_aggregated_dict.pickle","rb"))

#Plot mean number of hbonds per frame

sn.set_style("ticks")
sn.set_context("talk")

hbond_number_mean = []
hbond_number_SEM = []
for label in xlabels:
    means = []
    for d in hbond_count_dict[label]:
        means.append(np.sum(list(d.values())))
    hbond_number_mean.append(np.mean(means))
    hbond_number_SEM.append(np.std(means,ddof=1)/np.sqrt(len(means)))    

ax = plt.gca()
sn.barplot(
    x=xlabels, 
    y=hbond_number_mean, 
    palette=["#aad28e", "#8eaadc"],
    linewidth=1.5,      
    edgecolor="black",
    width=0.6,
    ax=ax
)

ax.errorbar(
    x=np.arange(len(xlabels)), 
    y=hbond_number_mean, 
    yerr=hbond_number_SEM, 
    fmt='none',      
    c='black',       
    capsize=0,
    linewidth=4
)

for spine in ax.spines.values():
    spine.set_linewidth(2.5)

ax.tick_params(width=2.5)
ax.tick_params("x", rotation=90)
ax.set_ylim([4,8])
ax.set_yticks([4,5,6,7,8])
ax.tick_params(axis="y",labelsize=20)
ax.set_ylabel("Mean hydrogen bond number")
fig = plt.gcf()
fig.set_size_inches((2.9,6.5))
plt.tight_layout()
ax.margins(0.2)

plt.savefig("SBD_LPPVK_mean_hbond_number.png", dpi=300, transparent=True)

#Plot mean number of hbonds per LPPVK peptide chemical group (backbone amide, carbonyl and side chain aggregated)
plot_hbond_counts_by_chemical_group(hbond_count_dict_SC_aggregated,"LPPVK","SBD_LPPVK_hbond_per_group_frequency_barplot.png")
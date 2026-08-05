import mdtraj as md
import numpy as np
import polars as pl

def calculate_interface_surface_area(xtc_files, pdb_file, sel1_string, sel2_string, output_filename, chunk_size=1000):
    """
    Calculate interface surface area between 2 selections in a set of MD trajectories

    xtc_files - list of .xtc trajectory files to process
    pdb_file - .pdb file containing structure associated with the trajectories
    sel1_string - selection string (mdtraj selection algebra) of the first group involved in the interaction (SBD)
    sel2_string - selection string (mdtraj selection algebra) of the second group involved in the interaction (peptide)
    output_filename - name of the .parquet file for saving the final DataFrame listing per frame interface surface areas
    chunk_size - length of the trajectory segment used during processing (memory usage optimization)
    
    Returns: None
    """
    top = md.load(pdb_file).topology
    idx1 = top.select(sel1_string)
    idx2 = top.select(sel2_string)
    complex_idx = np.union1d(idx1, idx2)
    
    traj_file_list = []
    traj_frame_list = []
    sasa_sel1_list = []
    sasa_sel2_list = []
    sasa_complex_list = []
    interfce_area_list = []

    for xtc_file in xtc_files:
        current_frame = 0 
        
        for chunk in md.iterload(xtc_file, top=pdb_file, chunk=chunk_size):
            n_frames = chunk.n_frames
            
            traj_complex = chunk.atom_slice(complex_idx)
            traj_sel1 = chunk.atom_slice(idx1)
            traj_sel2 = chunk.atom_slice(idx2)
            
            total_sasa_comp = md.shrake_rupley(traj_complex, mode='atom').sum(axis=1)
            total_sasa_s1 = md.shrake_rupley(traj_sel1, mode='atom').sum(axis=1)
            total_sasa_s2 = md.shrake_rupley(traj_sel2, mode='atom').sum(axis=1)

            interface_area = ((total_sasa_s1 + total_sasa_s2) - total_sasa_comp)/2

            sasa_sel1_list.append(total_sasa_s1)
            sasa_sel2_list.append(total_sasa_s2)
            sasa_complex_list.append(total_sasa_comp)
            interfce_area_list.append(interface_area)

            traj_file_list.extend([xtc_file] * n_frames)
            traj_frame_list.extend(range(current_frame, current_frame + n_frames))
            
            current_frame += n_frames
            
    df = pl.DataFrame({
        "traj_file": traj_file_list,
        "frame": traj_frame_list,
        "SASA_sel1_nm2": np.concatenate(sasa_sel1_list),
        "SASA_sel2_nm2": np.concatenate(sasa_sel2_list),
        "SASA_complex_nm2": np.concatenate(sasa_complex_list),
        "interface_area_nm2": np.concatenate(interfce_area_list)
    })

    #convert interface area to A^2
    df = df.with_columns(
        (pl.col("interface_area_nm2") * 100).alias("interface_area_A2")
    )
    
    df.write_parquet(output_filename)

replicas = [1, 2, 3]
trajectory_list = [f"../AncCQ_SBD_LPPVK/rep{r}/AncCQ_LPPVK_rep{r}_mol_center.xtc" for r in replicas]
top_file = "../peptide_Nter_beta4_distance_calculation/AncCQ_LPPVK_ions_minim3.part0001.pdb"
output_file = "AncCQ_SBD_LPPVK_interface_area.parquet"

calculate_interface_surface_area(trajectory_list, top_file, "resSeq 419 to 638 and not (water or resname POT CLA) ", "resSeq 130 to 139 and not (water or resname POT CLA)", output_file)

trajectory_list = [f"../AncCQ_plus14_SBD_LPPVK/rep{r}/AncCQ_plus14_LPPVK_rep{r}_mol_center.xtc" for r in replicas]
top_file = "../peptide_Nter_beta4_distance_calculation/AncCQ_plus14_LPPVK_ions_minim3.part0001.pdb"
output_file = "AncCQ_plus14_SBD_LPPVK_interface_area.parquet"

calculate_interface_surface_area(trajectory_list, top_file, "resSeq 419 to 638 and not (water or resname POT CLA) ", "resSeq 130 to 139 and not (water or resname POT CLA)", output_file)
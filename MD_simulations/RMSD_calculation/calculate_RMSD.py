import MDAnalysis as mda
from MDAnalysis.analysis import rms
import dask.bag as db
import numpy as np
import pandas as pd
from pathlib import Path
import re
from dask.diagnostics import ProgressBar

simulations = ["AncCQ_SBD_LPPVK", "AncCQ_plus14_SBD_LPPVK"]
def compute_rmsd_segment(start, stop, topo, traj, align_selection, group_selections, ref_pos, ref_selection):
    """
    Calculate RMSD for trajectory fragment

    start - traj fragment range start
    stop - traj fragment stop
    topo - topology filename (.tpr)
    traj - trajectory filename (.xtc)
    align_selection - selection string for structure superposition
    group_selection - selection string for additional groups for RMSD calculation
    ref_pos - positions of reference atoms (taken from frame 0)
    ref_selection - selection string for superposition reference
    """

    u = mda.Universe(topo, traj)
    ref = mda.Universe(topo)
    
    # Use initial atom positions as reference
    ref_atoms = ref.select_atoms(ref_selection)
    ref_atoms.positions = ref_pos
    
    target_atoms = u.select_atoms(align_selection)
    
    R = rms.RMSD(
        target_atoms, 
        reference=ref_atoms, 
        select=align_selection,
        groupselections=group_selections
    )
    
    R.run(start=start, stop=stop)
    
    return R.results.rmsd


if __name__ == "__main__":
    for simulation in simulations:
        for rep in ["rep1", "rep2", "rep3"]:
            print(f"Processing {simulation} {rep}")
            directory = Path(simulation) / rep

            if not directory.exists():
                print(f"Skipping {directory}: Not found")
                continue
            xtc_pattern = re.compile(r".*_mol_center\.xtc$")
            tpr_pattern = re.compile(r".*\.tpr$")
            
            xtc_matches = sorted([str(f) for f in directory.iterdir() if f.is_file() and xtc_pattern.match(f.name)])
            tpr_matches = sorted([str(f) for f in directory.iterdir() if f.is_file() and tpr_pattern.match(f.name)])

            if not xtc_matches:
                print(f"No trajectory parts found in {directory}")
                continue
            elif len(xtc_matches) != 1:
                print(f"More than one matching trajectory found in {directory}")
                continue
            if not tpr_matches:
                print(f"No tpr found in {directory}")
                continue
            elif len(tpr_matches) != 1:
                print(f"More than one matching tpr found in {directory}")
                continue
            
            extra_selections = ["backbone and chainid A","backbone", "backbone and chainid C"] # SBD, whole complex, peptide
            u = mda.Universe(tpr_matches[0], xtc_matches[0])
            u.trajectory[0]
            ref_pos = u.select_atoms("backbone").positions.copy()
            n_frames = len(u.trajectory)
            chunk_size = 1000 
            frame_ranges = [(i, min(i + chunk_size, n_frames)) for i in range(0, n_frames, chunk_size)]

            bag = db.from_sequence(frame_ranges) 
            with ProgressBar():
                results = bag.starmap(
                    compute_rmsd_segment, 
                    topo=tpr_matches[0], 
                    traj=xtc_matches[0], 
                    align_selection='backbone and chainid A and resid 4-109',
                    group_selections=extra_selections, 
                    ref_pos=ref_pos,
                    ref_selection="backbone"
                ).compute()

            final_rmsd = np.vstack(results)
            out_path = Path(f"./{simulation}_{rep}_rmsd_results")
            np.save(f"{out_path}.npy", final_rmsd)
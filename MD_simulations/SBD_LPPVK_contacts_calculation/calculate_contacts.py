import MDAnalysis as mda
import prolif as plf
import pickle
from pathlib import Path

def save_ifp(fp, filename):
    """
    Serialize and save ProLIF interaction fingerprint dictionary to a file.

    fp - ProLIF Fingerprint
    filename - name of the output file

    Returns: None
    """
    print(f"Saving interaction fingerprint to {filename}...")
    with open(filename, "wb") as f:
        pickle.dump(fp.ifp, f)
    print("Save complete!")

def generate_plf_fingerprint(tpr_file, xtc_file, output_filename_pickle, stride=10, peptide_selection_string="chainID C"):
    """
    Calculate ProLIF interaction fingerprint

    tpr_file - topology filename (.tpr)
    xtc_file - trajectory filename (.xtc)
    output_filename_pickle - name for the output file storing ProLIF interaction fingerprint dictionary
    stride - calculate interaction fingerprint for every n-th simulation frame
    peptide_selection_string - selection string for the peptide (MDAnalysis selection algebra)

    Returns: None
    """
    print(f"Loading Universe: {tpr_file} & {xtc_file}")
    u = mda.Universe(tpr_file, xtc_file)
    
    #Define AtomGroups
    peptide_ag = u.select_atoms(peptide_selection_string)
    sbd_selection = f"byres (protein and around 12.0 ({peptide_selection_string}))"
    sbd_ag = u.select_atoms(sbd_selection, updating=True)
    water_selection_string = f"byres ((resname SOL or resname WAT or resname HOH) and around 8.0 (({sbd_selection}) or {peptide_selection_string}))"
    water_ag = u.select_atoms(water_selection_string, updating=True)
    
    #Initialize and run fingerprint
    my_interactions = [
        "Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
        "Anionic", "Cationic", "CationPi", "PiCation", "VdWContact",
        "WaterBridge"
    ]

    fp_filepath = Path(output_filename_pickle)
    fp = plf.Fingerprint(interactions=my_interactions, 
                         parameters={
                             "WaterBridge": {"water": water_ag},
                             "VdWContact":{"tolerance":0.3}
                             }
                         )
    
    if not fp_filepath.is_file():
        print("Running fingerprint calculation...")
        fp.run(
            u.trajectory[::stride],
            lig=peptide_ag,
            prot=sbd_ag,
            n_jobs=40,
            parallel_strategy="chunk"
        )
        save_ifp(fp, output_filename_pickle)
    else:
        print(f"File: {output_filename_pickle} already exists!")

if __name__ == "__main__":
    for r in ["rep1","rep2","rep3"]:
        #Calculate contacts every 10-th frame (every 1 ns)
        generate_plf_fingerprint(
            f"AncCQ_SBD_LPPVK/AncCQ_LPPVK_{r}.tpr",
            f"AncCQ_SBD_LPPVK/{r}/AncCQ_LPPVK_{r}_mol_center.xtc",
            output_filename_pickle=f"AncCQ_SBD_LPPVK/AncCQ_SBD_LPPVK_{r}_interaction_fingerprint.pickle",
            stride=10
            )

        generate_plf_fingerprint(
            f"AncCQ_plus14_SBD_LPPVK/AncCQ_plus14_LPPVK_{r}.tpr",
            f"AncCQ_plus14_SBD_LPPVK/{r}/AncCQ_plus14_LPPVK_{r}_mol_center.xtc",
            output_filename_pickle=f"AncCQ_plus14_SBD_LPPVK/AncCQ_plus14_SBD_LPPVK_{r}_interaction_fingerprint.pickle",
            stride=10
            )
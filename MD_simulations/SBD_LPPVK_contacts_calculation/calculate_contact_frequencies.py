import MDAnalysis as mda
import prolif as plf
import pandas as pd
import pickle
import re
from collections import defaultdict
import numpy as np

NUMBERING_OFFSET_SBD = 428
NUMBERING_OFFSET_LPPVK = -91
FRAME_RANGE = (1001,101000) #0.1-10.1 μs

replicate_list = range(1,4)
fingerprint_files = [
    [f"AncCQ_SBD_LPPVK/AncCQ_SBD_LPPVK_rep{r}_interaction_fingerprint.pickle" for r in replicate_list],
    [f"AncCQ_plus14_SBD_LPPVK/AncCQ_plus14_SBD_LPPVK_rep{r}_interaction_fingerprint.pickle" for r in replicate_list],
]

tpr_files = [
    "AncCQ_SBD_LPPVK/AncCQ_LPPVK_rep1.tpr",
    "AncCQ_plus14_SBD_LPPVK/AncCQ_plus14_LPPVK_rep1.tpr",
]

output_filepath = "."

def load_ifp(filename):
    """
    Load saved ProLIF ifp dictionary from file

    filename - name of the file storing saved ifp dictionary

    Returns: ProLIF ifp dictionary
    """
    print(f"Loading interaction fingerprint from {filename}...")
    with open(filename, "rb") as f:
        loaded_ifp = pickle.load(f)
    print("Load complete!")
    return loaded_ifp

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

        print(f"  -> Added {frames_added_from_file} frames. (Global total: {global_frame_idx})")

    print("All files loaded and merged successfully!")
    return merged_ifp


def renumber_ifp_dictionary(fp, offset_protein, offset_ligand):
    """
    Modify residue numbering inside a ProLIF
    interaction fingerprint dictionary

    fp - prolif.Fingerprint
    offset_protein - residue numbering offset to be applied to the SBD residues
    offset_ligand - residue numbering offset to be applied to the ligand (peptide) residues

    Returns: renumbered prolif.Fingerprint
    """
    print(f"Applying an offset of {offset_protein} (SBD), {offset_ligand} (peptide) to the fingerprint dictionary")

    new_ifp = {}

    for frame_idx, pairs_dict in fp.ifp.items():
        new_pairs_dict = {}

        for (lig_res, prot_res), inter_dict in pairs_dict.items():

            # Regex to split: (Residue Name) (Number) (Chain)
            # Example: "PRO133.C" -> group(1)="PRO", group(2)="133", group(3)=".C"
            match_prot = re.match(r"^([a-zA-Z]+)(\d+)(.*)$", str(prot_res))
            match_lig = re.match(r"^([a-zA-Z]+)(\d+)(.*)$", str(lig_res))

            if match_prot:
                res_name = match_prot.group(1)
                res_num = int(match_prot.group(2))
                res_chain = match_prot.group(3)

                if res_name not in ["SOL", "WAT", "HOH"]:
                    new_res_num = res_num + offset_protein
                    new_prot_res_str = f"{res_name}{new_res_num}{res_chain}"

                    try:
                        from prolif.residue import ResidueId
                        new_prot_res = ResidueId.from_string(new_prot_res_str)
                    except ImportError:
                        new_prot_res = new_prot_res_str 
                else:
                    new_prot_res = prot_res 
            else:
                new_prot_res = prot_res # Fallback if it doesn't match the standard format


            if match_lig:
                res_name = match_lig.group(1)
                res_num = int(match_lig.group(2))
                res_chain = match_lig.group(3)

                if res_name not in ["SOL", "WAT", "HOH"]:
                    new_res_num = res_num + offset_ligand
                    new_lig_res_str = f"{res_name}{new_res_num}{res_chain}"

                    try:
                        from prolif.residue import ResidueId
                        new_lig_res = ResidueId.from_string(new_lig_res_str)
                    except ImportError:
                        new_lig_res = new_lig_res_str 
                else:
                    new_lig_res = lig_res
            else:
                new_lig_res = lig_res

            new_pairs_dict[(new_lig_res, new_prot_res)] = inter_dict

        new_ifp[frame_idx] = new_pairs_dict

    fp.ifp = new_ifp
    print("Renumbering complete!")
    return fp

def calculate_contact_freqs(fp, itypes_resolution_dict, ligand_segid_name, tpr_filename, merge_hb_donor_acceptors=True):
    """
    Calculate contact frequencies with residue or chemical group specificity level

    fp - prolif.Fingerprint
    itypes_resolution_dict - dictionary specifying which interactions should be counted and at which resolution level.
        Example: {'HBDonor': 'Chemical group', 'VdWContact': 'Residue'}
        Chemical group mapping refers only to the ligand (peptide) residues, protein is always treated at residue level
    ligand_segid_name - string specifying name of segment corresponding to the ligand
    tpr_filename - tpr filename used to map group atom ids
    merge_hb_donor_acceptors - if True, 'HBDonor' and 'HBAcceptor' are merged into a single 'Hbond' interaction type

    Returns: 
        df - pandas DataFrame listing frequencies of all peptide-SBD interactions
        ligand_contact_freq_df - pandas DataFrame listing overall contact frequency of each peptide residue with the SBD
    """

    counts = defaultdict(int)
    n_frames = len(fp.ifp)
    u = mda.Universe(tpr_filename)
    general_ligand_res_contact_freq_dict = {} #Used to calculate fraction of trajectory, where given residue paricipates in any interaction with the SBD

    if n_frames == 0:
        return pd.DataFrame()

    #Mapping label:'[atom_name-atom_type]'
    atom_type_mapping = {"Backbone_Amide_N":['N-NH1','HN-H'],
                         "N-terminus amine":["N-NH3","H1-HC","H2-HC","H3-HC"],
                         "Backbone":["CA-CT1","HA-HB1"],
                         "Backbone_Carbonyl":["C-C","O-O"],
                         "C-terminus carboxylate":["OT1-OC","OT2-OC","C-CC"]
                         }

    atom_group_mapping = {}
    if u is not None:
        try:
            ligand_atoms = u.select_atoms(f"segid {ligand_segid_name}")
        except Exception as e:
            raise RuntimeError(f"Could not select ligand segid '{ligand_segid_name}' in {tpr_filename}") from e

        for index,atom in enumerate(ligand_atoms):
            name = atom.name.upper()
            type = atom.type.upper()
            atom_identifier = f"{name}-{type}"

            for mapping in atom_type_mapping.items():
                if atom_identifier in mapping[1]:
                    atom_group_mapping[index]=mapping[0]
            
            #Anything not mapping to a specified groups belong to sidechain
            if index not in atom_group_mapping.keys():
                atom_group_mapping[index]="Sidechain"

    for frame in fp.ifp.items():
        observed_in_frame = set()
        lig_res_observed_in_frame = set()
        for (lig_res, prot_res), interactions in frame[1].items():
            lig_res_observed_in_frame.add(lig_res)
            for itype, occurrences in interactions.items():

                # Handle Hydrogen Bond merging
                display_itype = itype
                if merge_hb_donor_acceptors and itype in ['HBDonor', 'HBAcceptor']:
                    display_itype = 'Hbond'

                resolution = itypes_resolution_dict.get(display_itype) or itypes_resolution_dict.get(itype)

                if resolution == 'Residue':
                    contact_key = (lig_res, prot_res, display_itype)

                    if contact_key not in observed_in_frame:
                        observed_in_frame.add(contact_key)
                        counts[contact_key] += 1

                elif resolution == 'Chemical group':
                    for occ in occurrences:
                        lig_indices = occ['parent_indices']['ligand']

                        # Apply mapping. If the atom isn't a protein atom (e.g., ligand), it keeps its raw index.
                        lig_groups = tuple(sorted(set(atom_group_mapping.get(idx, idx) for idx in lig_indices)))

                        contact_key = (lig_res, prot_res, display_itype, lig_groups)

                        if contact_key not in observed_in_frame:
                            observed_in_frame.add(contact_key)
                            counts[contact_key] += 1
            
        for res in lig_res_observed_in_frame:
            if res in general_ligand_res_contact_freq_dict.keys():
                general_ligand_res_contact_freq_dict[res]+=1
            else:
                general_ligand_res_contact_freq_dict[res]=1

    def clean_format(item):
        """
        Clean format of peptide atom group for list containing only one entry

        item - list containing peptide atom groups involved in the interaction

        Returns: just ligand atom group when only one present, original list otherwise
        """
        if item is None:
            return None
        return item[0] if len(item) == 1 else item

    # Format the results into a pandas DataFrame
    results = []
    for key, count in counts.items():
        freq = count / n_frames

        if len(key) == 3:  # Residue level
            results.append({
                'Ligand_Residue': key[0],
                'Protein_Residue': key[1],
                'Interaction_Type': key[2],
                'Peptide_Atom_Groups': None,
                'Frequency': freq
            })
        else:  # Chemical group level
            results.append({
                'Ligand_Residue': key[0],
                'Protein_Residue': key[1],
                'Interaction_Type': key[2],
                'Peptide_Atom_Groups': clean_format(key[3]),
                'Frequency': freq
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=['Frequency', 'Protein_Residue'], ascending=[False, True]).reset_index(drop=True)
    
    ligand_contact_freq_df = pd.DataFrame(
        {
            "ligand_res":general_ligand_res_contact_freq_dict.keys(),
            "contact_freq":np.array(list(general_ligand_res_contact_freq_dict.values()))/n_frames
        }
    )

    return df,ligand_contact_freq_df

itypes_resolution_dict = {
   "HBDonor":"Chemical group",
   "HBAcceptor":"Chemical group",
   "Cationic":"Chemical group",
   "Anionic":"Chemical group",
   "CationPi":"Chemical group",
   "PiCation":"Chemical group",
   "PiStacking":"Chemical group",
   "VdWContact":"Residue" 
}

peptide_seg_name = "seg_1_Protein_chain_C"

for fp_files,tpr_filename in zip(fingerprint_files,tpr_files):
    print(f"Processing {tpr_filename}")
    fp=plf.Fingerprint()
    print("Loading fingerprints")
    fp.ifp = load_and_merge_ifps(fp_files,start_step=FRAME_RANGE[0],end_step=FRAME_RANGE[1])
    print("Renumbering residues")
    fp = renumber_ifp_dictionary(fp,NUMBERING_OFFSET_SBD,NUMBERING_OFFSET_LPPVK) 
    print("Calculating contact frequencies")
    contact_freqs_df, ligand_res_contact_freqs_df = calculate_contact_freqs(fp, itypes_resolution_dict, peptide_seg_name, tpr_filename)
    print("Saving frequencies")
    contact_freqs_df.to_csv(output_filepath+f"/{tpr_filename.split('/')[-1].strip('_rep1.tpr')}_peptide_contact_frequencies.csv")
    ligand_res_contact_freqs_df.to_csv(output_filepath+f"/{tpr_filename.split('/')[-1].strip('_rep1.tpr')}_peptide_contact_frequencies_per_peptide_res.csv")
#!/bin/bash

for i in {1..3}
do
    if [ ! -f pep_beta_4_com_dist_AncCQ_LPPVK_rep${i}.txt ]; then
        plumed driver --plumed peptide_beta4_distance_plumed_AncCQ_LPPVK.dat --timestep 100 --ixtc ../AncCQ_SBD_LPPVK/rep${i}/AncCQ_LPPVK_rep${i}_mol_center.xtc --pdb AncCQ_LPPVK_ions_minim3.part0001.pdb
        mv pep_beta_4_com_dist.txt pep_beta_4_com_dist_AncCQ_LPPVK_rep${i}.txt
    fi 
done

for i in {1..3}
do
    if [ ! -f pep_beta_4_com_dist_AncCQ_plus14_LPPVK_rep${i}.txt ]; then
        plumed driver --plumed peptide_beta4_distance_plumed_AncCQ_plus14_LPPVK.dat --timestep 100 --ixtc ../AncCQ_plus14_SBD_LPPVK/rep${i}/AncCQ_plus14_LPPVK_rep${i}_mol_center.xtc --pdb AncCQ_plus14_LPPVK_ions_minim3.part0001.pdb
        mv pep_beta_4_com_dist.txt pep_beta_4_com_dist_AncCQ_plus14_LPPVK_rep${i}.txt
    fi
done
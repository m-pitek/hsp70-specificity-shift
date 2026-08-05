# Hsp70-specificity-shift
Repository with data and analysis scripts for 'Molecular basis of the evolutionary shift in an Hsp70 client-binding preference'

## Repository structure:

<pre>
.
├── <b>AncCQ_SBD_variants_library</b> ← Data and scripts for processing, analysis and visualization
│   │                            of the AncCQ SBD variants library
│   │
│   ├── <b>Analysis_of_evolutionary_trajectories_accessibility</b> ← Analysis and visualization of all
│   │                                                         evolutionary trajectories leading from
│   │                                                         AncCQ to AncCQ+6
│   │
│   ├── <b>Enrichment_correlation</b> ← Correlation calculation between the enrichment of selected
│   │                            AncCQ SBD variants and the Δ<i>G</i> of peptide binding
│   │   
│   ├── <b>Epistasis</b> ← Analysis of epistatic interactions between a subset of 12 substitutions
│   │
│   ├── <b>Pareto_front</b> ← Determination of the Pareto front for LPPVK peptide binding and p5 peptide exclusion
│   │   
│   ├── <b>Sequencing_data_processing</b> ← Library quality control and enrichment calculation
│   │   │
│   │   ├── <b>Enrich2</b> ← Merging of library selection replicates using the Enrich2 framework
│   │   │   └── <b>Enrich2_output</b>
│   │   │
│   │   └── <b>Library_quality_control_after_Golden_Gate_Assembly</b> ← Processing of Nanopore reads from the assembled library
│   │                                                            before cloning into the T7 vector
│   └── <b>Substitution_effects</b> ← Calculation of the effect of each individual substitution on enrichment,
│                              in the background of all possible combinations of the remaining substitutions
│
│
├── <b>MD_simulations</b> ← Data and scripts for processing, analysis and visualization of MD simulations
│   │                of SBD-LPPVK complexes
│   │
│   ├── <b>interface_surface_area_calculation</b> ← Determination of the interface area between LPPVK and SBD in MD trajectories
│   │
│   ├── <b>LPPVK_Nter-SBD_strand_4_COM_distance_free_energy_profiles</b> ← Determination of free energy profiles
│   │                                                               for the distance between centers of mass of the LPPVK N-terminal
│   │                                                               region and the SBD strand β4 segment
│   │
│   ├── <b>RMSD_calculation</b>
│   │
│   ├── <b>SBD_LPPVK_contacts_calculation</b> ← Determination and analysis of contacts between LPPVK and SBD
│   │
│   └── <b>trajectory_processing</b> ← Pre-analysis MD trajectories processing
│
│
├── <b>Peptide_interaction_assays</b> ← Data and scripts for analysis and visualization of peptide binding
│   │                            using fluorescence polarization–based assays
│   │
│   ├── <b>Equilibrium_binding</b> ← <i>K</i><sub>D</sub> determination
│   │
│   └── <b>LPPVK_dissociation_kinetics</b> ← <i>k</i><sub>off</sub> determination for SBD-LPPVK complexes
│
│
└── <b>SBD_stability</b> ← Circular Dichroism (CD) spectra and melting curves for selected SBDs
</pre>

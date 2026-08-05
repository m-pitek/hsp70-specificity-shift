import pysam
import pandas as pd
import os

f462s_codon=83
f462s_codons=["TTT","AGC"]
silent_substitution_position=113
substituted_codon_positions_R1 = [17,32,35,59,65,74,95,110,143]
substituted_codon_positions_R2 = [19,31,82,133]
substituted_codon_seqs_R1 = [["CTG","TAC"],["ACC","AGC"],["CGT","CCG"],["ATT","GTG"],["ACC","GTG"],["AGC","ACC"],["GCG","GTG"],["AGC","GGC"],["GAA","GGC"]]
substituted_codon_seqs_R2 = [["GAC","GAA"],["GCG","ACC"],["ACC","AGC"],["ACC","AAA"]]
silent_substitution_seqs = ["GTG","GTT"]

technical_repeats_ids = ["GTC","AGTA","CAGAT"]

def reverse_complement(sequence):
    """
    Generate reverse complement DNA sequence

    sequence - str containing DNA sequence

    Returns: Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(sequence))

def get_read_id(read):
    """
    Extract technical replicate id from the R2 read

    read - string containing R2 read sequence

    Returns: Numerical id of the technical replicate of the read
    """
    if read.startswith(technical_repeats_ids[0]):
        return 0
    elif read.startswith(technical_repeats_ids[1]):
        return 1
    elif read.startswith(technical_repeats_ids[2]):
        return 2
    else:
        return -1

def process_read_pair(R1,R2):
    """
    Extract relevant information about codons at variable positions 
    and the technical replicate id from the read pair

    R1 - string containing a sequence of the read R1
    R2 - string containing a sequence of the read R2

    Returns: 
    codon_string (binary string encoding presence of the 14 substitutions)
    replicate_id - numerical id of the technical replicate of the read
    read_pair_correct - a flag indicating whether all variable positions contain expected sequences
    """
    codon_string=""
    read_pair_correct = True
    replicate_id = get_read_id(R2)
    if replicate_id != -1:
        #Read 1 codons
        for cpos,variants in zip(substituted_codon_positions_R1,substituted_codon_seqs_R1):
            codon = R1[cpos:cpos+3]
            if codon in variants:
                codon_string += str(variants.index(codon))
            else:
                read_pair_correct = False
                codon_string+="6"

        #Read 2 codons
        for cpos,variants in zip(substituted_codon_positions_R2,substituted_codon_seqs_R2):
            codon = reverse_complement(R2[cpos+replicate_id:cpos+replicate_id+3])
            if codon in variants:
                codon_string += str(variants.index(codon))
            else:
                read_pair_correct = False
                codon_string+="6"

        #Silent substitution (codon #14)
        codon = R1[silent_substitution_position:silent_substitution_position+3]
        if codon in silent_substitution_seqs:
            codon_string += str(silent_substitution_seqs.index(codon))
        else:
            read_pair_correct = False
            codon_string+="6"

        #F462S variant (negative binding control)
        codon = R1[f462s_codon:f462s_codon+3]
        if codon in f462s_codons:
            if f462s_codons.index(codon) == 1:
                codon_string="5"

    else:
        read_pair_correct = False
    return codon_string, replicate_id, read_pair_correct

def process_sample(R1_filename,R2_filename,sample_name):
    """
    Process all the reads originating from the single selection replicate or unselected library

    R1_filename - file containing given sample R1 reads after fastp filtering
    R2_filename - file containing given sample R2 reads after fastp filtering
    sample_name - name of the processed sample to be included in processed reads metadata

    Returns: None (appends processed reads to the file stored dataframe)
    """
    print(f"{sample_name}")
    it=0
    read_dicts = []

    with pysam.FastxFile(R1_filename) as fR1:
        with pysam.FastxFile(R2_filename) as fR2:
            for R1,R2 in zip(fR1,fR2):
                codon_string, replicate_id, correct = process_read_pair(R1.sequence, R2.sequence)
                if correct:
                    read_dicts.append({"sample":sample_name,"amplification_replicate":replicate_id,"R1_id":R1.name,"R2_id":R2.name,"codon_string":codon_string})
                it+=1
                if it%100000 == 0:
                    print(it/1e6)

        df = pd.DataFrame(read_dicts)
    if not os.path.exists("illumina_reads_extracted_codons.parquet"):
        df.to_parquet("illumina_reads_extracted_codons.parquet")
    else:
        old_df = pd.read_parquet("illumina_reads_extracted_codons.parquet")
        new_df = pd.concat([old_df,df],ignore_index=True)
        new_df.to_parquet("illumina_reads_extracted_codons.parquet")

#Sample IDs: 4 - unselected library, 5-7 - library after LPPVK selection, 8-10 - library after p5 selection
for i in range(4,11):
    process_sample(f"./fastp_filtered/IL-{i}_Q25_filtered_R1.fq.gz",f"./fastp_filtered/IL-{i}_Q25_filtered_R2.fq.gz",f"IL{i}")
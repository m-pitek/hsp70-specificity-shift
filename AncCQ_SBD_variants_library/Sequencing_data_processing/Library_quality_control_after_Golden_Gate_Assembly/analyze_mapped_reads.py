import pysam
import pandas as pd

mapped_reads = pysam.AlignmentFile("AncCQ_library_v4.mmap2.ambigous_ref_library_v3.bam","r")

silent_substitution_position=228
substituted_codon_positions = [132,147,150,174,180,189,210,225,258,294,345,396,408,432,silent_substitution_position]
substituted_codon_seqs = [["CTG","TAC"],
                          ["ACC","AGC"],
                          ["CGT","CCG"],
                          ["ATT","GTG"],
                          ["ACC","GTG"],
                          ["AGC","ACC"],
                          ["GCG","GTG"],
                          ["AGC","GGC"],
                          ["GAA","GGC"],
                          ["ACC","AAA"],
                          ["ACC","AGC"],
                          ["GCG","ACC"],
                          ["GAC","GAA"],
                          ["AGC","AAC"],
                          ["GTG","GTT"]]


def codon_string_to_segment_string(codon_string):
    """
    Map binary codon string coding to quaternary segment based coding

    codon_string - string containing binary coding of all 14 substitutions absence/presence

    Returns: string containing information about all 14 substitutions absence/presence in
    quaternary segment based coding
    """
    segment_string=""
    segments=[1,2,2,2,1,2,2,2]
    it=0
    s_dict={"00":0,"10":1,"01":2,"11":3,"0":0,"1":1}
    for s in segments:
        if codon_string[it:it+s] in list(s_dict.keys()):
            segment_string+=str(s_dict[codon_string[it:it+s]])
        else:
            segment_string+="4"
        it+=s
    return segment_string

#Segment substitutions extraction
codon_strings = []
record_codon_dict_list = []
for r in mapped_reads.fetch("ref",132,434):
    codon_string =""
    rpos = r.get_reference_positions(full_length=True)
    r_id = r.query_name
    for i in range(15):
        if all([p in rpos for p in range(substituted_codon_positions[i],substituted_codon_positions[i]+3)]):
            codon_pos = rpos.index(substituted_codon_positions[i])
            codon_seq = r.seq[codon_pos:codon_pos+3]
            if codon_seq in substituted_codon_seqs[i]:
                codon_string+=str(substituted_codon_seqs[i].index(codon_seq))
            else:
                codon_string+="3"
        else:
            codon_string+="2"
    strand = ""
    if r.is_reverse:
        strand="reverse"
    else:
        strand="forward"
    record_codon_dict_list.append({"seq_id":r_id,"strand":strand,"codon_string":codon_string[:-1],"segment_string":codon_string_to_segment_string(codon_string[:-1]),"ref_start":r.reference_start,"ref_end":r.reference_end,"silent_substitution":codon_string[-1]})
    codon_strings.append(codon_string)

codons_df = pd.DataFrame(record_codon_dict_list)
codons_df["read_correct"] = False

for i in range(len(codons_df)):
    row = codons_df.iloc[i]
    codon_string = row["codon_string"]
    if (codon_string.count('0')+codon_string.count('1') == len(codon_string)) and (row['silent_substitution'] in ['0','1']):
        codons_df.iloc[i,7] = True


codons_df.to_parquet("reads_codon_assignments.parquet")
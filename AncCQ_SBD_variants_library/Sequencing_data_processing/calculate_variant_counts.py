import pandas as pd

df = pd.read_parquet("illumina_reads_extracted_codons.parquet")

df_counts = df.groupby(["sample", "amplification_replicate", "codon_string"]).count().reset_index()
df_counts = df_counts.drop(["R2_id"], axis=1)
df_counts = df_counts.rename(columns={"R1_id": "read_count"})

df_counts.to_parquet("SBD_variants_counts.parquet")

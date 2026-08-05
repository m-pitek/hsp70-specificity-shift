import polars as pl

df = pl.read_parquet("../SBD_variants_counts.parquet")
pooled_df = df.group_by(["sample", "codon_string"]).agg(
    pl.sum("read_count").alias("read_count")
).sort("codon_string").sort("sample")
pooled_df.with_columns(("v"+pl.col("codon_string")).alias("codon_string"))

for sample in range(4,11):
    pooled_df.filter(
        pl.col("sample")==f"IL{sample}"
    )["codon_string","read_count"].rename(
        {"codon_string": ' ', "read_count": "count"}
    ).write_csv(f"IL{sample}.tsv",separator="\t")


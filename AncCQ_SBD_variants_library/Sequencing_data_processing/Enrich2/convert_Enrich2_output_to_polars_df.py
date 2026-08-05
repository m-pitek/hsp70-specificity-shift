import polars as pl
import pandas as pd
import numpy as np

df_scores = pd.read_hdf("Enrich2_output/AncCQ_library_exp.h5","/main/identifiers/scores")
df_scores_stacked = df_scores.stack(level=0,future_stack=True).reset_index()
df_scores_long = df_scores_stacked.rename(columns={
         ' ': 'codon_string',
         'condition' : 'selection'
           })

df_scores_long['selection'] = df_scores_long['selection'].replace({'LPPVK_selection':'LPPVK','p5_selection':'p5'})
df_scores_long['linear_score'] = np.exp(df_scores_long['score'])
df_scores_long['linear_SE'] = df_scores_long["linear_score"]*df_scores_long['SE']
df_scores_long["RSE"] = df_scores_long["linear_SE"]/df_scores_long["linear_score"]
df_scores_long.loc[df_scores_long["codon_string"]=="_wt","codon_string"] = 14*"0"

df_polars = pl.DataFrame(df_scores_long.sort_values("codon_string"))
df_polars = df_polars.with_columns(pl.col("codon_string").str.count_matches("1").alias("substitution_number"))

df_polars.write_parquet("SBD_enrichments_aggregated_normalized_Enrich2.parquet")

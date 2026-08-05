import pandas as pd
import numpy as np


def read_replicate_h5(filename, selection, replicate):
    """
    Reads Enrich2 output file containing enrichments for a single selection replicate

    filename - name of the Enrich2 output file
    selection - peptide name used for the selection
    replicate - replicate id

    Returns: Pandas dataframe containing enrichment scores (normalized relative to the AncCQ enrichment) for a given selection replicate
    """
    scores =  pd.read_hdf(filename,"/main/identifiers/scores").drop(['logratio','variance'],axis=1)
    scores['linear_score'] = np.exp(scores['score'])
    scores['linear_SE'] = scores["linear_score"]*scores['SE']
    scores["RSE"] = scores["linear_SE"]/scores["linear_score"] #RSE - relative standard error

    scores["codon_string"] = scores.index
    scores['selection'] = selection
    scores['selection_replicate'] = replicate

    scores = scores.reset_index()
    scores.loc[scores["codon_string"]=="_wt","codon_string"] = 14*"0"

    return scores


selections = ["LPPVK","p5"]

scores = []
for selection in selections:
    for rep in range(1,4):
        scores.append(read_replicate_h5(f"Enrich2_output/{selection}_rep{rep}_sel.h5",selection,rep-1))

df = pd.concat(scores,ignore_index=True)
df.to_parquet("SBD_per_selection_replicate_enrichments_Enrich2.parquet")

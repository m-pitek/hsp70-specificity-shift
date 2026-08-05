import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import gc
import pandas as pd
import polars as pl

# choose statistical or biochemical epistasis
ep_type = 'stat'
selections = ["LPPVK","p5"]

for selection in selections:
    print(f"Running epistasis order determination for {selection} selection.")
    df = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")
    df = df.filter((pl.col("codon_string")!="5")&
                (pl.col("codon_string").str.contains(r".........0...0")) & 
                (pl.col("selection")==selection)).sample(fraction=1.0,shuffle=True,seed=4242)


    #Modify codon_string by dropping values corresponding to the T494K and S540N
    indices_to_keep = [i for i, char in enumerate(".........0...0") if char != '0']
    df_with_float_arrays = df.with_columns(
        pl.col("codon_string")
        .str.split("")
        .list.gather(indices_to_keep) 
        .cast(pl.List(pl.Float64))
        .alias("float_array")
    )


    phenos = df.select("linear_score").to_numpy().reshape(-1)
    genos = np.vstack(df_with_float_arrays.select("float_array").to_numpy().flatten())

    if ep_type == 'stat':
        genos = 2*(genos-0.5)

    num_folds = 10
    max_epistasis_order = 5

    # proportion of data to be used as a test set
    prop_test = 0.1

    size_test = int(prop_test*len(genos))
    size_train = len(genos)-size_test

    # lists to store r squared values
    rsq_train_list = np.zeros((num_folds, max_epistasis_order+1))
    rsq_test_list = np.zeros((num_folds, max_epistasis_order+1))

    # loop over CV folds
    for f in range(num_folds):
        test_range_end = (f+1)*size_test
        if test_range_end > len(genos):
            test_range_end = len(genos)
        test_indices = range(f*size_test,test_range_end)

        genos_train = np.delete(genos.copy(), test_indices, 0)
        genos_test = genos[test_indices].copy()
        phenos_train = np.delete(phenos, test_indices, 0)
        phenos_test = phenos[test_indices].copy()

        # fit models of increasing order
        for order in range(0,max_epistasis_order+1):
            reg_current = linear_model.Ridge(alpha=0.01, solver='lsqr', fit_intercept=False)
            poly_current = PolynomialFeatures(order,interaction_only=True)
            genos_train_current = poly_current.fit_transform(genos_train)
            genos_test_current = poly_current.fit_transform(genos_test)
            reg_current.fit(genos_train_current, phenos_train)
            reg_coefs_current  = reg_current.coef_

            rsquared_train_current = 1-np.sum((phenos_train-reg_current.predict(genos_train_current))**2)/\
            np.sum((phenos_train-np.mean(phenos_train))**2)

            rsquared_test_current = 1-np.sum((phenos_test-reg_current.predict(genos_test_current))**2)/\
            np.sum((phenos_test-np.mean(phenos_test))**2)

            rsq_train_list[f, order] = rsquared_train_current
            rsq_test_list[f, order] = rsquared_test_current

        del reg_current
        del test_indices
        del genos_train
        del genos_test
        del phenos_train
        del phenos_test
        del reg_coefs_current
        del poly_current
        gc.collect()

    lst = []
    for f in range(num_folds):
        for o in range(0,max_epistasis_order+1):
            lst += [(f, o, rsq_train_list[f, o], rsq_test_list[f,o])]
    df_scores = pd.DataFrame(lst, columns=["fold_nb", "order", "train", "test"])
    df_scores.to_csv(f"model_order_optimization_r2_CV_{ep_type}_{selection}.csv", index=False)

    print("Means:")
    print(df_scores.groupby("order").agg({"train":"mean", "test": "mean"}))
    print("Standard deviations:")
    print(df_scores.groupby("order").agg({"train":"std", "test": "std"}))

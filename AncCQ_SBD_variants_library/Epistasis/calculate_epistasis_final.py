import numpy as np
import csv
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import polars as pl
import seaborn as sn
from matplotlib import pyplot as plt
from scipy.special import comb

# choose statistical or biochemical epistasis
#ep_type = 'biochem'
ep_type = 'stat'

# read in rsq data from CV folds
num_folds = 10
selections=["LPPVK","p5"]

color_dict = {"LPPVK":"#ff9a02",
              "p5":"#28c700"}
              
piechart_colors = {
    "LPPVK":["#ff954e","#ffb27e","#ffceae"],
    "p5":["#69ae55","#8efa71","#bdfaac"]
}


for selection in selections:
    rsq_stats = pl.read_csv(f'model_order_optimization_r2_CV_{ep_type}_{selection}.csv')

    # average over folds to select optimal order
    print(rsq_stats)
    mean_rsq_train = rsq_stats.group_by("order").mean().sort("order").select("train").to_numpy().reshape(-1)
    stdev_rsq_train = rsq_stats.group_by("order").agg(pl.col("train").std()).sort("order").select("train").to_numpy().reshape(-1)
    mean_rsq_test = rsq_stats.group_by("order").mean().sort("order").select("test").to_numpy().reshape(-1)
    stdev_rsq_test = rsq_stats.group_by("order").agg(pl.col("test").std()).sort("order").select("test").to_numpy().reshape(-1)
    print(stdev_rsq_test)

    #Plot model performance
    plt.clf()
    sn.set_context("talk")
    sn.set_style("ticks")
    ax = sn.lineplot(data = rsq_stats, x="order",y="train",errorbar="se",err_style="bars",marker="o",linestyle="--",err_kws={"ecolor":"k","capsize":0},markersize=8,color="#6B6B6B")
    sn.lineplot(data = rsq_stats, x="order",y="test",errorbar="se",err_style="bars",marker="o",linestyle="--",err_kws={"ecolor":"k","capsize":0,"solid_capstyle":"butt"},markersize=8,color="#FF4444")
    ax.set_ylim([-0.1,1])
    ax.set_xticks(range(0,6))
    ax.set_ylabel("$R^2$")
    ax.set_xlabel("Order of epistatic model")
    plt.gcf().set_size_inches((5,5))
    plt.tight_layout()
    plt.savefig(f"model_performance_{ep_type}_{selection}.png",dpi=300)
    ax.set_ylim([0.75,0.98])
    ax.set_xlim([0.8,4.5])

    plt.gcf().set_size_inches((3,4))
    plt.tight_layout()
    plt.savefig(f"model_performance_{ep_type}_{selection}_inset.png",dpi=300,transparent=True)

    optimal_order = rsq_stats.group_by("order").mean().sort("test").select("order").item(-1,0)

    print(mean_rsq_test)
    print(f'Optimal order, {selection}: {optimal_order}')

    # write averages to new file
    with open(f'model_order_optimization_r2_CV_{ep_type}_{selection}_averages.csv','w') as writefile:
        rsq_writer = csv.writer(writefile)
        rsq_writer.writerow(['Optimal order: ' + str(optimal_order)])
        rsq_writer.writerow(['Type','Order','Mean','Std'])
        for i in range(len(mean_rsq_train)):
            rsq_writer.writerow(['Train',str(i),mean_rsq_train[i],stdev_rsq_train[i]])
        for i in range(len(mean_rsq_test)):
            rsq_writer.writerow(['Test',str(i),mean_rsq_test[i],stdev_rsq_test[i]])
        writefile.close()

    # read in data

    geno_vectors = []
    phenos = []

    mutations = [str(x) for x in range(1,13)]

    df = pl.read_parquet("../Sequencing_data_processing/Enrich2/SBD_enrichments_aggregated_normalized_Enrich2.parquet")
    df = df.filter((pl.col("codon_string")!="5")&
                (pl.col("codon_string").str.contains(r".........0...0")) & 
                (pl.col("selection")==selection)).sample(fraction=1.0,shuffle=True,seed=4242)


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


    # Fit final models

    np.random.seed(4242)

    final_models_rsq = []
    # fit models of increasing order
    optimal_model_coefs = None
    for order in range(1,optimal_order+1):
        poly_current = PolynomialFeatures(order,interaction_only=True)
        genos_current = poly_current.fit_transform(genos)

        # fit
        reg_current = sm.OLS(phenos,genos_current).fit()
        reg_coefs_current = reg_current.params
        reg_CIs_current = reg_current.conf_int(alpha=0.05/float(len(reg_coefs_current)), cols=None)
        reg_stderr = reg_current.bse
        reg_pvalues = reg_current.pvalues
        if order == optimal_order:
            optimal_model_coefs = reg_coefs_current

        predicted_phenos = reg_current.predict(genos_current)
        rsquared_current = reg_current.rsquared
        final_models_rsq.append(rsquared_current)

        # write model to file
        coef_names = poly_current.get_feature_names_out(input_features = mutations)
        with open(f'AncCQ_{selection}_'+str(order)+'order_'+ep_type+'_epistasis_model_coefficients.txt','w') as writefile:
            coef_writer = csv.writer(writefile,delimiter='\t')
            coef_writer.writerow(['Params: ',len(reg_coefs_current)])
            coef_writer.writerow(['Performance: ',rsquared_current])
            coef_writer.writerow(['Term','Coefficient','Standard Error','p-value','95% CI lower','95% CI upper'])
            coef_writer.writerow(['Intercept',reg_coefs_current[0]])
            for i in range(1,len(reg_coefs_current)):
                coef_writer.writerow([','.join(coef_names[i].split(' ')),reg_coefs_current[i],reg_stderr[i],
                                    reg_pvalues[i],reg_CIs_current[i][0],reg_CIs_current[i][1]])
            writefile.close()


    relative_variance_per_order = []
    last_order_rsq = 0
    for rsq in final_models_rsq:
        relative_variance_per_order.append((rsq/final_models_rsq[-1])-last_order_rsq)
        last_order_rsq += relative_variance_per_order[-1]


    with open(f'AncCQ_{selection}_'+ep_type+'_relative_variance_explained.txt','w') as writefile:
        writefile.write(f"Order\tRelative variance\n")
        for order, rel_var in enumerate(relative_variance_per_order, start=1):
            writefile.write(f"{order}\t{rel_var}\n")
   
 #Plot variance explained by models of increasing order

    plt.clf()

    plt.figure(figsize=(8, 4))
    plt.pie(relative_variance_per_order,labels=["" for i in range(0,optimal_order)],colors=piechart_colors[selection], explode=(0,0,0.0), wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'antialiased': True})
    plt.savefig(f"relative_variance_explained_piechart_{ep_type}_{selection}.png")

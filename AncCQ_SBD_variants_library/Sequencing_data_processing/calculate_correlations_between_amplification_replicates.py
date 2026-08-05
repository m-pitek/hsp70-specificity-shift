import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

def calculate_replicate_correlations(df, sample_name, log_file):
    """
    Calculates Pearson correlation coefficients between PCR amplification replicates of a given sample

    df - dataframe containing read counts of each AncCQ SBD variant normalized to the total number of reads (counts per million)
    sample_name - name of the sample for correlation calculation
    log_file - name of the CSV file the correlations should be written to

    Returns: None
    """
    for i in [(0,1),(1,2),(0,2)]:
        x = df.loc[(df["sample"]==sample_name)&(df["amplification_replicate"]==i[0]), "normalized_read_count"].to_numpy()
        y = df.loc[(df["sample"]==sample_name)&(df["amplification_replicate"]==i[1]), "normalized_read_count"].to_numpy()
        pearson_r, p_value = pearsonr(x,y)
        print(f"{sample_name},{i},{pearson_r}")
        with open(log_file,"a") as f:
            f.write(f"{sample_name},{i[0]},{i[1]},{pearson_r}\n")


df = pd.read_parquet("SBD_variants_counts.parquet")

#Calculate correlations between technical replicates

for i in range(4,11):
    for j in range(0,3):
        df.loc[(df["sample"]==f"IL{i}") & (df["amplification_replicate"]==j),"normalized_read_count"] = \
        df.loc[(df["sample"]==f"IL{i}") & (df["amplification_replicate"]==j),"read_count"]/\
        (df.loc[(df["sample"]==f"IL{i}") & (df["amplification_replicate"]==j),"read_count"].sum()/1e6)

with open("technical_replicate_correlations.csv","w") as f:
    f.write("sample_name,first_replicate,second_replicate,Pearson-r\n")

for i in range(4,11):
    #As some technical replicates lack reads for negative control and one of the substitution variants 
    #exclude them from amplification replicate correlation calculation
    calculate_replicate_correlations(
        df.loc[(df["codon_string"]!="5") & (df["codon_string"]!="11100111010100")],
        f"IL{i}",
        "technical_replicate_correlations.csv"
    )
       
#Plot bar chart of pearson correlations between technical replicates

pearson_correlations_df = pd.read_csv("technical_replicate_correlations.csv")
pearson_correlations_df["first_replicate"] = (pearson_correlations_df["first_replicate"]+1).astype(str)  
pearson_correlations_df["second_replicate"] = (pearson_correlations_df["second_replicate"]+1).astype(str)
pearson_correlations_df["replicate_pair"] = pearson_correlations_df["first_replicate"]+"-"+pearson_correlations_df["second_replicate"]
pearson_correlations_df["name"] = pearson_correlations_df["sample_name"]

plt.clf()
sn.set_context("talk")
sn.set_style("ticks")
sn.barplot(pearson_correlations_df,x="name",y="Pearson-r",linewidth=1,edgecolor="k",width=0.7, hue="replicate_pair")
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.ylabel("Pearson $r$")
plt.xticks(rotation=90)
plt.yticks([1,0.95,0.90,0.85])
plt.ylim(0.84,1)
plt.xlabel("Sample")
plt.tight_layout()
plt.savefig("correlation_between_amplification_replicates.png",dpi=300)
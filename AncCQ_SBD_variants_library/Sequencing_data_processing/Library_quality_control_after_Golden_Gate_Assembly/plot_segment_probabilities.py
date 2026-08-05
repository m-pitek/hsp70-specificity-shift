import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import itertools

df = pd.read_parquet("reads_codon_assignments.parquet")

substitutions_dict = {1:[["L440","Y440"]], 2:[["T445","S445"], ["R446","P446"]], 3:[["I454","V454"], ["T456","V456"]], 4:[["S459","T459"], ["A466","V466"]],
                 5:[["S471","G471"],["V472$^{\\mathrm{GTG}}$","V472$^{\\mathrm{GTT}}$"]], 6:[["E482","G482"], ["T494","K494"]], 7:[["T511","S511"], ["A528","T528"]], 8:[["D532","E532"], ["S540","N540"]]}

segment_variant_colors=["#00cc99", "#00bbcc", "#0081cc", "#4645ba"]

def calculate_95_confidence_interval(sample_number,p):
    """
    Calculate 95% Wald confidence intervals

    sample_number - number of observations 
    p - probability

    Returns: 95% Wald confidence interval
    """
    return np.sqrt((p*(1-p))/sample_number)*1.96

def prep_segment_labels(substitutions):
    """
    Prepare x-axis labels for plotting segment variant probabilities based on number of substitutions
    present within given segment

    substitutions - list of substitutions present within segment

    Returns: List of labels describing all possible substitution combinations within given segment
    """
    if len(substitutions)>1:
        return [", ".join(reversed(t)) for t in itertools.product(*reversed(substitutions))]
    else:
        return substitutions[0]

def plot_segment_probabilities(array,segment_number,ax):
    """
    Plot probabilities for all possible substitutions variants of given segment

    array - array listing segment variants observed in all correct reads
    segment_number - segment number to plot (1 to 8 range)
    ax - matplotlib axes to be used for plotting

    Returns: None
    """

    c = dict(sorted(Counter(array[:,segment_number-1]).items()))
    substitutions = substitutions_dict[segment_number]
    xlabels = prep_segment_labels(substitutions)
    for i in range(0,len(xlabels)):
        if i not in c.keys():
            c[i]=0

    negative_keys = [key for key in c if key < 0]
    for key in negative_keys:
        del c[key]

    total = np.sum(list(c.values()))
    probabilities = np.array(list(c.values()))/total
    ax.bar(list(c.keys()),probabilities,color=segment_variant_colors,linewidth=0.75,edgecolor='k',width=0.7)
    ax.margins(x=0.15)
    ax.errorbar(
        x=list(c.keys()),
        y=probabilities,
        yerr=[calculate_95_confidence_interval(total,p) for p in probabilities],
        fmt='none',
        ecolor="black",
        elinewidth=1,
        capsize=2,
    )
    ax.set_ylim((0,0.6))
    ax.set_xticks([i for i in range(0,len(probabilities))],xlabels,rotation=90)

def update_segment_string_with_silent_substitution(row):
    """
    Add information about presence/absence of silent substitution to the segment string

    row - pandas dataframe row

    Returns: Updated segment string
    """
    s_list = list(row['segment_string'])
    #Get current value of segment 5
    current_val = int(s_list[4])

    #Update segment 5 value if silent substitution is present
    if int(row['silent_substitution']) == 1:
        if current_val < 2:
            s_list[4] = str(current_val + 2)
    return "".join(s_list)


df_correct_reads = df.loc[df["read_correct"]==True]

df_correct_reads.loc[:,'segment_string'] = df_correct_reads.apply(update_segment_string_with_silent_substitution, axis=1)

plots = [[['0'],"barplot_segment_occurence_probability_filtered_sequences_GTG.png"],
        [['1'],"barplot_segment_occurence_probability_filtered_sequences_GTT.png"],
        [['0','1'],"barplot_segment_occurence_probability_ALL_library.png"],
        ]

for plot in plots:
    sn.set_context("paper")
    sn.set_style("ticks")
    fig, axs = plt.subplots(1, 8, sharey=True,width_ratios=[0.5,1,1,1,1,1,1,1])
    segment_array = np.array([list(s) for s in df_correct_reads.loc[df_correct_reads['silent_substitution'].isin(plot[0]),"segment_string"].to_list()])
    segment_array = segment_array.astype(int)

    for i in range(1,9):
        plot_segment_probabilities(segment_array,i,axs[i-1])

    fig=plt.gcf()
    fig.set_size_inches(8,3)
    plt.tight_layout()
    plt.savefig(plot[1],dpi=300)
    plt.clf()
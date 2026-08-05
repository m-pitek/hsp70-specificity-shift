import seaborn as sn
from matplotlib import pyplot as plt
import polars as pl
import re
import io

def extract_block(file_path, start_pattern, end_pattern):
    """
    Extract data block between 2 regex patterns (not including regex patterns)

    file_path - name of the file to process
    start_pattern - regex pattern defining the beginning of the data block
    end_pattern - regex pattern defining the ned of the data block

    Returns: io.StringIO containing extracted data block
    """
    start_re = re.compile(start_pattern)
    end_re = re.compile(end_pattern)

    collecting = False
    block = []

    with open(file_path, 'r') as f:
        for line in f:
            if start_re.search(line):
                collecting = True
                continue

            if end_re.search(line):
                break

            if collecting:
                block.append(line)

    return io.StringIO("".join(block))

def load_spectrum(filename, protein_name):
    """
    Load CD spectrum from a JASCO exported .txt file

    filename - name of the file containing the spectrum
    protein_name - name of the protein to add to the returned DataFrame

    Returns: A polars DataFrame containing the CD spectrum
    """
    csv_buffer = extract_block(filename, r"XYDATA", r"##### Extended Information")
    df = pl.read_csv(csv_buffer,separator="\t",has_header=False)
    df = df.rename({"column_1":"wavelength","column_2":"CD","column_3":"HT"})
    df_cleaned = (
    df.drop_nulls()
        .with_columns(
            pl.col("wavelength").str.replace(",", ".").cast(pl.Float64),
            pl.col("CD").str.replace(",", ".").cast(pl.Float64),
            pl.col("HT").str.replace(",", ".").cast(pl.Float64),
        ).with_columns(
            pl.lit(protein_name).alias("name")
        )
    )
    return df_cleaned

def load_melting_curve_and_fit(melting_data_filename, melting_fit_filename, sample_name):
    """
    Load data files from CD melting files

    melting_data_filename - name of the txt file containing melting data (JASCO)
    melting_fit_filename - csv file containing melting curve fit (chira_kit)
    sample_name - sample name to be added as a column to the final dataframe

    Returns: A polars DataFrame containing the melting curve data points and the ChiraKit fit
    """

    csv_buffer = extract_block(melting_data_filename, r"XYDATA", r"##### Extended Information")
    df_melting_data = pl.read_csv(csv_buffer,separator="\t",has_header=False)
    df_melting_fit = pl.read_csv(melting_fit_filename,separator=",",has_header=True)
    df_melting_fit = df_melting_fit.drop("legend")

    df_melting_data = df_melting_data.rename({"column_1":"temperature_(°C)","column_2":"CD_signal_value","column_3":"HT"})
    df_melting_data_cleaned = (
    df_melting_data.drop_nulls()
        .with_columns(
            pl.col("temperature_(°C)").str.replace(",", ".").cast(pl.Float64),
            pl.col("CD_signal_value").str.replace(",", ".").cast(pl.Float64),
            pl.col("HT").str.replace(",", ".").cast(pl.Float64),
        ).with_columns(
            pl.lit(sample_name).alias("sample_name")
        )
    )
    return df_melting_data_cleaned.join(df_melting_fit,on="temperature_(°C)")

def plot_spectra(df_list, wavelength_range, colors, output_filename, ylim):
    """
    Plot CD spectra

    df_list - list of polars DataFrames containing spectra to plot
    wavelength_range - a tuple specifying wavelength range to plot
    colors - list specifying colors assigned to the spectra
    output_filename - name of the output .png file to save the plot
    ylim - a tuple specifying y-axis limits

    Returns: None
    """
    sn.set_style("ticks")
    sn.set_context("talk")
    df = pl.concat(df_list)
    df = df.with_columns((pl.col("CD")/1000).alias("kCD"))
    df = df.filter((wavelength_range[0]<pl.col("wavelength")) & (pl.col("wavelength")<wavelength_range[1]))
    ax = sn.lineplot(data=df,x="wavelength",y="kCD",hue="name",legend=False, palette=colors, linewidth=1.5)
    ax.set_xlim([180,265])
    ax.set_ylim(ylim)
    ax.set_ylabel("$[\\theta]_\\mathrm{MRE}$, 10$^{3}$$\\cdot$deg$\\cdot$cm$^2\\cdot$dmol$^{-1}$")
    ax.set_xlabel("Wavelength, nm")
    plt.gcf().set_size_inches([5.6,4])
    plt.tight_layout()
    plt.savefig(output_filename,dpi=300,transparent=True)
    plt.close('all')

def plot_melting_curves_and_fits(df_list, colors, output_filename, ylim, shift_to_zero=False, figsize=[6,4]):
    """
    Plot CD melting curves and ChiraKit fits

    df_list - list of polars DataFrames containing melting curves and ChiraKit fits to plot
    colors - list specifying colors assigned to the spectra
    output_filename - name of the output .png file to save the plot
    ylim - a tuple specifying y-axis limits
    shift_to_zero - if True, shift the melting curve such that signal of the first data point is 0
    figsize - a list specifying size of the matplotlib figure

    Returns: None
    """
    df = pl.concat(df_list)
    if shift_to_zero:
        df = df.with_columns([
    (
        pl.col("CD_signal_value_fit") -
        pl.col("CD_signal_value_fit")
        .filter(pl.col("temperature_(°C)") == pl.col("temperature_(°C)").min())
        .first()
        .over("sample_name")
    ).alias("CD_signal_value_fit"),

    (
        pl.col("CD_signal_value") -
        pl.col("CD_signal_value_fit")
        .filter(pl.col("temperature_(°C)") == pl.col("temperature_(°C)").min())
        .first()
        .over("sample_name")
    ).alias("CD_signal_value")
])

    plt.clf()
    plt.close('all')
    sn.set_style("ticks")
    sn.set_context("talk")
    ax = sn.scatterplot(data=df,
                        x="temperature_(°C)",
                        y="CD_signal_value",
                        hue="sample_name",
                        legend=False,
                        palette=colors,
                        s=8)

    sn.lineplot(data=df,
                x="temperature_(°C)",
                y="CD_signal_value_fit",
                hue="sample_name",
                legend=False,
                palette=colors,
                ax=ax,
                linewidth=2)

    ax.set_xlim([18,100])
    ax.set_ylim(ylim)
    wavelength = df["wavelength_(nm)"][0]
    if not shift_to_zero:
        ax.set_ylabel(f"CD$_\\mathrm{{wavelength}}$, mdeg")
    else:
        ax.set_ylabel("$\\Delta$CD$_\\mathrm{"+str(wavelength)+"}$, mdeg")
    ax.set_xlabel("Temperature, °C")
    plt.tight_layout()
    plt.gcf().set_size_inches(figsize)
    plt.savefig(output_filename, dpi=300, transparent=True)

#Plot AncCQ, AncCQ+14

AncCQ_spectrum = load_spectrum("5uM_AncCQ_SBD_CD_ME.txt","AncCQ")
AncCQ_200_melting = load_melting_curve_and_fit("5uM_AncCQ_SBD_Tm_200nm.txt","CD_Melting_Fitted_Data_AncCQ_200nm_chirakit_3_state.csv","AncCQ")
AncCQ_222_melting = load_melting_curve_and_fit("5uM_AncCQ_SBD_Tm_222nm.txt","CD_Melting_Fitted_Data_AncCQ_222nm_chirakit_2_state.csv","AncCQ")

AncCQ_plus14_spectrum = load_spectrum("5uM_AncCQ+14_CD_spec_ME.txt","AncCQ+14")
AncCQ_plus14_200_melting = load_melting_curve_and_fit("5uM_AncCQ+14_Tm_200nm.txt","CD_Melting_Fitted_Data_AncCQ_plus14_200nm_chirakit_3_state.csv","AncCQ+14")
AncCQ_plus14_222_melting = load_melting_curve_and_fit("5uM_AncCQ+14_Tm_222nm.txt","CD_Melting_Fitted_Data_AncCQ_plus14_222nm_chirakit_2_state.csv","AncCQ+14")

color_list = ["#22e7b6","#1A3DD8"]
plot_spectra([AncCQ_spectrum,AncCQ_plus14_spectrum],
             wavelength_range=(185,260),
             colors=color_list,
             output_filename="AncCQ_AncCQ_plus14_SBD_spectra.png",
             ylim=(-20,30))

plot_melting_curves_and_fits([AncCQ_200_melting,AncCQ_plus14_200_melting],
                         color_list,
                         output_filename="AncCQ_AncCQ_plus14_SBD_melting_200nm.png",
                         ylim=[-21,2],
                         shift_to_zero=True)

plot_melting_curves_and_fits([AncCQ_222_melting,AncCQ_plus14_222_melting],
                         color_list,
                         output_filename="AncCQ_AncCQ_plus14_SBD_melting_222nm.png",
                         ylim=[-1,10],
                         shift_to_zero=True,
                         figsize=[5.55,4])

#Plot AncCQ+3, AncCQ+6, AncCQ+T494K, AncCQ+S540N

AncCQ_plus6_spectrum = load_spectrum("5uM_AncCQ+6_CD_spec_ME.txt","AncCQ+6")
AncCQ_plus3_spectrum = load_spectrum("5uM_AncCQ+3_CD_spec_ME.txt","AncCQ+3")
AncCQ_T494K_spectrum = load_spectrum("5uM_AncCQ_T494K_CD_spec_ME.txt","AncCQ+T494K")
AncCQ_S540N_spectrum = load_spectrum("5uM_AncCQ_S540N_CD_spec_ME.txt","AncCQ+S540N")

AncCQ_plus6_200_melting = load_melting_curve_and_fit("5uM_AncCQ_plus6_Tm_200nm.txt","CD_Melting_Fitted_Data_AncCQ_plus6_200nm_chirakit_3_state.csv","AncCQ+6")
AncCQ_plus6_222_melting = load_melting_curve_and_fit("5uM_AncCQ_plus6_Tm_222nm.txt","CD_Melting_Fitted_Data_AncCQ_plus6_222nm_chirakit_2_state.csv","AncCQ+6")

AncCQ_plus3_200_melting = load_melting_curve_and_fit("5uM_AncCQ_plus3_Tm_200nm.txt","CD_Melting_Fitted_Data_AncCQ_plus3_200nm_chirakit_3_state.csv","AncCQ+3")
AncCQ_plus3_222_melting = load_melting_curve_and_fit("5uM_AncCQ_plus3_Tm_222nm.txt","CD_Melting_Fitted_Data_AncCQ_plus3_222nm_chirakit_2_state.csv","AncCQ+3")

AncCQ_T494K_200_melting = load_melting_curve_and_fit("5uM_AncCQ_T494K_Tm_200nm.txt","CD_Melting_Fitted_Data_AncCQ_T494K_200nm_chirakit_3_state.csv","AncCQ+T494K")
AncCQ_T494K_222_melting = load_melting_curve_and_fit("5uM_AncCQ_T494K_Tm_222nm.txt","CD_Melting_Fitted_Data_AncCQ_T494K_222nm_chirakit_2_state.csv","AncCQ+T494K")

AncCQ_S540N_200_melting = load_melting_curve_and_fit("5uM_AncCQ_S540N_Tm_200nm.txt","CD_Melting_Fitted_Data_AncCQ_S540N_200nm_chirakit_3_state.csv","AncCQ+S540N")
AncCQ_S540N_222_melting = load_melting_curve_and_fit("5uM_AncCQ_S540N_Tm_222nm.txt","CD_Melting_Fitted_Data_AncCQ_S540N_222nm_chirakit_2_state.csv","AncCQ+S540N")

color_list = ["#2b28cf","#26B193","#FF7300","#BD0068"]
plot_spectra([AncCQ_plus6_spectrum, AncCQ_plus3_spectrum, AncCQ_T494K_spectrum, AncCQ_S540N_spectrum],
             wavelength_range=(185,260),
             colors=color_list,
             output_filename="AncCQ_SBD_substitution_variants_spectra.png",
             ylim = (-20,30))

plot_melting_curves_and_fits([AncCQ_plus6_200_melting,AncCQ_plus3_200_melting,AncCQ_T494K_200_melting,AncCQ_S540N_200_melting],
                         color_list,
                         output_filename="AncCQ_SBD_substitution_variants_melting_200nm.png",
                         ylim=[-26,2],
                         shift_to_zero=True)

plot_melting_curves_and_fits([AncCQ_plus6_222_melting,AncCQ_plus3_222_melting,AncCQ_T494K_222_melting,AncCQ_S540N_222_melting],
                         color_list,
                         output_filename="AncCQ_SBD_substitution_variants_melting_222nm.png",
                         ylim=[-1,11],
                         shift_to_zero=True)

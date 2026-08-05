import subprocess
import re
from pathlib import Path

def trjcat_subprocess(input_trajectories, index_file, output_filename):
    """
    Concatenate trajectory segments using gmx trjcat

    input_trajectories - list of filest containing trajectory segments (.xtc)
    index_file - gromacs index file (.ndx)
    output_filename - name for the concatenated trajectory (.xtc)

    Returns: None
    """
    cmd = [
        'gmx', 'trjcat',
        '-f', *input_trajectories,
        '-n', str(index_file),
        '-o', str(output_filename),
    ]

    print(f"Running: {' '.join(str(x) for x in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            input='System',
            text=True,
            capture_output=True,
            check=True
        )
        print("Trajectory concatenated successfully.")

    except subprocess.CalledProcessError as e:
        print("Error executing trjcat!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

simulations = ["AncCQ_SBD_LPPVK", "AncCQ_plus14_SBD_LPPVK"]

for simulation in simulations:
    for rep in ["rep1", "rep2", "rep3"]:
        directory = Path(simulation) / rep

        if not directory.exists():
            print(f"Skipping {directory}: Not found")
            continue

        pattern = re.compile(r".*part00.*\.xtc$")
        matches = sorted([str(f) for f in directory.iterdir() if f.is_file() and pattern.match(f.name)])
        if not matches:
            print(f"No trajectory parts found in {directory}")
            continue

        output_filename = str(directory / (Path(matches[0]).stem.split(".")[0] + "_all.xtc"))
        index_file = f"{simulation}/index.ndx"

        print(f"Processing: {simulation} {rep}")
        trjcat_subprocess(matches, index_file, output_filename)

import subprocess
import re
from pathlib import Path

def trjconv_subprocess(input_trajectory, index_file, directory, tpr_filename):
    """
    Process concatenated trajectory (.xtc) with gmx trjconv

    input_trajectory - concatenated trajectory (.xtc)
    index_file -  GROMACS index file (.ndx)
    directory - path to the directory where input_trajectory is located
    tpr_filename - path to the GROMACS .tpr file

    Returns: None
    """
    #=================== GMX TRJCONV WHOLE========================================================
    output_filename = str(directory / (Path(input_trajectory).name.replace("_all.xtc","_whole.xtc")))

    cmd = [
        'gmx', 'trjconv',
        '-f', input_trajectory,
        '-s', tpr_filename,
        '-n', str(index_file),
        '-o', str(output_filename),
        '-pbc', "whole",
    ]

    print(f"Running: {' '.join(str(x) for x in cmd)}")
    if not Path(output_filename).exists() and not Path(str(directory / (Path(input_trajectory).name.replace("_all.xtc","_cluster.xtc")))).exists() and not Path(str(directory / (Path(input_trajectory).name.replace("_all.xtc","_mol_center.xtc")))).exists():
        try:
            result = subprocess.run(
                cmd,
                input='System',
                text=True,
                capture_output=True,
                check=True
            )
            print("trjconv whole run successfully.")

        except subprocess.CalledProcessError as e:
            print("Error executing trjconv!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)

    #=================== GMX TRJCONV CLUSTER========================================================
    input_filename = output_filename
    output_filename = str(directory / (Path(input_trajectory).name.replace("_all.xtc","_cluster.xtc")))
    cmd = [
        'gmx', 'trjconv',
        '-f', input_filename,
        '-s', tpr_filename,
        '-n', str(index_file),
        '-o', str(output_filename),
        '-pbc', "cluster",
    ]

    print(f"Running: {' '.join(str(x) for x in cmd)}")
    if not Path(output_filename).exists() and not Path(str(directory / (Path(input_trajectory).name.replace("_all.xtc","_mol_center.xtc")))).exists():
        try:
            result = subprocess.run(
                cmd,
                input='Protein\nSystem\n',
                text=True,
                capture_output=True,
                check=True
            )
            print("trjconv cluster run successfully.")

        except subprocess.CalledProcessError as e:
            print("Error executing trjconv!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)

    #Remove processing intermediate to save space
    try:
        result = subprocess.run(
            ["rm", input_filename],
            text=True,
            capture_output=True,
            check=True
        )

    except subprocess.CalledProcessError as e:
        print("Error removing whole traj!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

#=================== GMX TRJCONV MOL CENTER ========================================================
    input_filename = output_filename
    output_filename = str(directory / (Path(input_trajectory).name.replace("_all.xtc","_mol_center.xtc")))
    cmd = [
        'gmx', 'trjconv',
        '-f', input_filename,
        '-s', tpr_filename,
        '-n', str(index_file),
        '-o', str(output_filename),
        '-pbc', "mol",
        '-ur', "compact",
        '-center'
    ]

    print(f"Running: {' '.join(str(x) for x in cmd)}")
    if not Path(output_filename).exists():
        try:
            result = subprocess.run(
                cmd,
                input='Protein\nSystem\n',
                text=True,
                capture_output=True,
                check=True
            )
            print("trjconv mol center run successfully.")

        except subprocess.CalledProcessError as e:
            print("Error executing trjconv!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)

    #Remove processing intermediate to save space
    try:
        result = subprocess.run(
            ["rm", input_filename],
            text=True,
            capture_output=True,
            check=True
        )

    except subprocess.CalledProcessError as e:
        print("Error removing clustered traj!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)


simulations = ["AncCQ_SBD_LPPVK", "AncCQ_plus14_SBD_LPPVK"]

for simulation in simulations:
    for rep in ["rep1", "rep2", "rep3"]:
        directory = Path(simulation) / rep

        if not directory.exists():
            print(f"Skipping {directory}: Not found")
            continue

        pattern = re.compile(r".*_all\.xtc$")
        pattern_tpr = re.compile(r".*\.tpr$")

        matches = sorted([str(f) for f in directory.iterdir() if f.is_file() and pattern.match(f.name)])
        tpr_matches = sorted(str(f) for f in directory.iterdir() if f.is_file() and pattern_tpr.match(f.name))

        if not matches:
            print(f"No trajectory parts found in {directory}")
            continue
        elif len(matches) != 1:
            print(f"More than one matching trajectory found in {directory}")
            continue

        if not tpr_matches:
            print(f"No tpr file found in {directory}")
            continue
        elif len(tpr_matches) != 1:
            print(f"More than one matching tpr file found in {directory}")
            continue

        index_file = f"{simulation}/index.ndx"

        print(f"Processing: {simulation} {rep}")
        trjconv_subprocess(matches[0], index_file, directory,tpr_matches[0])

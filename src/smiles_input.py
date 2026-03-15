import subprocess
from openff.toolkit import Molecule, Topology
from rdkit.Chem import MolFromSmiles, Descriptors
from scipy.constants import Avogadro
import os
import math

def format_pdb_atom_line(line: str, resname: str, resid: int = 1, chain_id: str = "A") -> str:
    """
    Replace residue columns in PDB file with an actual label, used for grouping later
    """
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return line
    
    chars = list(line.rstrip("\n"))
    # pdb lines are fixed width, so we need to make sure the line is long enough
    while len(chars) < 80:
        chars.append(" ")

    resname = resname[:3].upper()[:3]  # Ensure resname is 3 characters
    resid_str = str(int(resid)).rjust(4)  # Right-justify resid to 4 characters
    chain_id = (chain_id or "A")[0]  # Use first character of chain_id or default to 'A'

    chars[17:20] = list(resname)  # Residue name in columns 18-20
    chars[21] = chain_id  # Chain ID in column 22
    chars[22:26] = list(resid_str)  # Residue ID in columns 23-26

    return "".join(chars) + "\n"

def pdb_from_smiles(smiles : str, filepath = "./pdbs", resname: str = "MOL", resid: int = 1, chain_id: str = "A") -> Molecule:
    """
    Generates a pdb from a SMILES string and saves it as the SMILES.pdb
    """
    m = Molecule.from_smiles(smiles)
    m.generate_conformers()
    os.makedirs(filepath, exist_ok = True)

    pdb_path = f"{filepath}/{smiles}.pdb"
    tmp_path = f"{filepath}/{smiles}_tmp.pdb"

    m.to_file(f"{filepath}/{smiles}.pdb", file_format = "pdb")

    #label residues
    with open(tmp_path, "r") as fin, open(pdb_path, "w") as fout:
        for line in fin:
            fout.write(format_pdb_atom_line(line, resname, resid, chain_id))

    os.remove(tmp_path)

    return m

def read_config(filepath : str) -> dict:
    """
    Reads a config file, generating a dict with all of the information
    """
    config = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            #skip empties
            if not line.strip():
                continue
            #make key value pairs for the dict
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()

                values = value.split()
                config[key] = values
    
    if not config:
        raise ValueError("Config file is empty")

    return config

def get_mol_wt(smiles: str):
    """
    Returns average mol wt from smiles string
    """
    return Descriptors.MolWt(MolFromSmiles(smiles))


def convert_density(density : float, smiles : str):
    """
    Converts density from g/cm3 to molecules per cubic angstrom
    """
    molwt = get_mol_wt(smiles)

    return density / molwt * Avogadro / 1E24

def build_structure_pdbs(config : dict):
    """
    Makes the structure pdbs from the config dicts
    """
    try:
        values = config["solvents"] + config["compounds of interest"]
    except:
        values = config["solvents"]
    molecules = []
    for smiles in values:
        molecules.append(pdb_from_smiles(smiles))
    return config["solvents"], config["compounds of interest"]

def get_boxes(config : dict):
    """
    Generates box size based on number of molecules
    Returns as tuple
    First element is box side length
    Second is list of list for each molecule in config
    """
    solvent_smiles = config["solvents"]
    solvent_quantities = config["number of solvent molecules"]
    solvent_densities = config["density of solvents in g/cm3"]
    converted_solvent_densities = []
    for i in range(len(solvent_smiles)):
        converted_solvent_densities.append(convert_density(
            float(solvent_densities[i]), solvent_smiles[i]))
    
    coi_exists = True
    try:
        coi_smiles = config["compounds of interest"]
        coi_quantities = config["number of CoI molecules"]
        coi_densities = config["density of CoI in g/cm3"]
        converted_coi_densities = []
        for i in range(len(coi_smiles)):
            converted_coi_densities.append(convert_density(
                float(coi_densities[i]), coi_smiles[i]))
        coi_volumes = []
        for i in range(len(converted_coi_densities)):
            coi_volumes.append(float(coi_quantities[i]) / converted_coi_densities[i])
    
    except:
        coi_exists = False

    #now our densities are in atoms per cubic angstrom, we can find the total volume of each molecule
    solvent_volumes = []
    for i in range(len(converted_solvent_densities)):
        solvent_volumes.append(float(solvent_quantities[i]) / converted_solvent_densities[i])

    #now that we have all the volumes, we can figure out the total box size
    total_volume = sum(solvent_volumes)
    if coi_exists:
        total_volume += sum(coi_volumes)
    edge_length = total_volume ** (1/3)

    #from the edge length we can figure out how separated the z axes should be
    z_edges = []
    previous_z = 0
    for volume in solvent_volumes:
        z_coord = previous_z + (volume * edge_length / total_volume)
        z_edges.append(z_coord)
        previous_z = z_coord
    
    #now we generate a full box list of lists, first box starts at 0, 0, 0 and goes to x, y, z1; second box from there to x, y, z2, etc.
    boxes = []
    last_z = 0
    for z_edge in z_edges:
        boxes.append([0, 0, last_z, edge_length, edge_length, z_edge])
        last_z = z_edge

    #finally, find the radius of a sphere with the volume of the total of the coi volumes
    #to shove all the cois into
    if coi_exists:
        coi_radius = (3 * sum(coi_volumes) / (4 * math.pi)) ** (1/3)
    else:
        coi_radius = None
    return edge_length, boxes, coi_radius

def write_input(filepath : str, filename : str = "default", tolerance : float = 2.0):
    """
    Writes a .inp file and pdb structure files
    """
    config = read_config(filepath)
    edge_length, boxes, radius = get_boxes(config)
    solvent_smiles = config["solvents"]
    solvent_quantities = config["number of solvent molecules"]
    #if there aren't any compounds of interest, skip this step
    coi_exists = True
    try:
        coi_smiles = config["compounds of interest"]
        coi_quantities = config["number of CoI molecules"]
    except:
        coi_exists = False

    center = edge_length / 2

    for smile in solvent_smiles:
        pdb_from_smiles(smile)
    if coi_exists:
        for smile in coi_smiles:
            pdb_from_smiles(smile)

    with open(f"{filename}.inp", "w") as f:
        f.write(f"tolerance {tolerance}\n")
        f.write(f"filetype pdb\n")
        f.write(f"output {filename}.pdb\n")

        for i in range(len(solvent_smiles)):
            f.write(f"structure pdbs/{solvent_smiles[i]}.pdb\n")
            f.write(f"\tnumber {solvent_quantities[i]}\n")
            box = boxes[i]
            f.write(f"\tinside box {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]}\n")
            if coi_exists:
                f.write(f"\toutside sphere {center} {center} {center} {radius}\n")
            f.write("end structure\n\n")
        if(coi_exists):
            for i in range(len(coi_smiles)):
                f.write(f"structure pdbs/{coi_smiles[i]}.pdb\n")
                f.write(f"\tnumber {coi_quantities[i]}\n")
                f.write(f"\tinside sphere {center} {center} {center} {radius}\n")
                f.write("end structure\n\n")
    
    f.close()

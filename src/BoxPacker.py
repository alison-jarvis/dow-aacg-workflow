#from src.smiles_input import read_config, get_boxes
from smiles_input import read_config, get_boxes
import os
from pathlib import Path
import subprocess
from openff.toolkit import Molecule, Topology
from rdkit.Chem import MolFromSmiles, Descriptors
from scipy.constants import Avogadro

class BoxPacker:

    def __init__(self, filepath_to_config, filename = "default", tolerance = 2.0, overwrite = False):
        self.filepath = filepath_to_config
        self.filename = filename
        self.tolerance = tolerance
        self.config = read_config(filepath_to_config)
        self.project_dir = "_".join(self.config["project name"])
        self.pdb_dir = f"{self.project_dir}/pdbs"
        path = Path(self.project_dir)

        #mixture types: 0 = fully separated, 1 = fully mixed, 2 = separated, COI dissolved into specified solvent
        self.mixture_type = self.config["mixture type"]
        if(len(self.mixture_type) != 1):
            raise ValueError("Please enter either 0, 1, or 2 for mixture type")
        self.mixture_type = int(self.mixture_type[0])
        try:
            self.mixture_parameters = self.config["mixture parameters"]
        except:
            self.mixture_parameters = None

        if path.exists() and not overwrite:
            raise FileExistsError(f"The directory '{path}' already exists. Rename project or set overwrite flag in constructor.")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Project directory '{path}' created")

        os.makedirs(self.pdb_dir, parents=True, exist_ok=True)

        self.solvents, self.cois = self.build_structure_pdbs()
        self.edge_length, self.boxes, self.coi_radius = get_boxes(self.config)
        self.write_input()

    def format_pdb_atom_line(self, line: str, resname: str, resid: int=1, chain_id: str="A") -> str:
        """
        Replace residue columns in PDB file with an actual label, used for grouping later
        """
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            return line
        
        chars = list(line.rstrip("\n"))

        # pdb lines are fixed width, so we need to make sure the line is long enough
        while len(chars) < 80:
            chars.append(" ")

        resname = resname.upper()[:3].rjust(3)  # Ensure resname is 3 characters
        resid_str = str(int(resid)).rjust(4)  # Ensure resid is 4 characters
        chain_id = (chain_id or "A")[0]  # Use first character of chain_id or default to 'A'

        chars[17:20] = list(resname)  # Residue name in columns 18-20
        chars[21] = chain_id  # Chain ID in column 22
        chars[22:26] = list(resid_str)  # Residue ID in columns 23-26

        return "".join(chars) + "\n"

    def pdb_from_smiles(self, smiles : str, resname: str="MOL", resid: int=1, chain_id: str="A") -> Molecule:
        """
        Generates a pdb from a SMILES string and saves it as the SMILES.pdb
        """
        m = Molecule.from_smiles(smiles)
        m.generate_conformers()

        pdb_path = f"{self.pdb_dir}/{smiles}.pdb"
        tmp_path = f"{self.pdb_dir}/{smiles}_tmp.pdb"

        m.to_file(tmp_path, file_format = "pdb")

        #label residues
        with open(tmp_path, "r") as fin, open(pdb_path, "w") as fout:
            for line in fin:
                fout.write(self.format_pdb_atom_line(line, resname, resid, chain_id))

        os.remove(tmp_path)

        return m

    def build_structure_pdbs(self):
        """
        Makes the structure pdbs from the config dicts
        """
        try:
            values = self.config["solvents"] + self.config["compounds of interest"]
        except:
            values = self.config["solvents"]
        molecules = []
        for smiles in values:
            molecules.append(self.pdb_from_smiles(smiles))
        return self.config["solvents"], self.config["compounds of interest"]
    
    def write_input(self):
        """
        Writes a .inp file and pdb structure files
        """
        solvent_smiles = self.config["solvents"]
        solvent_quantities = self.config["number of solvent molecules"]
        #if there aren't any compounds of interest, skip this step
        coi_exists = True
        try:
            coi_smiles = self.config["compounds of interest"]
            coi_quantities = self.config["number of CoI molecules"]
        except:
            coi_exists = False

        center = self.edge_length / 2

        with open(f"{self.project_dir}/{self.filename}.inp", "w") as f:
            f.write(f"pbc 0. 0. 0. {self.edge_length} {self.edge_length} {self.edge_length}\n")
            if self.mixture_type == 0:
                f.write(f"tolerance {self.tolerance}\n")
                f.write(f"filetype pdb\n")
                f.write(f"output {self.project_dir}/{self.filename}.pdb\n")

                for i in range(len(solvent_smiles)):
                    f.write(f"structure {self.project_dir}/pdbs/{solvent_smiles[i]}.pdb\n")
                    f.write(f"\tnumber {solvent_quantities[i]}\n")
                    box = self.boxes[i]
                    f.write(f"\tinside box {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]}\n")
                    if coi_exists:
                        f.write(f"\toutside sphere {center} {center} {center} {self.coi_radius}\n")
                    f.write("end structure\n\n")
                if(coi_exists):
                    for i in range(len(coi_smiles)):
                        f.write(f"structure {self.project_dir}/pdbs/{coi_smiles[i]}.pdb\n")
                        f.write(f"\tnumber {coi_quantities[i]}\n")
                        f.write(f"\tinside sphere {center} {center} {center} {self.coi_radius}\n")
                        f.write("end structure\n\n")

            elif self.mixture_type == 1:
                f.write(f"tolerance {self.tolerance}\n")
                f.write(f"filetype pdb\n")
                f.write(f"output {self.project_dir}/{self.filename}.pdb\n")
                #no need to check boxes, just cram everything into the box 0, 0, 0, edgelength, edgelength, edgelength
                for i in range(len(solvent_smiles)):
                    f.write(f"structure {self.project_dir}/pdbs/{solvent_smiles[i]}.pdb\n")
                    f.write(f"\tnumber {solvent_quantities[i]}\n")
                    f.write(f"\tinside box 0. 0. 0. {self.edge_length} {self.edge_length} {self.edge_length}\n")
                    f.write("end structure\n\n")
                if(coi_exists):
                    for i in range(len(coi_smiles)):
                        f.write(f"structure {self.project_dir}/pdbs/{coi_smiles[i]}.pdb\n")
                        f.write(f"\tnumber {coi_quantities[i]}\n")
                        f.write(f"\tinside box 0. 0. 0. {self.edge_length} {self.edge_length} {self.edge_length}\n")
                        f.write("end structure\n\n")

            elif self.mixture_type == 2:
                #check if there are compounds of interest
                if not coi_exists:
                    raise ValueError("For mixture type 2 (COIs dissolved in solvent) COIs must be inputted.")
                #check if mixture params match number of cois
                if self.mixture_parameters is None or len(self.mixture_parameters) != len(self.cois):
                    raise ValueError("Number of mixture parameters must match with number of COIs")
                
                f.write(f"tolerance {self.tolerance}\n")
                f.write(f"filetype pdb\n")
                f.write(f"output {self.project_dir}/{self.filename}.pdb\n")

                for i in range(len(solvent_smiles)):
                    f.write(f"structure {self.project_dir}/pdbs/{solvent_smiles[i]}.pdb\n")
                    f.write(f"\tnumber {solvent_quantities[i]}\n")
                    box = self.boxes[i]
                    f.write(f"\tinside box {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]}\n")
                    f.write("end structure\n\n")

                for i in range(len(coi_smiles)):
                    f.write(f"structure {self.project_dir}/pdbs/{coi_smiles[i]}.pdb\n")
                    f.write(f"\tnumber {coi_quantities[i]}\n")
                    param = int(self.mixture_parameters[i])
                    box = self.boxes[param]
                    f.write(f"\tinside box {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]}\n")
                    f.write("end structure\n\n")
        f.close()

    def pack_the_mol(self):
        try:
            result = subprocess.run(f"packmol < {self.project_dir}/{self.filename}.inp",
                       shell = True, capture_output = True, text = True, check = True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
        
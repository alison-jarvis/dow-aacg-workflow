from rdkit import Chem
from collections import deque

def generate_auto_mapping(smiles : str, cg_resname = "MOL"):
    """generates a CG mapping dict for any molecule using graph traversal"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    #add explicit hydrogens
    mol = Chem.AddHs(mol)

    element_counts = {}
    pdb_atom_names = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol().upper()
        element_counts[sym] = element_counts.get(sym, 0) + 1
        pdb_atom_names[atom.GetIdx()] = f"{sym}{element_counts[sym]}"

    #separate heavy atoms from hydrogens
    heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    num_heavy = len(heavy_atoms)

    #preassign expected bead sizes
    if num_heavy <= 4:
        target_sizes = [num_heavy]
    elif num_heavy == 5:
        target_sizes = [3, 2]
    else:
        target_sizes = [3] * (num_heavy // 3)
        remainder = num_heavy % 3

        if remainder == 1:
            target_sizes[0] += 1
        elif remainder == 2:
            target_sizes[0] += 1
            target_sizes[-1] += 1

    unassigned_heavy = set(a.GetIdx() for a in heavy_atoms)
    bead_chunks = []

    def get_heavy_degree(atom):
        #return the number of neighbors that are heavy atoms (if 1, we're at the end of a chain)
        return sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1)
    
    #greedy bfs to partition graph
    while unassigned_heavy:
        #try to start at end, otherwise pick whatever
        start_candidates = [i for i in unassigned_heavy if get_heavy_degree(mol.GetAtomWithIdx(i)) <= 1]
        start_idx = start_candidates[0] if start_candidates else list(unassigned_heavy)[0]

        current_target = target_sizes.pop(0) if target_sizes else 3

        current_chunk = []
        queue = deque([start_idx])

        while queue and len(current_chunk) < current_target:
            curr = queue.popleft()

            if curr in unassigned_heavy:
                current_chunk.append(curr)
                unassigned_heavy.remove(curr)
                
                #queue neighboring heavy atoms that haven't been assigned to a chunk
                atom = mol.GetAtomWithIdx(curr)
                for nbr in atom.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx in unassigned_heavy and nbr_idx not in queue:
                        queue.append(nbr_idx)

        bead_chunks.append(current_chunk)

    beads_output = []
    atom_to_bead = {}

    for i, heavy_chunk in enumerate(bead_chunks):
        atom_names = []
        bead_name = f"B{i + 1}"

        #add heavy atoms to this bead
        for idx in heavy_chunk:
            atom = mol.GetAtomWithIdx(idx)
            atom_names.append(pdb_atom_names[idx])
            atom_to_bead[idx] = bead_name

            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    h_idx = nbr.GetIdx()
                    atom_names.append(pdb_atom_names[h_idx])
                    atom_to_bead[h_idx] = bead_name
                
        beads_output.append({
            "type" : f"CG{i + 1}",
            "name" : f"B{i + 1}",
            "atom_names" : atom_names
        })
    
    cg_bonds_set = set()
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        b1 = atom_to_bead.get(a1)
        b2 = atom_to_bead.get(a2)

        if b1 and b2 and b1 != b2:
            cg_bonds = tuple(sorted([b1, b2]))
            cg_bonds_set.add(cg_bonds)
        
    bonds_output = sorted(list(cg_bonds_set))

    return {
        cg_resname: {
            "cg_resname" : cg_resname,
            "beads" : beads_output,
            "bonds" : bonds_output
        }
    }
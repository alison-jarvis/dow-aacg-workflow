import re
import csv
import numpy as np
import MDAnalysis as mda
from pathlib import Path
import sys


def atom_number(atom_name: str) -> int:
    m = re.search(r"(\d+)$", atom_name)
    if m is None:
        raise ValueError(f"Could not parse numeric suffix from atom name '{atom_name}'")
    return int(m.group(1))


def minimum_image(delta, box_lengths):
    """
    Apply minimum image convention for an orthorhombic box
    delta: (..., 3)
    box_lengths: (3,)
    """
    return delta - box_lengths * np.round(delta / box_lengths)


def build_bead_mapping(universe):
    """
    Returns a list of bead definitions
    Each bead definition is a dict with:
      - bead_type: WAT, D1, D2, D3
      - atomgroup: MDAnalysis AtomGroup
      - resid: residue id for the CG residue
      - resname: residue name for the CG residue
      - atomname: atom/bead name in the CG PDB
      - mol_index: molecule index
    """
    bead_defs = []
    mol_index = 1
    found_resnames = sorted(set(res.resname for res in universe.residues))
    print("Found residue names:", found_resnames)

    for res in universe.residues:
        if res.resname == "H2O":
            bead_defs.append({
                "bead_type": "WAT",
                "atomgroup": res.atoms,
                "resid": mol_index,
                "resname": "WAT",
                "atomname": "W",
                "mol_index": mol_index,
            })
            mol_index += 1

        elif res.resname == "C12":
            carbon_atoms = [a for a in res.atoms if a.name.startswith("C")]
            carbon_atoms = sorted(carbon_atoms, key=lambda a: atom_number(a.name))

            if len(carbon_atoms) != 12:
                raise ValueError(
                    f"Expected 12 carbon atoms for dodecane residue {res.resid}, found {len(carbon_atoms)}"
                )

            carbon_chunks = [
                carbon_atoms[0:4],
                carbon_atoms[4:8],
                carbon_atoms[8:12],
            ]

            for bead_idx, chunk in enumerate(carbon_chunks, start=1):
                bead_atom_indices = set()

                for carbon in chunk:
                    bead_atom_indices.add(carbon.index)
                    for nbr in carbon.bonded_atoms:
                        if nbr.name.startswith("H"):
                            bead_atom_indices.add(nbr.index)

                bead_atom_indices = sorted(bead_atom_indices)
                bead_ag = universe.atoms[bead_atom_indices]

                bead_defs.append({
                    "bead_type": f"D{bead_idx}",
                    "atomgroup": bead_ag,
                    "resid": mol_index,
                    "resname": "DOD",
                    "atomname": f"B{bead_idx}",
                    "mol_index": mol_index,
                })

            mol_index += 1

        elif res.resname == "C4H":
            bead_defs.append({
                "bead_type": "DIO",
                "atomgroup": res.atoms,
                "resid": mol_index,
                "resname": "DIO",
                "atomname": "D",
                "mol_index": mol_index,
            })
            mol_index += 1

        else:
            raise ValueError(
                f"Unsupported residue name '{res.resname}'. "
                f"Found residue names in topology: {found_resnames}"
            )

    return bead_defs


def compute_bead_positions(universe, bead_defs, start=None, stop=None, step=None):
    """
    Compute COM bead positions for each frame
    Returns:
      positions: (n_frames, n_beads, 3)
      box_lengths: (n_frames, 3)
    """
    traj = universe.trajectory[start:stop:step]
    n_frames = len(traj)
    n_beads = len(bead_defs)

    positions = np.zeros((n_frames, n_beads, 3), dtype=np.float64)
    box_lengths = np.zeros((n_frames, 3), dtype=np.float64)

    for fi, ts in enumerate(traj):
        box_lengths[fi] = ts.dimensions[:3]

        for bi, bead in enumerate(bead_defs):
            positions[fi, bi] = bead["atomgroup"].center_of_mass()

    return positions, box_lengths


def write_cg_pdb(output_path, bead_defs, positions_frame, conect=True):
    """
    Write a single-frame starting CG PDB from bead COM positions
    """
    serial = 1
    lines = []
    serial_map = {}

    for bi, bead in enumerate(bead_defs):
        x, y, z = positions_frame[bi]
        atomname = bead["atomname"][:4].rjust(4)
        resname = bead["resname"][:3].rjust(3)
        resid = int(bead["resid"])
        chain = "A"

        #pdb has fixed width columns, need to be careful with formatting
        line = (
            f"ATOM  {serial:5d} {atomname} {resname} {chain}{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          "
            f"{'C':>2s}\n"
        )
        lines.append(line)
        serial_map[bi] = serial
        serial += 1

    if conect:
        by_mol = {}
        for bi, bead in enumerate(bead_defs):
            by_mol.setdefault(bead["mol_index"], []).append((bi, bead["atomname"]))

        for mol_id, items in by_mol.items():
            atomname_to_bi = {name: bi for bi, name in items}
            if {"B1", "B2", "B3"}.issubset(atomname_to_bi):
                s1 = serial_map[atomname_to_bi["B1"]]
                s2 = serial_map[atomname_to_bi["B2"]]
                s3 = serial_map[atomname_to_bi["B3"]]
                lines.append(f"CONECT{s1:5d}{s2:5d}\n")
                lines.append(f"CONECT{s2:5d}{s1:5d}{s3:5d}\n")
                lines.append(f"CONECT{s3:5d}{s2:5d}\n")

    lines.append("END\n")

    with open(output_path, "w") as f:
        f.writelines(lines)

def rdf_for_pair(positions, box_lengths, sel_a, sel_b, r_max=20.0, n_bins=200):
    """
    Compute average RDF over all frames for a pair of bead selections.
    positions: (n_frames, n_beads, 3)
    box_lengths: (n_frames, 3)
    sel_a, sel_b: lists/arrays of bead indices
    """
    sel_a = np.asarray(sel_a, dtype=int)
    sel_b = np.asarray(sel_b, dtype=int)

    #histogram bins
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)

    hist = np.zeros(n_bins, dtype=np.float64)
    norm = np.zeros(n_bins, dtype=np.float64)

    same_selection = np.array_equal(np.sort(sel_a), np.sort(sel_b))

    for fi in range(positions.shape[0]):
        pos_a = positions[fi, sel_a]
        pos_b = positions[fi, sel_b]
        box = box_lengths[fi]
        volume = np.prod(box)

        if len(pos_a) == 0 or len(pos_b) == 0:
            continue

        delta = pos_a[:, None, :] - pos_b[None, :, :]
        delta = minimum_image(delta, box)
        dist = np.linalg.norm(delta, axis=-1)

        if same_selection:
            iu = np.triu_indices(len(pos_a), k=1)
            dvals = dist[iu]
            n_a = len(pos_a)
            expected = (n_a * (n_a - 1) / (2.0 * volume)) * shell_volumes
        else:
            dvals = dist.ravel()
            n_a = len(pos_a)
            n_b = len(pos_b)
            rho_b = n_b / volume
            expected = n_a * rho_b * shell_volumes

        hist += np.histogram(dvals, bins=edges)[0]
        norm += expected

    g_r = np.divide(hist, norm, out=np.zeros_like(hist), where=norm > 0)
    return r, g_r


def save_rdfs_csv(output_csv, rdf_results):
    """
    rdf_results
    """
    keys = list(rdf_results.keys())
    r = rdf_results[keys[0]][0]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r"] + keys)
        for i in range(len(r)):
            row = [r[i]] + [rdf_results[k][1][i] for k in keys]
            writer.writerow(row)


def build_cg_outputs(topology_path, trajectory_path, frame_index=0):
    project_dir = Path(topology_path).parent

    universe = mda.Universe(topology_path, trajectory_path)

    bead_defs = build_bead_mapping(universe)
    positions, box_lengths = compute_bead_positions(universe, bead_defs)

    bead_types = [b["bead_type"] for b in bead_defs]
    type_to_indices = {}
    for i, bt in enumerate(bead_types):
        type_to_indices.setdefault(bt, []).append(i)

    write_cg_pdb(
        output_path=project_dir / "cg_start.pdb",
        bead_defs=bead_defs,
        positions_frame=positions[frame_index],
        conect=True,
    )

    rdf_results = {}
    pair_list = [
        ("WAT", "WAT"),
        ("WAT", "DIO"),
        ("WAT", "D1"),
        ("WAT", "D2"),
        ("WAT", "D3"),
        ("DIO", "DIO"),
        ("DIO", "D1"),
        ("DIO", "D2"),
        ("DIO", "D3"),
        ("D1", "D1"),
        ("D1", "D2"),
        ("D1", "D3"),
        ("D2", "D2"),
        ("D2", "D3"),
        ("D3", "D3"),
    ]

    for a, b in pair_list:
        r, g = rdf_for_pair(
            positions,
            box_lengths,
            type_to_indices[a],
            type_to_indices[b],
            r_max=20.0,
            n_bins=200,
        )
        rdf_results[f"{a}-{b}"] = (r, g)

    save_rdfs_csv(project_dir / "cg_rdfs.csv", rdf_results)

    print(f"Wrote {project_dir / 'cg_start.pdb'}")
    print(f"Wrote {project_dir / 'cg_rdfs.csv'}")



def main():
    import sys

    topology_path = sys.argv[1]
    trajectory_path = sys.argv[2]
    frame_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    build_cg_outputs(topology_path, trajectory_path, frame_index=frame_index)


if __name__ == "__main__":
    main()
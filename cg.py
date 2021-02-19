import argparse
import mdtraj as md
from pathlib import Path
from dbm.ff import *
from dbm.mol import *
from dbm.box import *
import numpy as np


def main():

    # ## Read mapping

    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("ff")
    args = parser.parse_args()

    aa_dir = Path(args.dir) / "aa"
    cg_dir = Path(args.dir) / "cg"

    ff = FF(args.ff)


    for aa_path in aa_dir.glob('*.gro'):
        cg_path = cg_dir / aa_path.name
        file_name = aa_path.stem
        box = Box(aa_path)
        atoms, beads, mols = [], [], []
        if cg_path.exists():
            raise Exception('CG file already exists!')
        else:
            aa = md.load(str(aa_path))

            for res in aa.topology.residues:
                mols.append(Mol(res.name))
                map_file = Path("./data/mapping/")/ (res.name + ".map")
                res_beads = {}
                for line in read_between("[map]", "[/map]", map_file):
                    type_name = line.split()[1]
                    atom_ndx, atom_type, bead_ndx, bead_type = int(line.split()[0])-1, line.split()[1], int(line.split()[2])-1, line.split()[3]
                    if bead_ndx not in res_beads.keys():
                        beads.append(Bead(mols[-1],
                                          np.zeros((3,)),
                                          ff.bead_types[bead_type]))
                        res_beads[bead_ndx] = beads[-1]
                        mols[-1].add_bead(beads[-1])
                    atoms.append(Atom(res_beads[bead_ndx],
                                      mols[-1],
                                      res_beads[bead_ndx].center,
                                      ff.atom_types[type_name],
                                      box.move_inside(aa.xyz[0, Atom.index])))
                    mols[-1].add_atom(atoms[-1])
                    res_beads[bead_ndx].add_atom(atoms[-1])

        Atom.index = 0
        Bead.index = 0
        Mol.index = 0

        for bead in beads:
            aa_pos = [a.ref_pos for a in bead.atoms]
            aa_mass = [a.type.mass for a in bead.atoms]
            #periodic boundary...
            aa_pos = [box.diff_vec(pos - aa_pos[0])+aa_pos[0] for pos in aa_pos]
            #print(aa_pos[0])

            com, tot_mass = 0.0, 0.0
            for pos, mass in zip(aa_pos, aa_mass):
                #print(pos, mass)
                com += pos * mass
            #print(":::::::::::::::::::::::::::")
            com = box.move_inside(com / sum(aa_mass))
            bead.center = com
            print(com)



        with open(cg_path, 'w') as f:
            f.write('{:s}\n'.format('CG structure from '+file_name))
            f.write('{:5d}\n'.format(len(beads)))

            for b in beads:
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    b.mol.index,
                    b.mol.name,
                    b.type.name+str(b.mol.beads.index(b)+1),
                    b.index,
                    b.center[0],
                    b.center[1],
                    b.center[2],
                    0, 0, 0))

            f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                box.dim[0][0],
                box.dim[1][1],
                box.dim[2][2],
                box.dim[1][0],
                box.dim[2][0],
                box.dim[0][1],
                box.dim[2][1],
                box.dim[0][2],
                box.dim[1][2]))


if __name__ == "__main__":
    main()

import numpy as np
import os
from os import path
import pickle
import mdtraj as md

for file in os.listdir("./aa"):
    if file.endswith(".gro"):
        cg_file = os.path.join("./cg", file)
        aa_file = os.path.join("./aa", file)

        aa = md.load(aa_file)
        
        n_mol = aa.topology.n_residues

        f_read = open(aa_file, "r")
        bd = np.array(f_read.readlines()[-1].split(), np.float32)
        f_read.close()
        bd = list(bd)
        for n in range(len(bd), 10):
            bd.append(0.0)

        with open(cg_file, 'w') as f:
            f.write('{:s}\n'.format('CG ethylbenzene'))
            f.write('{:5d}\n'.format(n_mol*3))

            n = 1
            for res in aa.topology.residues:
                indices = aa.topology.select("resid "+str(res.index))
                coords = aa.xyz[0,indices]

                pos1 = coords[13]
                pos3 = coords[17]

                c_mass = 12.0110
                h_mass = 1.0080
                tot_mass = 6 * c_mass + 5*h_mass
                pos2 = c_mass*(coords[1]+coords[2]+coords[4]+coords[6]+coords[7]+coords[9]) + h_mass*(coords[0]+coords[3]+coords[5]+coords[8]+coords[10])
                pos2 = pos2 / tot_mass
                print(res)
                print(pos1)
                print(pos2)
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "G065",
                    "B" + str(1),
                    n,
                    pos1[0],
                    pos1[1],
                    pos1[2],
                    0, 0, 0))
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "G065",
                    "S" + str(2),
                    n+1,
                    pos2[0],
                    pos2[1],
                    pos2[2],
                    0, 0, 0))
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "G065",
                    "B" + str(3),
                    n+2,
                    pos3[0],
                    pos3[1],
                    pos3[2],
                    0, 0, 0))
                n = n+3

            f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                bd[0],
                bd[1],
                bd[2],
                bd[3],
                bd[4],
                bd[5],
                bd[6],
                bd[7],
                bd[8]))

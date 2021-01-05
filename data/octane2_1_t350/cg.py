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
            f.write('{:s}\n'.format('CG octane'))
            f.write('{:5d}\n'.format(n_mol*4))

            n = 1
            for res in aa.topology.residues:
                indices = aa.topology.select("resid "+str(res.index))
                coords = aa.xyz[0,indices]

                pos1 = coords[1]
                #pos2 = coords[7]
                #pos3 = coords[13]
                #pos4 = coords[19]

                c_mass = 12.0110
                h_mass = 1.0080
                tot_mass = 2 * c_mass + 4*h_mass
                d_i = 6
                pos2 = c_mass*(coords[7] + 0.5*(coords[4] + coords[10])) + h_mass*(coords[8]+coords[9] + 0.5*(coords[5] + coords[6] + coords[11] + coords[12]))
                pos2 = pos2 / tot_mass
                pos3 = c_mass*(coords[7+d_i] + 0.5*(coords[4+d_i] + coords[10+d_i])) + h_mass*(coords[8+d_i]+coords[9+d_i] + 0.5*(coords[5+d_i] + coords[6+d_i] + coords[11+d_i] + coords[12+d_i]))
                pos3 = pos3 / tot_mass
                pos4 = c_mass*(coords[7+2*d_i] + 0.5*(coords[4+2*d_i] + coords[10+2*d_i])) + h_mass*(coords[8+2*d_i]+coords[9+2*d_i] + 0.5*(coords[5+2*d_i] + coords[6+2*d_i] + coords[11+2*d_i] + coords[12+2*d_i]))
                pos4 = pos4 / tot_mass


                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "L49C",
                    "B" + str(1),
                    n,
                    pos1[0],
                    pos1[1],
                    pos1[2],
                    0, 0, 0))
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "L49C",
                    "B" + str(2),
                    n+1,
                    pos2[0],
                    pos2[1],
                    pos2[2],
                    0, 0, 0))
          
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "L49C",
                    "B" + str(3),
                    n+2,
                    pos3[0],
                    pos3[1],
                    pos3[2],
                    0, 0, 0))
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                    res.index+1,
                    "L49C",
                    "B" + str(4),
                    n+3,
                    pos4[0],
                    pos4[1],
                    pos4[2],
                    0, 0, 0))
                    
                n = n+4

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

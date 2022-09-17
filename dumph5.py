
import h5py
import os

def h5py_printall(filename):

    # Open h5py file
    h5f = h5py.File(filename, 'r')

    # Get number of iterations
    niters = h5f['geom'].shape[0]

    # Get number atoms
    natoms = h5f['geom'].shape[1]

    # Iterate and print vectors
    """
    for key in h5f.keys():
        if key == 'poten' or key == 'kinen':
            continue
        print(key)
        data3d = h5f[key]
        for atomid in range(0, natoms):
            for it in range(0, niters):
                vec = data3d[it, atomid, :]
                print(('{:25.17f}'*3).format(vec[0], vec[1], vec[2]))
            print("")
    print("")
    """

    # Iterate and print energies
    poten = h5f['poten']
    kinen = h5f['kinen']
    print(('{:>25s}'*3).format('Potential', 'Kinetic', 'Total'))
    for it in range(0, niters):
        pot = poten[it]
        kin = kinen[it]
        tot = pot + kin
        print(('{:8.2f}'+'{:25.17f}'*3).format(h5f['time'][it], pot, kin, tot))
    print("")

    print(h5f.keys())
    print(h5f['time'][1])
    print(len(h5f['geom']))
    # Close
    h5f.close()



import sys
h5py_printall(sys.argv[1])




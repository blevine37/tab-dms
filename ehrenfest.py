########################################
# Ehrenfest code for TDCI
# Python 2, because TCController uses it
########################################

import tccontroller
import numpy as np
import shutil, os, subprocess, time
import h5py

########################################
# Job Template
########################################

if not os.path.exists("./templates"):
    os.makedirs("./templates")

TERACHEM = "/home/ateplukhin/Source/terachem/build/bin/" # terachem executable stored here
# make sure you include temppath and tempname in your job template!
#  those keywords are search and replaced
job_template_contents = "#!/bin/bash\n\
                         source /home/ateplukhin/.bashrc\n\
                         cd temppath\n\
                        "+TERACHEM+"terachem tempname.in > tempname.out\n"

JOB_TEMPLATE = "./templates/template.job"
f = open(JOB_TEMPLATE,'w')
f.write(job_template_contents)
f.close()

########################################
# TDCI TeraChem Template
########################################

nfields = 1                # number of distinct fields (generally for multichromatic floquet)
nstep = 12000               # number of timesteps
tdci_simulation_time = 12   # in femtoseconds
krylov_end = True           # Generate approximate eigenstates at end of calculation?
krylov_end_n = 64           # Number of steps to save wfn on to generate approx eigenstates with.
                            #   There will be 2*krylov_end_n approximate eigenstates returned.
krylov_end_interval = 80     # Number of steps between saved steps.
tdci_options = {
  "gpus"                 : "1 0",
  "timings"              : "yes",
  "precision"            : "double",
  "threall"              : "1.0e-20",
  "convthre"             : "1.0e-6",
  "basis"                : "sto-3g",
  "coordinates"          : "coords.xyz", # <-- don't change this
  "method"               : "hf",
  "run"                  : "tdci", # <-- don't change this
  "charge"               : "0",
  "spinmult"             : "1",
  "csf_basis"            : "no",
  "tdci_simulation_time" : str(tdci_simulation_time),
  "tdci_nstep"           : str(nstep),
  "tdci_eshift"          : "gs",
  "tdci_stepprint"       : "1",
  "tdci_nfields"         : str(nfields),
  "tdci_laser_freq"      : "3.018721363175e+15", # "2.5311296E+15",
  "tdci_photoneng"       : "0.45879422",         # "0.38467766",
  "tdci_fstrength"       : "1.0E+16",   # TODO: replace field generation params with file readin
  "tdci_fdirection"      : "x",
  "tdci_ftype"           : "cw",
  "tdci_corrfn_t"        : "p0",
  "tdci_write_field"     : "no",
  "tdci_floquet"         : "no",
  "tdci_floquet_photons" : "4",
  "casci"                : "yes",
  "ci_solver"            : "direct",
  "dcimaxiter"           : "300",
  "dciprintinfo"         : "yes",
  "dcipreconditioner"    : "orbenergy",
  "closed"               : "5",
  "active"               : "6",
  "cassinglets"          : "3",
  "castriplets"          : "0",
  "cascharges"           : "yes",
  "cas_ntos"             : "yes",
  "tdci_gradient"        : "yes",  # <-- don't change this
  "tdci_fieldfile0"      : "field0.bin",

  # Krylov subspace options
  "tdci_krylov_end"      : ("yes" if krylov_end else "no"),
  "tdci_krylov_end_n"    : krylov_end_n,
  "tdci_krylov_end_interval": krylov_end_interval,

  # These options will be removed on first step, don't change them.
  #"tdci_krylov_init"     : ("cn_krylov_init.bin" if krylov_end else "no"),
  "tdci_diabatize_orbs"  : "yes",
  "tdci_recn_readfile"   : "recn_init.bin",
  "tdci_imcn_readfile"   : "imcn_init.bin",
  "tdci_prevorbs_readfile": "PrevC.bin",
  "tdci_prevcoords_readfile": "PrevCoors.bin"
}
if krylov_end:
  tdci_options["tdci_krylov_init"] = "yes"
  tdci_options["tdci_krylovmo_readfile"] = "cn_krylov_init.bin"

TDCI_TEMPLATE = "./templates/tdci.in"
tccontroller.dict_to_file(tdci_options, TDCI_TEMPLATE)

########################################
# External field
########################################

# Field file should include values for half-steps, so the length of the array
#   should be 2*nsteps!

# Depending on the external field you want, you might have to write some
#   code here to generate the waveform, below is a CW tuned to ethylene
#   Function should accept np.arrays in units of AU time and return AU E-field units.
def f0_values(t):
  EPSILON_C = 0.00265316
  E_FIELD_AU = 5.142206707E+11
  HZtoAU = 2.418884E-17
  E_strength_Wm2 = 1.0E+16 # In W/m^2
  E_str = (np.sqrt(2.0*E_strength_Wm2 / EPSILON_C) )/E_FIELD_AU  # transform to au field units
  field_freq_hz = 3.444030610581e+15 # tuned to S0 <-> S1 for rabi flop example
  return E_str*np.sin(2.0*np.pi * field_freq_hz*HZtoAU * t)

FIELD_INFO = { "tdci_simulation_time": tdci_simulation_time,
               "nstep"               : nstep,
               "nfields"             : nfields,
               "f0"                  : f0_values,
               "krylov_end"          : krylov_end,
               "krylov_end_n"        : krylov_end_n
             }

########################################
# TCdata keys
########################################

"""
  Dictionary keys in TCdata:
           "recn"              - 1d array, number of determinants (ndets)
           "imcn"              - 1d array, ndets
           "eng"               - float, Energy of current wfn
           "grad"              - 2d array, Natoms x 3 dimensions.
           "recn_krylov"       - 1d array, 2*krylov_end_n
           "imcn_krylov"       - 1d array, 2*krylov_end_n
           "krylov_states"     - 2d array Approx Eigenstates in MO basis. 2*krylov_end_n x ndets
           "krylov_energies"   - 1d array of energies of each approx eigenstate
           "krylov_gradients"  - 3d array of approx eigenstate gradients, Napprox x Natoms x 3dim

"""

########################################
# Geometry read and write
########################################

def xyz_write(atoms, coords, filename):
    f = open(filename,'w')
    f.write(str(len(atoms))+'\n\n')
    for atom, coord in zip(atoms, coords):
        f.write(('{:>3s}'+'{:25.17f}'*3+'\n').format(atom, coord[0], coord[1], coord[2]))
    f.close()

def xyz_read(filename):
    f = open(filename,'r')
    n = int(f.readline())
    f.readline()
    atoms  = []
    coords = np.empty([n, 3])
    for i in range(0, n):
        fields = f.readline().split()
        if len(fields) != 4: break
        atoms.append(fields[0])
        coords[i][0] = float(fields[1])
        coords[i][1] = float(fields[2])
        coords[i][2] = float(fields[3])
    f.close()
    return (atoms, coords)

########################################
# TeraChem gradient-only calculation
# Returns accelerations
########################################

# Note 1: Ideally this function must be a part of tccontroller, because
#         tccontroller is responsible for interfacing with TeraChem.
#         Solution A: the function could be moved almost as-is into tccontroller.
#         Solution B: nextStep() method could allow propagation for 0 fs.
#         Solution C: nextStep() method could provide an option to change the order of propagation and grad.
#         In case of B and C, this function can be deleted.
# Note 2: Since this is a plain gradient TC calculation, the gradient is
#         returned as text. Ideally, we want it in binary form.
#         Solution B or C will solve this issue, because nextStep() returns grad in binary
#         Solution D: change TC source to output binary gradient, but this looks redundant,
#         because nextStep() already returns grad in binary.

def tc_grad(tc, atoms, masses, coords):

    # Print begin
    print("")
    print("###############################")
    print("##### TERACHEM GRAD BEGIN #####")
    print("###############################")
    print("")

    # Create directory
    graddir = "./grad"
    if not os.path.exists(graddir):
        os.makedirs(graddir)

    # Wrire job file
    shutil.copy(JOB_TEMPLATE, graddir + "/grad.job")
    tccontroller.search_replace_file(graddir + "/grad.job", "temppath", graddir)
    tccontroller.search_replace_file(graddir + "/grad.job", "tempname", "grad")

    # Write xyz file
    xyz_write(atoms, coords, graddir + "/grad.xyz")

    # Write TeraChem input file
    terachem_input = {
      "gpus"                 : "1 0",
      "timings"              : "yes",
      "precision"            : "double",
      "threall"              : "1.0e-20",
      "convthre"             : "1.0e-6",
      "basis"                : "sto-3g",
      "coordinates"          : "grad.xyz",
      "method"               : "hf",
      "run"                  : "gradient",
      "charge"               : "0",
      "spinmult"             : "1",
      "csf_basis"            : "no",
      ""                     : "",
      "casci"                : "yes",
      "ci_solver"            : "direct",
      "dcimaxiter"           : "300",
      "dciprintinfo"         : "yes",
      "dcipreconditioner"    : "orbenergy",
      ""                     : "",
      "closed"               : "5",
      "active"               : "6",
      "cassinglets"          : "3",
      "castriplets"          : "0",
      "cascharges"           : "yes",
      "cas_ntos"             : "yes",
      ""                     : "",
      "cassavevectors"       : "yes",
      ""                     : "",
      "casgradmult"          : "1 1 1",
      "casgradstate"         : "0 1 2"
    }
    tccontroller.dict_to_file(terachem_input, graddir + "/grad.in")

    # Start TeraChem
    p = subprocess.Popen("bash " + graddir + "/grad.job", shell=True)

    # Wait to finish
    while True:
        time.sleep(5)
        if p.poll() == None:
            print("Still running...\n")
        else:
            print("Done!\n")
            break

    # Get gradient
    n = len(atoms)
    grad = np.empty([n, 3])
    f = open(graddir+"/grad.out",'r')
    while not "Gradient units are Hartree/Bohr" in f.readline(): pass
    print(f.readline())
    print(f.readline())
    for i in range(0, n):
        fields = f.readline().split()
        if len(fields) != 3: break
        grad[i][0] = float(fields[0])
        grad[i][1] = float(fields[1])
        grad[i][2] = float(fields[2])
    f.close()

    # Get forces (Hartree/Bohr)
    forces = - grad
    print(forces)

    # Get accelerations
    accs = np.copy(forces)
    for a, mass in zip(accs, masses):
        a /= mass

    # Print end
    print("")
    print("###############################")
    print("###### TERACHEM GRAD END ######")
    print("###############################")
    print("")

    # Return accelerations
    return accs

########################################
# TeraChem propagation and gradient
# Returns accelerations
########################################

def tc_prop_and_grad(tc, atoms, masses, coords):

    # Print begin
    print("")
    print("###############################")
    print("######## TERACHEM BEGIN #######")
    print("###############################")
    print("")

    # Temprorary xyz filename
    xyzfilename = "temp.xyz"

    # Write xyz file
    xyz_write(atoms, coords, xyzfilename)

    # Call TeraChem
    TCdata = tc.nextstep(xyzfilename)
    print("")

    # Get forces (Hartree/Bohr)
    forces = - TCdata['grad']
    print(forces)

    # Get accelerations
    accs = np.copy(forces)
    for a, mass in zip(accs, masses):
        a /= mass

    # Print end
    print("")
    print("###############################")
    print("######### TERACHEM END ########")
    print("###############################")
    print("")

    # Return accelerations
    return accs

########################################
# Masses initialization
########################################

def getmasses(atoms):

    # Mass data
    # Source: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
    massdata = {}
    massdata['H'] =  1.00782503223
    massdata['C'] = 12.00000000000
    massdata['N'] = 14.00307400443
    massdata['O'] = 15.99491461957
    massdata['S'] = 31.9720711744
    
    # Build nympy array of masses
    natoms = len(atoms)
    masses = np.empty([natoms])
    for i in range(0, natoms):
        masses[i] = massdata[atoms[i]]

    # Convert from Unified atomic mass unit (Dalton) to atomic unit of mass (au, m_e)
    # Source: https://en.wikipedia.org/wiki/Dalton_(unit)
    masses *= 1822.888486209

    # Return masses
    return masses

########################################
# h5py
########################################

def h5py_update(geom, vels, accs):

    # Get array dimension
    n = geom.shape[0]

    # Open h5py file
    h5f = h5py.File('data.hdf5', 'a')

    # Create datasets
    if len(list(h5f.keys())) == 0 :
        print('Creating datasets')
        h5f.create_dataset('geom', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
        h5f.create_dataset('vels', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
        h5f.create_dataset('accs', (0, n, 3), maxshape=(None, n, 3), dtype='float64')

    # Resize
    for key in h5f.keys():
        dset = h5f[key]
        dset.resize(dset.len() + 1, axis=0)

    # Store data
    h5f['geom'][-1] = geom
    h5f['vels'][-1] = vels
    h5f['accs'][-1] = accs

    # Close
    h5f.close()

def h5py_printall():

    # Open h5py file
    h5f = h5py.File('data.hdf5', 'r')

    # Iterate and print
    for key in h5f.keys():
        print(key)
        for vecs in h5f[key]:
            for vec in vecs:
                print(('{:25.17f}'*3).format(vec[0], vec[1], vec[2]))
            print("")

    # Close
    h5f.close()


########################################
# Main function
# All calculations are in atomic units (au)
# Source: https://en.wikipedia.org/wiki/Hartree_atomic_units
########################################

# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903

# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17

# Job directory
JOBDIR = "./"

# Print header
print("TDCI + TAB-DMS code")
print("Ehrenfest version")
print("")

# Initialize TC controller
print("Initializing tccontroller\n")
tc = tccontroller.tccontroller(JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE, FIELD_INFO, False)

# Geometry file name
geomfilename = "ethylene.xyz"
print("Geometry file: " + geomfilename)

# Read geometry file
print("Reading geometry file")
atoms, cs_curr = xyz_read(geomfilename)

# Get masses
print("Getting masses")
masses = getmasses(atoms)

# Convert coordinates from Angstroms to au
xs_curr = cs_curr / bohrtoangs

# Initialize velocities
print("Initializing velocities")
vs_curr = np.zeros([len(atoms), 3])

# Initialize accelerations
print("Initializing accelerations")
as_curr = tc_grad(tc, atoms, masses, cs_curr)

# Get time step in au (from femtoseconds)
delta = tdci_simulation_time * 1e-15 / autimetosec
print("Time step in fs: " + str(tdci_simulation_time))
print("Time step in au: " + str(delta))
print("")

# Store inital state in HDF5
print("Storing intial state in HDF5")
h5py_update(xs_curr, vs_curr, as_curr)
print("")

# Main dynamics loop
for it in range(1, 2):

    # Log iteration start
    print("Iteration " + str(it).zfill(4) + " started")

    # Calculate next geometry
    print("Calculating next geometry")
    xs_next = xs_curr + vs_curr * delta + as_curr * delta**2 / 2

    # Calculate next accelerations
    print("Calculating next accelerations")
    as_next = tc_prop_and_grad(tc, atoms, masses, xs_next * bohrtoangs)

    # Calculate next velocities
    print("Calculating next velocities")
    vs_next = vs_curr + (as_curr + as_next) * delta / 2

    # Update HDF5
    print("Updating HDF5")
    h5py_update(xs_next, vs_next, as_next)

    # Update current
    xs_curr = xs_next
    vs_curr = vs_next
    as_curr = as_next

    # Print results
    print("Iteration " + str(it).zfill(4) + " finished")
    print("")
    print("")

# Write final geometry
print("Writing final geometry")
xyz_write(atoms, xs_next * bohrtoangs, "final.xyz")

# Write final velocities
print("Writing final velocities")
xyz_write(atoms, vs_next, "final.vel")

# Write final accelerations
print("Writing final accelerations")
xyz_write(atoms, as_next, "final.acc")

# Print HDF5 contents
print("Printing HDF5 contents")
h5py_printall()

print("Finished!")


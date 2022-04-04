########################################
# Ehrenfest code for TDCI
# Python 2, because TCController uses it
########################################

import tccontroller
import numpy as np
import shutil
import sys

########################################
# Job Template
########################################

TERACHEM = "/home/ateplukhin/Source/terachem/build/bin/" # terachem executable stored here
# make sure you include temppath and tempname in your job template!
#  those keywords are search and replaced
job_template_contents = "#!/bin/bash\n\
                         source /home/ateplukhin/.bashrc\n\
                         cd temppath\n\
                        "+TERACHEM+"terachem tempname.in > tempname.out\n"

JOB_TEMPLATE = "/home/ateplukhin/Source/tccontroller/templates/template.job"
f = open(JOB_TEMPLATE,'w')
f.write(job_template_contents)
f.close()

########################################
# TDCI TeraChem Template
########################################

nfields = 1                # number of distinct fields (generally for multichromatic floquet)
nstep = 12000               # number of timesteps
tdci_simulation_time = 12   # in femtoseconds
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
  "tdci_laser_freq"      : "2.5311296E+15",  
  "tdci_photoneng"       : "0.38467766",
  "tdci_fstrength"       : "1.0E+16",   # TODO: replace field generation params with file readin
  "tdci_fdirection"      : "x",
  "tdci_ftype"           : "cw",
  "tdci_corrfn_t"        : "p0",
  "tdci_write_field"     : "no",
  "tdci_floquet"         : "no",
  "tdci_floquet_photons" : "3",
  "casci"                : "yes",
  "ci_solver"            : "direct",
  "dcimaxiter"           : "300",
  "dciprintinfo"         : "yes",
  "dcipreconditioner"    : "orbenergy",
  "closed"               : "7",
  "active"               : "2",
  "cassinglets"          : "3",
  "castriplets"          : "0",
  "cascharges"           : "yes",
  "cas_ntos"             : "yes",
  "tdci_gradient"        : "yes",  # <-- don't change this
  "tdci_fieldfile0"      : "field0.csv",

  # Krylov subspace options
  #"tdci_krylov_end"      : "yes",
  #"tdci_krylov_end_n"    : "25",
  #"tdci_krylov_end_interval": "5",

  # These options will be removed on first step, don't change them.
  "tdci_diabatize_orbs"  : "yes",
  "tdci_recn_readfile"   : "recn_init.bin",
  "tdci_imcn_readfile"   : "imcn_init.bin",
  "tdci_prevorbs_readfile": "PrevC.bin",
  "tdci_prevcoords_readfile": "PrevCoors.bin"
}

TDCI_TEMPLATE = "/home/ateplukhin/Source/tccontroller/templates/tdci.in"
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
               "f0"                  : f0_values
             }

########################################
# TCdata keys
########################################

"""
  Dictionary keys in TCdata:
           "recn"              - 1d array, number of determinants
           "imcn"              - 1d array, number of determinants
           "eng"               - float, Energy of current wfn
           "grad"              - 2d array, Natoms x 3 dimensions.
           "krylov_states"     - 2d array of CI vectors of each approx eigenstate
           "krylov_energies"   - 1d array of energies of each approx eigenstate
           "krylov_gradients"  - 3d array of approx eigenstate gradients, Napprox x Natoms x 3dim

"""

########################################
# Geometry read and write
########################################

def xyz_write(atoms, coords, filename):
    f = open(filename,'w')
    f.write(str(len(atoms))+'\n\n')
    for atom, coord in zip(atoms,coords):
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
# Vectors read and write
########################################

def vecs_write(vecs, filename):
    f = open(filename,'w')
    f.write(str(len(vecs))+'\n\n')
    for vec in vecs:
        f.write(('{:25.17f}'*3+'\n').format(vec[0], vec[1], vec[2]))
    f.close()

def vecs_read(filename):
    f = open(filename,'r')
    n = int(f.readline())
    f.readline()
    vecs = np.empty([n, 3])
    for i in range(0, n):
        fields = f.readline().split()
        if len(fields) != 3: break
        vecs[i][0] = float(fields[0])
        vecs[i][1] = float(fields[1])
        vecs[i][2] = float(fields[2])
    f.close()
    return vecs

########################################
# Accelerations
########################################

def getaccs(tc, masses, xyzfilename):

    # Print begin
    print("")
    print("###############################")
    print("##### ACCELERATIONS BEGIN #####")
    print("###############################")
    print("")

    # Call TeraChem
    TCdata = tc.nextstep(xyzfilename)
    print("")

    # Get forces (Hartree/Bohr)
    forces = TCdata['grad']
    print(forces)

    # Get accelerations
    accs = np.copy(forces)
    for a, mass in zip(accs, masses):
        a /= mass

    # Print end
    print("")
    print("###############################")
    print("###### ACCELERATIONS END ######")
    print("###############################")
    print("")

    # Return accelerations
    return accs

########################################
# Masses initialization
########################################

def getmasses(geomfilename):

    # Mass data
    # Source: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
    massdata = {}
    massdata['H'] =  1.00782503223
    massdata['C'] = 12.00000000000
    massdata['N'] = 14.00307400443
    massdata['O'] = 15.99491461957
    massdata['S'] = 31.9720711744
    
    # Get atoms only
    atoms, _ = xyz_read(geomfilename)

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
# Files initialization, iteration 0
########################################

def initgeomfile(xyzfilename, geomfilename):
    shutil.copyfile(xyzfilename, geomfilename)

def initvelsfile(xyzfilename, velfilename):
    f = open(xyzfilename,'r')
    natoms = int(f.readline())
    f.close()
    vels = np.zeros([natoms, 3])
    vecs_write(vels, velfilename)

def initaccsfile(xyzfilename, accfilename, tc, masses):
    accs = getaccs(tc, masses, xyzfilename)
    vecs_write(accs, accfilename)

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
JOBDIR = "/home/ateplukhin/jobs/ehrenfest/"

# Print header
print("TDCI + TAB-DMS code")
print("Ehrenfest version")
print("")

# TC controller initialization
print("Initializing tccontroller\n")
tc = tccontroller.tccontroller(JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE, FIELD_INFO, False)

# Geometry file name
molname      = "ethylene"
geomfilename = molname + ".xyz"
print("Geometry file: " + geomfilename)

# Get masses
print("Getting masses")
masses = getmasses(geomfilename)

# Iteration 0: geometry file
print("Initializing geometry file")
initgeomfile(geomfilename, molname + str(0).zfill(4) + ".xyz")

# Iteration 0: velocities file
print("Initializing velocities file")
initvelsfile(geomfilename, molname + str(0).zfill(4) + ".vel")

# Iteration 0: accelerations file
print("Initializing accelerations file")
initaccsfile(geomfilename, molname + str(0).zfill(4) + ".acc", tc, masses)

# Get time step in au (from femtoseconds)
delta = tdci_simulation_time * 1e-15 / autimetosec
print("Time step in fs: " + str(tdci_simulation_time))
print("Time step in au: " + str(delta))
print("")

# Main dynamics loop
for it in range(0, 5):

    # Log iteration start
    print("Iteration " + str(it).zfill(4) + " started")

    # Current filenames (to read from)
    geomfilename_curr = JOBDIR + molname + str(it).zfill(4)   + ".xyz"
    velsfilename_curr = JOBDIR + molname + str(it).zfill(4)   + ".vel"
    accsfilename_curr = JOBDIR + molname + str(it).zfill(4)   + ".acc"

    # Next filenames (to write to)
    geomfilename_next = JOBDIR + molname + str(it+1).zfill(4) + ".xyz"
    velsfilename_next = JOBDIR + molname + str(it+1).zfill(4) + ".vel"
    accsfilename_next = JOBDIR + molname + str(it+1).zfill(4) + ".acc"

    # Read current geometry
    print("Reading current geometry")
    atoms, cs_curr = xyz_read(geomfilename_curr)

    # Read current velocities
    print("Reading current velocities")
    vs_curr = vecs_read(velsfilename_curr)

    # Read current accelerations
    print("Reading current accelerations")
    as_curr = vecs_read(accsfilename_curr)

    # Convert current coordinates from Angstroms to au
    xs_curr = cs_curr / bohrtoangs

    # Calculate next geometry
    print("Calculating next geometry")
    xs_next = xs_curr + vs_curr * delta + as_curr * delta**2 / 2

    # Convert current coordinates from au to Angstroms
    cs_next = xs_next * bohrtoangs

    # Write next geometry
    print("Writing next geometry")
    xyz_write(atoms, cs_next, geomfilename_next)

    # Calculate next accelerations
    print("Calculating next accelerations")
    as_next = getaccs(tc, masses, geomfilename_next)

    # Write next accelerations
    print("Writing next accelerations")
    vecs_write(as_next, accsfilename_next)

    # Calculate next velocities
    print("Calculating next velocities")
    vs_next = vs_curr + (as_curr + as_next) * delta / 2

    # Write next velocities
    print("Writing next velocities")
    vecs_write(vs_next, velsfilename_next)

    # Print results
    print("Iteration " + str(it).zfill(4) + " finished")
    print("")
    print("")

print("Finished!")


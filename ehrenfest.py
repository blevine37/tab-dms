########################################
# Ehrenfest code for TDCI
# Python 2, because TCController uses it
#
# All calculations are in atomic units (au)
# Source: https://en.wikipedia.org/wiki/Hartree_atomic_units
########################################

import tccontroller
import numpy as np
import shutil, os, subprocess, time
import h5py

# to install h5py:
# $ apt install libhdf5-dev
# $ HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
# $ pip2 install h5py

########################################
# Job Template
########################################

if not os.path.exists("./templates"):
  os.makedirs("./templates")

TERACHEM = "/home/adurden/terachem/build/bin/" # directory containing terachem executable

# Job Template: Used to make a shell script that executes terachem 
#   make sure you include temppath and tempname in your job template!
#   those keywords are search and replaced
job_template_contents = "#!/bin/bash\n\
                         source /home/adurden/.bashrc\n\
                         cd temppath\n\
                        "+TERACHEM+"terachem tempname.in > tempname.out\n"

JOB_TEMPLATE = "./templates/template.job"
f = open(JOB_TEMPLATE,'w')
f.write(job_template_contents)
f.close()

########################################
# Times
########################################

# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17

# Dynamics time step in atomic units
delta = 10

# TDCI simulation time in femtoseconds
tdci_simulation_time = delta * autimetosec * 1e15 # fs/s

# TDCI number of time steps
nstep = 1000

########################################
# TDCI TeraChem Template
########################################

nfields = 1                    # number of distinct fields (generally for multichromatic floquet)
krylov_end = True              # Generate approximate eigenstates at end of calculation?
krylov_end_n = 6              # Number of steps to save wfn on to generate approx eigenstates with.
                               # There will be 2*krylov_end_n approximate eigenstates returned.
krylov_end_interval = 20       # Number of steps between saved steps.
TDCI_TEMPLATE = {
  "gpus"                 : "1 0",
  "precision"            : "double",
  "threall"              : "1.0e-20",
  "convthre"             : "1.0e-6",
  "basis"                : "3-21g",
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
  "tdci_fstrength"       : "0.0E+16",
  "tdci_fdirection"      : "x",
  "tdci_ftype"           : "cw",
  "tdci_corrfn_t"        : "p0",
  "tdci_write_field"     : "no",
  "tdci_floquet"         : "no",
  "tdci_floquet_photons" : "4",
  "fon"                  : "yes",
  "fon_method"           : "gaussian",
  "fon_temperature"      : "0.25",
  "fon_mix"              : "no",
  #"casscf"               : "yes",
  "casci"                : "yes",
  "ci_solver"            : "direct",
  "dcimaxiter"           : "300",
  "dciprintinfo"         : "yes",
  "dcipreconditioner"    : "orbenergy",
  "closed"               : "6",
  "active"               : "4",
  "cassinglets"          : "8",
  "castriplets"          : "0",
  "cascharges"           : "yes",
  "cas_ntos"             : "yes",
  "tdci_gradient"        : "yes",  # <-- don't change this
  "tdci_gradient_half"   : "yes",  # <-- don't change this
  "tdci_fieldfile0"      : "field0.bin",

  # orbital options
  #"fon"                  : "yes",
  #"fon_method"           : "gaussian",
  #"fon_temperature"      : "0.25",

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
  # CI vector transformation from Krylov basis to determinant basis
  # if Yes, we expect nextstep arguments to be in Krylov basis.
  TDCI_TEMPLATE["tdci_krylov_init"] = "no"
  TDCI_TEMPLATE["tdci_krylovmo_readfile"] = "cn_krylov_init.bin"


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
#  return E_str*np.sin(2.0*np.pi * field_freq_hz*HZtoAU * t)
  return 0 * t   # Field is temporarily off

FIELD_INFO = { "tdci_simulation_time": tdci_simulation_time,
               "nstep"               : nstep,
               "nfields"             : nfields,
               "f0"                  : f0_values,
               "krylov_end"          : krylov_end,
               "krylov_end_n"        : krylov_end_n
             }


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

def tc_grad(tc, atoms, masses, coords, ReCn=None, ImCn=None, return_states=False):

  # Print begin
  print("")
  print("###############################")
  print("##### TERACHEM GRAD BEGIN #####")
  print("###############################")
  print("")
  """
    Dictionary keys in grad output:
	     "eng"               - float, Energy of current wfn
	     "grad"              - 2d array, Natoms x 3 dimensions.
	     "states"            - 2d array, Nstates x ndets.
	     "states_eng"        - 2d array, Nstates.

    INPUT:
	      xyz                - string, path of xyz file.
	      ReCn (optional)    - Real component of CI vector. If none, ground state is used. 
	      ImCn (optional)    - Imaginary component of CI vector.
  """

  
  if (ReCn is None):
    grad_data = tc.grad(atoms) # Terachem, do your thing!
    ReCn = grad_data["states"][0]
  else:
    grad_data = tc.grad(atoms, ReCn, ImCn) # Initial Conditions
  grad = grad_data["grad"]
  en = grad_data["eng"]

  norm = np.sum( np.array(ReCn)**2 )
  if ImCn is not None:
    norm += np.sum( np.array(ImCn)**2 )

  print("gradnorm: "+str(norm))
  # Get forces (Hartree/Bohr)
  forces = - (grad/norm)
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

  # Return accelerations and energy
  if return_states:
    return (accs, en, grad_data["states"])
  else:
    return (accs, en)

########################################
# TeraChem propagation and gradient
# Returns accelerations
########################################

def getAccel(grad):

  # Get forces (Hartree/Bohr)
  accs = -grad
  # Get accelerations
  for a, mass in zip(accs, masses):
    a /= mass


  # Return accelerations
  return accs


def tc_prop_and_grad(tc, atoms, masses, coords, ReCn=None, ImCn=None):

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
  """
    Dictionary keys in nextstep output:
	     "recn"              - 1d array, number of determinants (ndets)
	     "imcn"              - 1d array, ndets
	     "eng"               - float, Energy of current wfn
	     "grad"              - 2d array, Natoms x 3 dimensions.
	     "grad_half"         - 2d array, Natoms x 3 dimensions.
	     "recn_krylov"       - 1d array, 2*krylov_end_n
	     "imcn_krylov"       - 1d array, 2*krylov_end_n
	     "krylov_states"     - 2d array Approx Eigenstates in MO basis. 2*krylov_end_n x ndets
	     "krylov_energies"   - 1d array of energies of each approx eigenstate
	     "krylov_gradients"  - 3d array of approx eigenstate gradients, Napprox x Natoms x 3dim

    INPUT:
	      xyz                - string, path of xyz file.
	      ReCn (optional)    - Real component of CI vector. If none, ground state is used. 
	      ImCn (optional)    - Imaginary component of CI vector.
  """
  if (ReCn is None):
    TCdata = tc.nextstep(xyzfilename)
  else: # For inital conditions
    TCdata = tc.nextstep(xyzfilename, ReCn, ImCn)
  print("")

  # Get forces (Hartree/Bohr)
  forces = - TCdata['grad']
  print(forces)

  # Get accelerations
  accs = np.copy(forces)
  for a, mass in zip(accs, masses):
    a /= mass

  # Get energy
  en = float(TCdata['eng'])

  # Print end
  print("")
  print("###############################")
  print("######### TERACHEM END ########")
  print("###############################")
  print("")

  # Return accelerations
  return (accs, en)

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
  
  # Build numpy array of masses
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
# Kinetic energy calculation
########################################

def kincalc(masses, vs):

  # Initialize energy variable
  ke = 0

  # Iterate over masses
  for m, v in zip(masses, vs):
    ke += m * v.dot(v) / 2

  # Return energy
  return ke

########################################
# h5py
########################################
# time in attoseconds
def h5py_update(geom, vels, accs, poten, kinen, time):

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
    h5f.create_dataset('poten', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('kinen', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('time', (0,), maxshape=(None,), dtype='float64')

  # Resize
  for key in h5f.keys():
    dset = h5f[key]
    dset.resize(dset.len() + 1, axis=0)

  # Store data
  h5f['geom'][-1] = geom
  h5f['vels'][-1] = vels
  h5f['accs'][-1] = accs
  h5f['poten'][-1] = poten
  h5f['kinen'][-1] = kinen
  h5f['time'][-1] = time

  # Close
  h5f.close()

def h5py_printall():

  # Open h5py file
  h5f = h5py.File('data.hdf5', 'r')

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
    print(('{:25.17f}'*3).format(pot, kin, tot))
  print("")

  # Close
  h5f.close()

# Need to make sure hdf5 and job directories from previous runs don't get in the way
def clean_files():
  if os.path.exists("oldrun/"):
    shutil.rmtree("oldrun/")
  os.makedirs("oldrun/")
  if os.path.exists("electronic"):
    shutil.move("electronic", "oldrun/electronic")
  if os.path.exists("data.hdf5"):
    shutil.move("data.hdf5", "oldrun/data.hdf5")


########################################
# Main function
########################################

clean_files()
l = tccontroller.logger()
# logprint(string) will write a timestamped string to the log, and to STDOUT
logprint = l.logprint


# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903

# Job directory
JOBDIR = "./"

# Print header
logprint("TDCI + TAB-DMS code")
logprint("Ehrenfest version")
logprint("")

# Time steps
logprint("Propagation time step in au: " + str(delta))
logprint("TDCI simulation half time step in fs: " + str(tdci_simulation_time))
logprint("")

# Initialize TC controller
logprint("Initializing tccontroller\n")
tc = tccontroller.tccontroller(JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE,
                               FIELD_INFO, logger=l, SCHEDULER=False)

# Geometry file name
geomfilename = "ethylene.xyz"
logprint("Geometry file: " + geomfilename)

# Read geometry file
logprint("Reading geometry file")
atoms, cs_curr = xyz_read(geomfilename)

# Initial time
time = 0

# Get masses
logprint("Getting masses")
masses = getmasses(atoms)

# Convert coordinates from Angstroms to au
xs_curr = cs_curr / bohrtoangs

# Initialize velocities
logprint("Initializing velocities")
vs_curr = np.zeros([len(atoms), 3])

# AD: TODO: some optimization can be done here
#     right now tc.grad expects an arbitrary CI vector (or defaults to S0)
#     That means if we want S1 initial conditions, we have to call TC to solve states
#     and THEN feed one of those states back into tc.grad... not efficient.
#     If we add some more params to terachem, we can do this in one call

# Get states for inital conditions
gradout = tc.grad(geomfilename) 
#initial_recn = gradout["states"][1]
initial_recn = gradout["states"][0]

# Initialize accelerations
logprint("Initializing accelerations")
as_curr, pe_curr = tc_grad(tc, geomfilename, masses, cs_curr, ReCn=initial_recn)

# Calculate initial kinetic energy
logprint("Calculating initial kinetic energy")
ke_curr = kincalc(masses, vs_curr)

# Store inital state in HDF5
logprint("Storing intial state in HDF5")
h5py_update(xs_curr, vs_curr, as_curr, pe_curr, ke_curr, time)
logprint("")

# Main dynamics loop
for it in range(1, 10000):

  # Log iteration start
  logprint("Iteration " + str(it).zfill(4) + " started")

  # Calculate next geometry
  logprint("Calculating next geometry")
  xs_next = xs_curr + vs_curr * delta + as_curr * delta**2 / 2

  # Propagate electronic wavefunction to next time step
  logprint("Propagating electronic wavefunction to next time step using next coordinates")
  TCdata = None
  xyz_write(atoms, xs_next*bohrtoangs, "temp.xyz")
  if (it == 1):
    TCdata = tc.nextstep("temp.xyz", ReCn=initial_recn, ImCn=None)
  else:
    TCdata = tc.nextstep("temp.xyz")

  # Calculate acceleration from gradient at half time-step.
  as_next = getAccel(TCdata["grad_half"])
  pe_next = float(TCdata['eng'])


  # Calculate next velocities
  logprint("Calculating next velocities")
  vs_next = vs_curr + (as_curr + as_next) * delta / 2

  # Calculate next kinetic energy
  logprint("Calculating next kinetic energy")
  ke_next = kincalc(masses, vs_next)

  time += delta * autimetosec * 1e+18 # Time in Attoseconds

  # Update HDF5
  logprint("Updating HDF5")
  h5py_update(xs_next, vs_next, as_next, pe_next, ke_next, time)

  # Print HDF5 contents
  logprint("Printing HDF5 contents")
  h5py_printall()

  # Update current
  xs_curr = xs_next
  vs_curr = vs_next
  as_curr = as_next
  pe_curr = pe_next
  ke_curr = ke_next

  # Print results
  logprint("Iteration " + str(it).zfill(4) + " finished")
  logprint("")
  logprint("")

# Write final geometry
logprint("Writing final geometry")
xyz_write(atoms, xs_next * bohrtoangs, "final.xyz")

# Write final velocities
logprint("Writing final velocities")
xyz_write(atoms, vs_next, "final.vel")

# Write final accelerations
logprint("Writing final accelerations")
xyz_write(atoms, as_next, "final.acc")

# Print HDF5 contents
logprint("Printing HDF5 contents")
h5py_printall()

logprint("Finished!")


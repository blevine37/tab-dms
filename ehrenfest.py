########################################
# Ehrenfest code for TDCI
#
# All calculations are in atomic units (au)
# Source: https://en.wikipedia.org/wiki/Hartree_atomic_units
########################################

import tccontroller
import numpy as np
import shutil, os, subprocess, time
import h5py
import config

# to install h5py:
# $ apt install libhdf5-dev
# $ HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
# $ pip2 install h5py

# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903
# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17



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
  ke = 0 # Initialize energy variable
  # Iterate over masses
  for m, v in zip(masses, vs):
    ke += m * v.dot(v) / 2
  return ke

########################################
# h5py
########################################
# time in attoseconds
def h5py_update(geom, vels, accs, poten, kinen, Time, TCdata=None):

  # Get array dimension
  n = geom.shape[0]
  ndets = 0
  #if ReCn is not None:
  #  ndets = len(ReCn)

  # Open h5py file
  h5f = h5py.File('data.hdf5', 'a')

  # Create datasets
  if len(list(h5f.keys())) == 0 :
    print('Creating datasets')
    # why doesnt this work ;-;
    #h5f.create_dataset('atoms', (1,n), maxshape=(1,n), dtype=h5py.string_dtype('utf-8',30))
    #h5f['atoms'] = atoms
    h5f.create_dataset('recn', (0, ndets), maxshape=(None, ndets), dtype='float64')
    h5f.create_dataset('imcn', (0, ndets), maxshape=(None, ndets), dtype='float64')

    str_dtype = h5py.special_dtype(vlen=str)
    h5f.create_dataset('tdci_dir', (100,), maxshape=(None,), dtype=str_dtype)

    h5f.create_dataset('geom', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('vels', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('accs', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('poten', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('kinen', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('time', (0,), maxshape=(None,), dtype='float64')

  # Resize
  for key in ['geom','vels','accs','poten','kinen','time', 'tdci_dir', 'recn', 'imcn']:
    dset = h5f[key]
    dset.resize(dset.len() + 1, axis=0)

  # Store data
  h5f['geom'][-1] = geom
  h5f['vels'][-1] = vels
  h5f['accs'][-1] = accs
  h5f['poten'][-1] = poten
  h5f['kinen'][-1] = kinen
  h5f['time'][-1] = Time
  if TCdata is None:
    h5f['recn'][-1] = np.zeros(ndets)
    h5f['imcn'][-1] = np.zeros(ndets)
    h5f['tdci_dir'][-1] = ""
  else:
    h5f['recn'][-1] = TCdata["recn"]
    h5f['imcn'][-1] = TCdata["imcn"]
    h5f['tdci_dir'][-1] = TCdata["tdci_dir"]
  # Close
  h5f.close()
  # had an error earlier when the h5 file was opened again before it was finished closing
  # think that might be due to filesystem lag or something idk so here's a sleep
  time.sleep(2)

def h5py_printall():

  time.sleep(2)
  # Open h5py file
  h5f = h5py.File('data.hdf5', 'r')

  # Get number of iterations
  niters = h5f['geom'].shape[0]

  # Get number atoms
  natoms = h5f['geom'].shape[1]

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




class Ehrenfest:
  def __init__(self, delta, logprint, tc):
    self.delta = delta
    self.logprint = logprint
    self.tc = tc
    self.masses = getmasses(atoms)

  # Accepts a gradient from TCdata, returns (normalized) acceleration.
  def getAccel(self, grad, ReCn, ImCn):
    norm = np.sum( np.array(ReCn)**2 )
    if ImCn is not None:
      norm += np.sum( np.array(ImCn)**2 )
    # Get forces (Hartree/Bohr)
    accs = -(grad/norm)
    # Get accelerations
    for a, mass in zip(accs, self.masses):
      a /= mass
    # Return accelerations
    return accs

  def savestate(self, x, v, a, t, pe, TCdata):
    # Update HDF5
    self.logprint("Updating HDF5")
    ke = kincalc(self.masses, v)
    # TODO: save the tdci directory and other stuff from TCdata
    h5py_update(x, v, a, pe, ke, t, TCdata)
    # Print HDF5 contents
    self.logprint("Printing HDF5 contents")
    h5py_printall()

  def loadstate(self):
    h5f = h5py.File('data.hd5f','r')
    fields = ['atoms', 'geom','vels','accs','poten','kinen','time']
    for field in fields:
      if field not in h5f.keys():
	print("h5 file uninitialized")
	os.kill()
    x_curr = h5py['geom'][-1]
    v_curr = h5py['geom'][-1]
    a_curr = h5py['geom'][-1]
    pe_curr = h5py['geom'][-1]
    ke_curr = h5py['geom'][-1]
    t = h5py['time'][-1]
    recn = h5py['recn']
    imcn = h5py['imcn']
    self.tc.N = len(h5py['geom'])
    return x_curr, v_curr, a_curr, t, pe_curr, recn, imcn
    
  # Prepare initial state and start propagation.
  def run_ehrenfest(self):
    x, v, a, pe, recn, imcn = None, None, None, None, None, None
    t = 0
    if config.RESTART:
      x, v, a, t, pe, recn, imcn = self.loadstate()
    else:
      clean_files() # Clean job directory
      geomfilename = config.xyzpath # .xyz filename in job directory
      self.atoms, x = xyz_read(geomfilename)
      self.masses = getmasses(self.atoms)
      v = np.zeros([len(atoms), 3]) # Initial velocity
      # Call Terachem to Calculate states
      gradout = self.tc.grad(x) # x should already be in angstroms here
      recn = gradout["states"][config.initial_electronic_state]
      imcn = np.zeros(len(recn))
      # Run ANOTHER grad calculation with our initial CI vector
      TCdata_init = self.tc.grad(x, recn, imcn)
      pe = TCdata_init["eng"]
      a = self.getAccel(TCdata_init["grad"], recn, imcn)
      print("a init:"+str(a))
      #a = np.zeros([len(atoms),3]) # ignore gradient for one step...
      x = x / bohrtoangs # Initial geometry in Bohr
    self.propagate(x, v, a, t, pe, recn, imcn)
    

  # Does Ehrenfest propagation loop
  def propagate(self, x_init, v_init, a_init, t_init, pe_init, ReCn_init, ImCn_init=None):
    t = t_init
    x, v, a, pe, ReCn, ImCn = x_init, v_init, a_init, pe_init, ReCn_init, ImCn_init
    TCdata = None
    self.savestate(x, v, a, t, pe, TCdata)
    for it in range(0,99999999): # go forever! :D
      t += delta * autimetosec * 1e+18 # Time in Attoseconds
      x, v, a, TCdata = self.step(x, v, a, ReCn=ReCn, ImCn=ImCn) # Do propagation step
      ReCn, ImCn = None, None # Defaults to =TCdata["recn"],TCdata["imcn"] from prevstep.
      self.savestate(x, v, a, t, TCdata["eng"], TCdata)
      logprint("Iteration " + str(it).zfill(4) + " finished")
      
    
  # Accepts current state, runs TDCI, calculates next state.
  def step(self, x, v, a, ReCn=None, ImCn=None):
    logprint = self.logprint
    x_next = x + v*self.delta + (a*self.delta**2)/2.
    TCdata = self.tc.nextstep(x_next*bohrtoangs, ReCn=ReCn, ImCn=ImCn) # Do TDCI! \(^0^)/
    a_next = self.getAccel(TCdata["grad_half"], TCdata["recn"], TCdata["imcn"])
    v_next = v + (a + a_next) * self.delta / 2
    return x_next, v_next, a_next, TCdata
    



########################################
# Job Template
########################################

if not os.path.exists("./templates"):
  os.makedirs("./templates")


JOB_TEMPLATE = "./templates/template.job"
f = open(JOB_TEMPLATE,'w')
f.write(config.job_template_contents)
f.close()

########################################
# Times
########################################


# Dynamics time step in atomic units
delta = config.TIMESTEP_AU

# TDCI simulation time in femtoseconds
tdci_simulation_time = (delta/1.0) * autimetosec * 1e15 # fs/s

# Explicitly aliasing the config variables to keep track of them
RESTART = config.RESTART
TERACHEM = config.TERACHEM
TIMESTEP_AU = config.TIMESTEP_AU
nstep = config.NSTEPS_TDCI
nfields = config.nfields
krylov_end = config.krylov_end
krylov_end_n = config.krylov_end_n
krylov_end_interval = config.krylov_end_interval
f0_values = config.f0_values

TDCI_TEMPLATE = config.TDCI_TEMPLATE
TDCI_TEMPLATE["coordinates"] = "coords.xyz"
TDCI_TEMPLATE["run"] = "tdci"
TDCI_TEMPLATE["tdci_simulation_time"] = str(tdci_simulation_time)
TDCI_TEMPLATE["tdci_nstep"] = str(nstep)
TDCI_TEMPLATE["tdci_nfields"] = str(nfields)
TDCI_TEMPLATE["tdci_gradient"] = "yes"
TDCI_TEMPLATE["tdci_grad_init"] = "yes"
TDCI_TEMPLATE["tdci_grad_half"] = "yes"
TDCI_TEMPLATE["tdci_fieldfile0"] = "field0.bin"
# Krylov subspace options
TDCI_TEMPLATE["tdci_krylov_end"] = ("yes" if krylov_end else "no")
TDCI_TEMPLATE["tdci_krylov_end_n"] = krylov_end_n
TDCI_TEMPLATE["tdci_krylov_end_interval"] = krylov_end_interval
# These options will be removed on first step
TDCI_TEMPLATE["tdci_diabatize_orbs"]  = "yes"
TDCI_TEMPLATE["tdci_recn_readfile"]   = "recn_init.bin"
TDCI_TEMPLATE["tdci_imcn_readfile"]  = "imcn_init.bin"
TDCI_TEMPLATE["tdci_prevorbs_readfile"] = "PrevC.bin"
TDCI_TEMPLATE["tdci_prevcoords_readfile"] = "PrevCoors.bin"

atoms, xyz = xyz_read(config.xyzpath)

FIELD_INFO = { "tdci_simulation_time": tdci_simulation_time,
               "nstep"               : nstep,
               "nfields"             : nfields,
               "f0"                  : f0_values,
               "krylov_end"          : krylov_end,
               "krylov_end_n"        : krylov_end_n,
               "atoms"               : atoms
             }




########################################
# Main function
########################################

clean_files()
l = tccontroller.logger()
# logprint(string) will write a timestamped string to the log, and to STDOUT
logprint = l.logprint

# Job directory
JOBDIR = "./"

tc = tccontroller.tccontroller(JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE,
                               FIELD_INFO, logger=l, SCHEDULER=False)

ehrenfest = Ehrenfest(delta, logprint, tc)



# Print header
logprint("TDCI + TAB-DMS")
logprint("=================")

# Time steps
logprint("Propagation time step in au: " + str(delta))
logprint("TDCI simulation half time step in fs: " + str(tdci_simulation_time))
logprint("")

# Do Ehrenfest dynamics!
ehrenfest.run_ehrenfest()



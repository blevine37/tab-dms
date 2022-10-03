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
import utils

# to install h5py:
# $ apt install libhdf5-dev
# $ HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
# $ pip2 install h5py



########################################
# Constants
########################################

# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903
# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17



class Ehrenfest:
  def __init__(self, delta, logprint, tc):
    self.logprint = logprint
    self.tc = tc
    self.delta = tc.config.TIMESTEP_AU
    self.atoms = tc.config.FIELD_INFO["atoms"]
    self.masses = utils.getmasses(self.atoms)

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

  def ke_calc(self, v):
    ke = 0 # Initialize energy variable
    # Iterate over masses
    for m_i, v_i in zip(self.masses, v):
      ke += m_i * v_i.dot(v_i) / 2
    return ke

  def savestate(self, x, v, a, t, pe, TCdata):
    # Update HDF5
    self.logprint("Updating HDF5")
    ke = self.ke_calc(v)
    # TODO: save the tdci directory and other stuff from TCdata
    utils.h5py_update(x, v, a, pe, ke, t, TCdata)
    # Print HDF5 contents
    self.logprint("Printing HDF5 contents")
    utils.h5py_printall()

  def loadstate(self):
    h5f = h5py.File('data.hd5f','r')
    fields = ['atoms', 'geom','vels','accs','poten','kinen','time']
    for field in fields:
      if field not in h5f.keys():
	print("h5 file uninitialized")
	os.kill()
    x_curr = h5f['geom'][-1]
    v_curr = h5f['geom'][-1]
    a_curr = h5f['geom'][-1]
    pe_curr = h5f['geom'][-1]
    ke_curr = h5f['geom'][-1]
    t = h5f['time'][-1]
    recn = h5f['recn']
    imcn = h5f['imcn']
    self.tc.N = len(h5f['geom'])
    return x_curr, v_curr, a_curr, t, pe_curr, recn, imcn
    
  # Prepare initial state and start propagation.
  def run_ehrenfest(self):
    x, v, a, pe, recn, imcn = None, None, None, None, None, None
    t = 0
    if self.tc.config.RESTART:
      x, v, a, t, pe, recn, imcn = self.loadstate()
    else:
      utils.clean_files(self.tc.config.JOBDIR) # Clean job directory
      geomfilename = self.tc.config.xyzpath # .xyz filename in job directory
      self.atoms, x = utils.xyz_read(geomfilename)
      self.masses = utils.getmasses(self.atoms)
      v = np.zeros([len(self.atoms), 3]) # Initial velocity
      # Call Terachem to Calculate states
      gradout = self.tc.grad(x) # x should already be in angstroms here
      recn = gradout["states"][self.tc.config.initial_electronic_state]
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
      t += self.delta * autimetosec * 1e+18 # Time in Attoseconds
      x, v, a, TCdata = self.step(x, v, a, ReCn=ReCn, ImCn=ImCn) # Do propagation step
      ReCn, ImCn = None, None # Defaults to =TCdata["recn"],TCdata["imcn"] from prevstep.
      self.savestate(x, v, a, t, TCdata["eng"], TCdata)
      self.logprint("Iteration " + str(it).zfill(4) + " finished")
      
    
  # Accepts current state, runs TDCI, calculates next state.
  def step(self, x, v, a, ReCn=None, ImCn=None):
    logprint = self.logprint
    x_next = x + v*self.delta + (a*self.delta**2)/2.
    TCdata = self.tc.nextstep(x_next*bohrtoangs, ReCn=ReCn, ImCn=ImCn) # Do TDCI! \(^0^)/
    a_next = self.getAccel(TCdata["grad_half"], TCdata["recn"], TCdata["imcn"])
    v_next = v + (a + a_next) * self.delta / 2
    return x_next, v_next, a_next, TCdata
    



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
    print(v)
    ke = 0 # Initialize energy variable
    # Iterate over masses
    for m_i, v_i in zip(self.masses, v):
      ke += m_i * v_i.dot(v_i) / 2
    return ke

  def savestate(self, x, v, v_half, a, t, TCdata, atoms=None):
    # Update HDF5
    data = { 'x'         : x,
             'v'         : v,
             'v_half'    : v_half,
             'a'         : a,
             'ke'        : self.ke_calc(v),
             'time'      : t,
              }
    if TCdata is None:
      import pdb; pdb.set_trace()
      data.update({ 
                    'pe'        : 0,
	            'tdci_dir'  : "",
                 })

    else:
      data.update({ 
                    'pe'        : TCdata['eng'],
		    'recn_half' : TCdata['recn'],
	            'imcn_half' : TCdata['imcn'],
	            'tdci_dir' : TCdata['tdci_dir'],
                 })

    if atoms is not None:
      data['atoms'] = atoms
    self.logprint("Updating HDF5")
    # TODO: save the tdci directory and other stuff from TCdata
    utils.h5py_update(data)
    # Print HDF5 contents
    self.logprint("Printing HDF5 contents")
    utils.h5py_printall()

  def loadstate(self):
    config = self.tc.config
    N = int(config.restart_frame)
    x_, v_half_, a_, t_, recn_, imcn_, self.atoms = utils.h5py_copy_partial(config.restart_hdf5, config.restart_frame, config)
    time.sleep(2)
    self.tc.N = config.restart_frame
    self.tc.restart_setup()
    print(( x_, v_half_, a_, t_, recn_, imcn_))
    return x_, v_half_, a_, t_, recn_, imcn_
    
  # Prepare initial state and start propagation.
  def run_ehrenfest(self):
    x, v, a, pe, recn, imcn = None, None, None, None, None, None
    t = 0
    if self.tc.config.RESTART:
      x, v, a, t, recn, imcn = self.loadstate()
    else:
      utils.clean_files(self.tc.config.JOBDIR) # Clean job directory
      geomfilename = self.tc.config.xyzpath # .xyz filename in job directory
      self.atoms, x = utils.xyz_read(geomfilename)
      #utils.h5py_update({'atoms': self.atoms})
      self.masses = utils.getmasses(self.atoms)
      # Call Terachem to Calculate states
      gradout = self.tc.grad(x) # x should already be in angstroms here
      t = 0.0
      recn = gradout["states"][self.tc.config.initial_electronic_state]
      imcn = np.zeros(len(recn))
      x = x / bohrtoangs # Initial geometry in Bohr
      v_timestep = np.zeros([len(self.atoms), 3]) # Velocity at t=0
      a = np.zeros([len(self.atoms), 3]) # Accel at t=0
      v, a, TCdata = self.halfstep(x, v_timestep, recn, imcn) # Do TDCI halfstep!! \(^0^)/
      self.savestate(x, v_timestep, v, a, t, TCdata, atoms=self.atoms) # Save initial state?

    self.propagate(x, v, t, recn, imcn)
    

  # Does Ehrenfest propagation loop
  
  # t_init    : time t
  # x_init    : Coordinates at time t
  # v_init    : Velocity at t+(dt/2)
  # ReCn_init : Real CI Vector at time t+(dt/2)
  # ImCn_init : Imaginary CI Vector at time t+(dt/2)
  # 
  def propagate(self, x_init, v_init, t_init, ReCn_init, ImCn_init=None):
    t = t_init
    x, v, ReCn, ImCn = x_init, v_init, ReCn_init, ImCn_init
    a = 0.0 # initial acceleration is not used
    TCdata = None
    for it in range(0,99999999): # go forever! :D
      t += self.delta * autimetosec * 1e+18 # Time in Attoseconds
      #xprev = np.copy(x)
      x, v_timestep, v, a, TCdata = self.step(x, v, ReCn=ReCn, ImCn=ImCn) # Do propagation step
      ReCn, ImCn = None, None # Defaults to =TCdata["recn"],TCdata["imcn"] from prevstep.
      #self.savestate(xprev, v_timestep, v, a, t, TCdata)
      self.savestate(x, v_timestep, v, a, t, TCdata)
      self.logprint("Iteration " + str(it).zfill(4) + " finished")
      
    
  # Accepts current state, runs TDCI, calculates next state.
  # Input:
  #  x      : Coordinates at time t-dt
  #  v      : Velocity at time t-(dt/2)
  # Output:
  #  x_next     : Coordinates at time t
  #  v_timestep : Velocity at time t
  #  v_next     : Velocity at time t+(dt/2)
  #  a          : Acceleration at time t
  #  TCdata     : Return dictionary from tccontroller TDCI job from t-(dt/2) to t+(dt/2)
  def step(self, x, v, ReCn=None, ImCn=None):
    x_next = x + v*self.delta
    TCdata = self.tc.nextstep(x_next*bohrtoangs, ReCn=ReCn, ImCn=ImCn) # Do TDCI! \(^0^)/
    a = self.getAccel(TCdata["grad_half"], TCdata["recn"], TCdata["imcn"])
    v_timestep = v + a*self.delta/2.
    v_next = v + a*self.delta
    return x_next, v_timestep, v_next, a, TCdata
    
  # Input:
  #  x      : Coordinates at time t
  #  v      : Velocity at time t
  # Output: 
  #  v_next : Velocity at time t+(dt/2)
  #  a      : Acceleration at time t
  #  TCdata : Return dictionary from tccontroller TDCI job from t to t+(dt/2)
  def halfstep(self, x, v, ReCn=None, ImCn=None):
    TCdata = self.tc.halfstep(x*bohrtoangs, ReCn=ReCn, ImCn=ImCn) # Do TDCI! \(^0^)/
    a = self.getAccel(TCdata["grad"], TCdata["recn"], TCdata["imcn"])
    print((v, a, self.delta))
    v_next = v + a*self.delta/2.
    return v_next, a, TCdata


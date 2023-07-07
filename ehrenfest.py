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
      #import pdb; pdb.set_trace()
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
    utils.h5py_update(data)

  def loadstate(self):
    config = self.tc.config
    N = int(config.restart_frame) # Restart at frame N
    # Create a new hdf5 file with steps 0,...,N-1, and then restart calculation with frame N.
    x_, v_half_, a_, t_, recn_, imcn_, self.atoms = utils.h5py_copy_partial(config.restart_hdf5, config.restart_frame, config)
    time.sleep(2) # Wait a sec to make sure IO operations are done.
    self.tc.N = config.restart_frame
    self.tc.restart_setup() # Prepare tccontroller for running from a restart
    print(( x_, v_half_, a_, t_, recn_, imcn_))
    return x_, v_half_, a_, t_, recn_, imcn_
    
  # Prepare initial state and start propagation.
  def run_ehrenfest(self):
    x, v, v_timestep, a, pe, recn, imcn = None, None, None, None, None, None, None
    t = 0
    if self.tc.config.RESTART:
      x, v, a, t, recn, imcn = self.loadstate()
    else:
      utils.clean_files(self.tc.config.JOBDIR) # Clean job directory
      geomfilename = self.tc.config.xyzpath # .xyz filename in job directory
      self.atoms, x = utils.xyz_read(geomfilename) # x in angstroms initially
      x = x / bohrtoangs # cast Initial geometry to Bohr
      self.masses = utils.getmasses(self.atoms)
      #utils.h5py_update({'atoms': self.atoms})

      if self.tc.config.WIGNER_PERTURB: # Perturb according to wigner distribution
        TCdata = None
        if self.tc.config.HESSIAN_FILE is None:
          TCdata = self.tc.hessian(x*bohrtoangs, self.tc.config.WIGNER_TEMP )
        else:
          TCdata = utils.read_hessfile(len(x), self.tc.config.HESSIAN_FILE )
        x, v_timestep = utils.initial_wigner( self.tc.config.WIGNER_SEED,
                                              x, TCdata["hessian"], self.masses,
                                              self.tc.config.WIGNER_TEMP ) 
	#x = x / bohrtoangs
	#v_timestep = v_timestep / bohrtoangs
        print(x)
        utils.xyz_write(self.atoms, x*bohrtoangs, "wigner_output.xyz")


      # Call Terachem to Calculate states
      gradout = self.tc.grad(x*bohrtoangs)
      t = 0.0
      recn = gradout["states"][self.tc.config.initial_electronic_state]
      imcn = np.zeros(len(recn))
      v_timestep = np.zeros([len(self.atoms), 3]) # Velocity at t=0


      a = np.zeros([len(self.atoms), 3]) # Accel at t=0
      v, a, TCdata = self.halfstep(x, v_timestep, recn, imcn) # Do TDCI halfstep
      self.savestate(x, v_timestep, v, a, t, TCdata, atoms=self.atoms) # Save initial state

    self.propagate(x, v, t, recn, imcn)
    

  # Does Ehrenfest propagation loop
  
  # t_init    : time t
  # x_init    : Coordinates at time t
  # v_init    : Velocity at t+(dt/2)
  # ReCn_init : Real CI Vector at time t+(dt/2)
  # ImCn_init : Imaginary CI Vector at time t+(dt/2)
  # 
  def propagate(self, x_init, v_init, t_init, ReCn_init, ImCn_init=None):
    realtime_start = time.time()  # For benchmarking
    t = t_init
    it = int(t_init/(self.delta*autimetosec*1e+18))
    self.logprint("Starting at time "+str(t)+" as, step "+str(it))
    self.logprint("Running until time "+str(self.tc.config.DURATION*1000)+" as, "+str(self.tc.config.MAXITERS)+" steps")
    x, v, ReCn, ImCn = x_init, v_init, ReCn_init, ImCn_init
    a = 0.0 # initial acceleration is not used
    TCdata = None
    while it < self.tc.config.MAXITERS: # main loop!
      t += self.delta * autimetosec * 1e+18 # Time in Attoseconds
      x_prev, v_prev, ReCn_prev, ImCn_prev, TCdata_prev = x, v, ReCn, ImCn, TCdata
      x, v_timestep, v, a, TCdata = self.step(x, v, ReCn=ReCn, ImCn=ImCn) # Do propagation step
      ReCn, ImCn = TCdata["recn"], TCdata["imcn"]
      self.savestate(x, v_timestep, v, a, t, TCdata)
      self.logprint("Iteration " + str(it).zfill(4) + " finished")
      it+=1
    self.logprint("Completed Ehrenfest Propagation!")
    time_simulated = (t-t_init)/1000.
    import datetime
    realtime = str( datetime.timedelta( seconds=(time.time() - realtime_start) )) 
    self.logprint("Simulated "+str(time_simulated)+" fs with "+str(it)+" steps in "+realtime+" Real time.")

      
    
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
    a = self.getAccel(TCdata["grad_end"], TCdata["recn"], TCdata["imcn"])
    print((v, a, self.delta))
    v_next = v + a*self.delta/2.
    return v_next, a, TCdata



import os, sys, shutil, struct, time
import numpy as np
import h5py
import subprocess
########################################
# Constants
########################################

# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903
# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17
AUT2AS = 24.188843265857 # Atomic unit of time to Attoseconds



# Need to make sure hdf5 and job directories from previous runs don't get in the way
def clean_files(jobdir):
  if (not (os.path.exists(jobdir+"electronic") or os.path.exists(jobdir+"data.hdf5"))):
    return
  if os.path.exists(jobdir+"oldrun/"):
    shutil.rmtree(jobdir+"oldrun/")
  os.makedirs(jobdir+"oldrun/")
  if os.path.exists(jobdir+"electronic"):
    shutil.move(jobdir+"electronic", jobdir+"oldrun/electronic")
  if os.path.exists(jobdir+"data.hdf5"):
    shutil.move(jobdir+"data.hdf5", jobdir+"oldrun/data.hdf5")


# make sure filename is clear to write without losing existing files
# if the folder contains foo, foo.1, foo.2
# makes the following renames: foo.2 -> foo.3, foo.1 -> foo.2, foo -> foo.1
# (deletes after maxrotate)
def rotate_file(filename, maxrotate=10):
  if not os.path.exists(filename):
    return None
  # 'dotted' means ended with a rotation dot, i.e. foo.1 instead of foo
  already_dotted = False
  if len(filename.split('.')) > 1:
    # apparently .isnumeric only works on unicode in py2.7
    if unicode(filename.split('.')[-1]).isnumeric():
      already_dotted = True
  newname = filename+'.1' # If not dotted
  if already_dotted: # Overwrite if already dotted
    newname = filename.split('.')
    newname[-1] = str(int(newname[-1])+1)
    newname = '.'.join(newname)

  if os.path.exists(newname):
    if int(newname.split('.')[-1]) >= maxrotate:
      p = subprocess.Popen('rm -rf '+newname, shell=True)
      p.wait()
    rotate_file(newname, maxrotate)
  p = subprocess.Popen('mv '+filename+' '+newname, shell=True)
  p.wait()
  return None



########################################
# Geometry read and write
########################################

# XYZ files always expected to use Angstroms
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


def read_hessfile(natoms, filepath):
  # 2*sizeof(int)+sizeof(double)+natoms*sizeof(double4);
  nbytes = 2*4+8+(natoms)*(4*8)
  f = open(filepath, 'rb')
  #print(len(f.read()))
  #f.seek(0)
  first = f.read(nbytes)
  second = f.read(((3*natoms)**2)*8)
  third = f.read((3*3*natoms)*8)
  rest = f.read()
  if len(rest) != 0:
    print("ERROR: Extra bytes in Hessian.bin, something isn't right...")
  #print( (len(first),len(second),len(third),len(rest)))
  hessian = np.array(struct.unpack('d'*((3*natoms)**2), second))
  hessian.resize((3*natoms,3*natoms))
  size = 3 # if RunPolarizability in TC, this should be 12. Not sure if we ever need that.
  dipolederiv = np.array(struct.unpack('d'*(size*3*natoms), third))
  return {"hessian": hessian, "dipolederiv": dipolederiv}



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
# Gaussian width initialization
########################################

def getdcps(atoms):
  
  # Gaussian width used to compute decoherence time constant for TAB
  # Source: https://doi.org/10.1016/j.chemphys.2010.03.020
  dcpdata = {}
  dcpdata['H'] = 4.7
  dcpdata['C'] = 22.7
  dcpdata['N'] = 19.0
  dcpdata['O'] = 12.2
  dcpdata['F'] = 8.5
  dcpdata['S'] = 16.7
  dcpdata['Cl'] = 7.4

  # Build numpy array of Gaussian widths
  natoms = len(atoms)
  dcps = np.empty([natoms])
  for i in range(0, natoms):
    dcps[i] = dcpdata[atoms[i]]

  # Return Gaussian widths
  return dcps


# Wigner distribution

def initial_wigner(iseed, x, hessian, masses, temp=0.0):
  """Wigner distribution of positions and momenta
  Works at finite temperature if a temp parameter is passed
  If temp is not provided temp = 0 is assumed"""

  print( "## randomly selecting Wigner initial conditions at T="+str(temp))
  #ndims = self.get_numdims()
  ndims = 3*len(masses)
  #h5f = h5py.File('hessian.hdf5', 'r')
  #pos = h5f['geometry'][:].flatten()
  #h = h5f['hessian'][:]
  pos = np.copy(x)
  pos.resize((ndims))
  h = np.copy(hessian)
  #m = self.get_masses()
  m = np.zeros((ndims))
  for i in range(0,ndims): # our masses has len natoms, this code needs len 3*natoms
    m[i] = float(masses[i/3])
  sqrtm = np.sqrt(m)

  # build mass weighted hessian
  h_mw = np.zeros_like(h)

  for idim in range(ndims):
    h_mw[idim, :] = h[idim, :] / sqrtm

  for idim in range(ndims):
    h_mw[:, idim] = h_mw[:, idim] / sqrtm

  # symmetrize mass weighted hessian
  h_mw = 0.5 * (h_mw + h_mw.T)

  # diagonalize mass weighted hessian
  evals, modes = np.linalg.eig(h_mw)

  # sort eigenvectors
  idx = evals.argsort()[::-1]
  evals = evals[idx]
  modes = modes[:, idx]

  print('# eigenvalues of the mass-weighted hessian are (a.u.)')
  print(evals)

  # Checking if frequencies make sense
  freq_cm = np.sqrt(evals[0:ndims - 6])*219474.63
  n_high_freq = 0
  print('Frequencies in cm-1:')
  for freq in freq_cm:
    if freq > 5000: n_high_freq += 1
    print(str(freq))
    assert not np.isnan(freq), "NaN encountered in frequencies! Exiting"

  if n_high_freq > 0: print("Number of frequencies > 5000cm-1:"+str(n_high_freq))

  # seed random number generator
  np.random.seed(iseed)
  alphax = np.sqrt(evals[0:ndims - 6]) / 2.0

  # finite temperature distribution
  if temp > 1e-05:
    beta = 1 / (temp * 0.000003166790852)
    print("beta = "+str(beta))
    alphax = alphax * np.tanh(np.sqrt(evals[0:ndims - 6]) * beta / 2)
  sigx = np.sqrt(1.0 / (4.0 * alphax))
  sigp = np.sqrt(alphax)

  dtheta = 2.0 * np.pi * np.random.rand(ndims - 6)
  dr = np.sqrt(np.random.rand(ndims - 6))

  dx1 = dr * np.sin(dtheta)
  dx2 = dr * np.cos(dtheta)

  rsq = dx1 * dx1 + dx2 * dx2

  fac = np.sqrt(-2.0 * np.log(rsq) / rsq)

  x1 = dx1 * fac
  x2 = dx2 * fac

  posvec = np.append(sigx * x1, np.zeros(6))
  momvec = np.append(sigp * x2, np.zeros(6))

  deltaq = np.matmul(modes, posvec) / sqrtm
  pos += deltaq
  mom = np.matmul(modes, momvec) * sqrtm

  #self.set_positions(pos)
  #self.set_momenta(mom)

  zpe = np.sum(alphax[0:ndims - 6])
  ke = 0.5 * np.sum(mom * mom / m)
  #         print np.sqrt(np.tanh(evals[0:ndims-6]/(2*0.0031668)))
  print("# ZPE = "+str(zpe))
  print("# kinetic energy = "+str(ke))
  print("END WIGNER")

  v = np.zeros_like(mom)
  print(mom)
  print(m)
  for i in range(0,len(mom)):
    v[i] = mom[i]/m[i]

  pos.resize((len(masses), 3))
  v.resize((len(masses), 3))

  print("newpos:")
  print(pos)
  print("newv:")
  print(v)
  return pos, v

#Initialize wavefunction from file
def read_c0files():
  try:
       bytesize = os.path.getsize('ReCn0.bin')
       cnlen = bytesize/8 #assume stream of doubles
       f = open('ReCn0.bin', 'rb')
       recn = np.array(struct.unpack('d'*cnlen, f.read()))
       f.close()
       f = open('ImCn0.bin', 'rb')
       imcn = np.array(struct.unpack('d'*cnlen, f.read()))
       f.close()
  except:
       print 'Something went wrong reading ./ReCn0.bin ./ImCn0.bin'
       sys.exit()
  return recn, imcn


########################################
# h5py
########################################
# time in attoseconds
def h5py_update(data):
  #import pdb; pdb.set_trace()

  # Get array dimension
  n = 0
  if 'atoms' in data: 
    n = len(data['atoms'])
  else:
    n = data['x'].shape[0]
  ndets = 0
  try:
    ndets = len(data['recn_half'])
  except:
   pass

  # Open h5py file
  h5f = h5py.File('data.hdf5', 'a')

  # Create datasets
  if 'atoms' not in h5f.keys():

    str_dtype = h5py.special_dtype(vlen=str)
    h5f.create_dataset('atoms', (1, n), maxshape=(1, n), dtype=str_dtype)

    h5f.create_dataset('tdci_dir', (0,), maxshape=(None,), dtype=str_dtype)
    h5f.create_dataset('x', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('v', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('v_half', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('a', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('pe', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('ke', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('time', (0,), maxshape=(None,), dtype='float64')

  if (('recn_half' in data.keys()) and ('recn_half' not in h5f.keys())):
    ndets = len(data['recn_half'])
    h5f.create_dataset('recn_half', (1, ndets), maxshape=(None, ndets), dtype='float64')
    h5f.create_dataset('imcn_half', (1, ndets), maxshape=(None, ndets), dtype='float64')


  static_keys = ['atoms']
  for key in h5f.keys():
    dset = h5f[key]
    if key not in static_keys:
      # Need to be careful with inputs to make sure some arrays dont wind up longer than others
      dset.resize(dset.len() + 1, axis=0)
    if key in data:
      dset[-1] = data[key]

  # Close
  h5f.close()
  # had an error earlier when the h5 file was opened again before it was finished closing
  # think that might be due to filesystem lag or something idk so here's a sleep
  time.sleep(1)

# Old runs use testN.in/out filenames, we use tc.in/out now.
def rename_legacy(dirpath):
  d = dirpath
  if os.path.exists(d+"electronic/1/test1.out"): # Need to rename all the testN.in/out files to tc.in/out
    for i in range(1,self.nstep):
      dt = d + "electronic/"+str(i)+"/"
      os.rename( dt+"test"+str(i)+".in", dt+"tc.in")
      os.rename( dt+"test"+str(i)+".out", dt+"tc.out")


# Take an h5 file from a previous run, and copy it up to a frame
def h5py_copy_partial(oldh5f, lastframe, config):
  #new_oldfile = "".join(oldh5f.split(".")[:-1])
  #shutil.copy(oldh5f, new_oldfile+"_old.hdf5" )
  new_oldfile = oldh5f
  dirpath = os.path.dirname(oldh5f)
  rename_legacy(dirpath)
  if dirpath != "": dirpath=dirpath+"/"
  if os.path.exists(dirpath+"data.hdf5"):
     shutil.move(dirpath+"data.hdf5", dirpath+"data_old.hdf5")
  if os.path.basename(oldh5f) == "data.hdf5":
     new_oldfile = dirpath+"data_old.hdf5"

  oldh = h5py.File(new_oldfile, 'r')
  print("Writing data.hdf5 to path:")
  print(dirpath+"data.hdf5")
  h5f = h5py.File(dirpath+"data.hdf5", 'a')
  
  maxn = oldh['x'].shape[0]
  natoms = oldh['x'].shape[1]
  ndets = oldh['recn_half'].shape[1]

  if maxn < lastframe:
    raise ValueError('Tried to restart on frame '+str(lastframe)+', but '+str(oldh5f)+' only has '+str(maxn)+' frames.')

  # Create new datasets
  
  n = natoms
  str_dtype = h5py.special_dtype(vlen=str)
  h5f.create_dataset('atoms', (1, n), maxshape=(1, n), dtype=str_dtype)

  h5f.create_dataset('tdci_dir', (lastframe,), maxshape=(None,), dtype=str_dtype)
  h5f.create_dataset('x', (lastframe, n, 3), maxshape=(None, n, 3), dtype='float64')
  h5f.create_dataset('v', (lastframe, n, 3), maxshape=(None, n, 3), dtype='float64')
  h5f.create_dataset('v_half', (lastframe, n, 3), maxshape=(None, n, 3), dtype='float64')
  h5f.create_dataset('a', (lastframe, n, 3), maxshape=(None, n, 3), dtype='float64')
  h5f.create_dataset('pe', (lastframe,), maxshape=(None,), dtype='float64')
  h5f.create_dataset('ke', (lastframe,), maxshape=(None,), dtype='float64')
  h5f.create_dataset('time', (lastframe,), maxshape=(None,), dtype='float64')

  h5f.create_dataset('recn_half', (lastframe, ndets), maxshape=(None, ndets), dtype='float64')
  h5f.create_dataset('imcn_half', (lastframe, ndets), maxshape=(None, ndets), dtype='float64')

  # Copy data
  #static_keys = ['atoms', 'tdci_dir']
  static_keys = ['atoms']
  h5f['atoms'][0] = oldh['atoms'][0]
  for key in h5f.keys():
    if key in static_keys: continue
    for i in range(0,lastframe):
      #print((key, i, h5f[key][i], oldh[key][i]))
      sys.stdout.flush()
      h5f[key][i] = oldh[key][i]

  # Copy all the stuff to restart job lastframe
  
  # Rename old TDCI subdirectory, create new one
  #shutil.move(config.JOBDIR+"electronic", config.JOBDIR+"electronic_old")
  rotate_file(config.JOBDIR+"electronic_old")
  p = subprocess.Popen('mkdir '+config.JOBDIR+'electronic_old', shell=True)
  p.wait()
  i = lastframe
  while (i<maxn):
    if os.path.exists(oldh['tdci_dir'][i]):
      p = subprocess.Popen('mv '+oldh['tdci_dir'][i]+' '+config.JOBDIR+'electronic_old/.', shell=True)
      p.wait()
    i+=1

  p = subprocess.Popen('mkdir '+oldh['tdci_dir'][lastframe], shell=True)
  p.wait()
  prevjob_dir = oldh['tdci_dir'][lastframe-1]
  newjob_dir = oldh['tdci_dir'][lastframe]
  newjob_old = config.JOBDIR+'electronic_old/'+newjob_dir.split('/')[-1]+'/'
  # Set up new jobfiles
  # These are actually just discarded...
  p = subprocess.Popen('cp '+prevjob_dir+"NewCoors.bin "+newjob_dir+"/PrevCoors.bin ;"+
                       'cp '+prevjob_dir+"NewC.bin "+newjob_dir+"/PrevC.bin ;"+
                       'cp '+prevjob_dir+"ReCn_end.bin "+newjob_dir+"/recn_init.bin ;"+
                       'cp '+prevjob_dir+"ImCn_end.bin "+newjob_dir+"/imcn_init.bin ;"+
                       'cp '+prevjob_dir+"field0.bin "+newjob_dir+"/field0.bin ;"+
                       'cp '+newjob_old+"temp.xyz "+newjob_dir+"/temp.xyz ;"+
                       'cp '+newjob_old+"tc.in "+newjob_dir+"/tc.in ;", shell=True)
  p.wait()
  
  x = xyz_read(prevjob_dir+"/temp.xyz")[1]/bohrtoangs
  v_half = np.array(h5f['v_half'][lastframe-1])
  a = np.array(h5f['a'][lastframe-1])
  t = float(h5f['time'][lastframe-1])
  recn = None # Ugh we're not storing recn_end in hdf5...
  imcn = None
  atoms = list(h5f['atoms'][0])
  oldh.close()
  h5f.close()
  time.sleep(1)
  return x, v_half, a, t, recn, imcn, atoms

def h5py_printall():

  time.sleep(2)
  # Open h5py file
  h5f = h5py.File('data.hdf5', 'r')

  # Get number of iterations
  niters = h5f['x'].shape[0]

  # Get number atoms
  natoms = h5f['x'].shape[1]

  # Iterate and print energies
  poten = h5f['pe']
  kinen = h5f['ke']
  print(('{:>25s}'*3).format('Potential', 'Kinetic', 'Total'))
  for it in range(0, niters):
    pot = poten[it]
    kin = kinen[it]
    tot = pot + kin
    print(('{:5d} {:06.0f} '+'{:25.17f}'*3).format( it, h5f['time'][it], pot, kin, tot))
  print("")

  # Close
  h5f.close()


# Prints the last TDCI job in an electronic/ folder
def lastfolder(electronicdir):
  folders = os.listdir(electronicdir)
  folders.sort(reverse=True, key=lastfolder_sort)
  return folders[0]

# sorting method for lastfolder()
def lastfolder_sort(element):
  try:
    return int(element)
  except:
    return 0

# Takes inputfile.py module and copies its namespace to this class instance
# Then modifies stuff that the user shouldn't touch.
# like TDCI_TEMPLATE options that tccontroller fiddles with.
class ConfigHandler:
  def __init__(self, config):
    self.TAB = config.TAB
    self.JOBDIR = config.JOBDIR
    self.initial_electronic_state = config.initial_electronic_state
    self.RESTART = config.RESTART
    if self.RESTART: # Shouldnt need to include them if you're not restarting.
      if (config.restart_frame is True): # Auto-detect
        lf = lastfolder(self.JOBDIR+"electronic/")
        self.restart_frame = int(lf.split("/")[-1])-2 # Two before last detected folder to be safe...
      else:
        self.restart_frame = config.restart_frame
      self.restart_hdf5 = config.restart_hdf5
    self.SCHEDULER = config.SCHEDULER
    self.TERACHEM = config.TERACHEM
    self.TIMESTEP_AU = config.TIMESTEP_AU # Dynamics time step in atomic units
    self.nfields = config.nfields
    self.krylov_end = config.krylov_end
    self.krylov_end_n = config.krylov_end_n
    self.krylov_end_interval = config.krylov_end_interval
    self.TDCI_TEMPLATE = config.TDCI_TEMPLATE
    self.job_template_contents = config.job_template_contents
    self.JOB_TEMPLATE = config.job_template_contents
    self.f0_values = config.f0_values
    
    self.USEC0FILE = False #Initial WF 
    try: self.USEC0FILE = config.USEC0FILE
    except: pass

    self.GRADSELECT = False #Do not calculate gradient of unpopulated states
    try: self.GRADSELECT = config.GRADSELECT
    except: pass 

    self.FIX_FOMO = False
    try: self.FIX_FOMO = config.FIX_FOMO
    except: pass

    self.DURATION = 999999999999999
    try: self.DURATION = config.DURATION
    except: pass

    self.TIMESTEP_E_AS = 0.10 # Electronic Timestep in Attoseconds
    try: self.TIMESTEP_E_AS = config.TIMESTEP_E_AS
    except: pass
    
    # Make sure TIMESTEP_AU is actually a multiple of TIMESTEP_E_AS
    self.NSTEPS_TDCI = int((AUT2AS*self.TIMESTEP_AU)/self.TIMESTEP_E_AS)
    self.nstep = self.NSTEPS_TDCI
    self.TIMESTEP_AU = (self.TIMESTEP_E_AS*self.NSTEPS_TDCI)/AUT2AS
    self.MAXITERS = int(self.DURATION*1000/(AUT2AS*self.TIMESTEP_AU))+1

    self.WIGNER_PERTURB = False
    self.WIGNER_TEMP = 0.0
    self.WIGNER_SEED = 0
    try:
      self.WIGNER_PERTURB = config.WIGNER_PERTURB
      self.WIGNER_TEMP = config.WIGNER_TEMP
      self.WIGNER_SEED = config.WIGNER_SEED
    except: pass

    self.HESSIAN_FILE = None
    try:
      self.HESSIAN_FILE = config.HESSIAN_FILE
    except: pass

    self.atoms, self.xyz = xyz_read(config.xyzpath)
    self.xyzpath = config.xyzpath
    try: self.velpath = config.velpath
    except: self.velpath = False 
    # TDCI simulation time in femtoseconds
    self.tdci_simulation_time = float(config.TIMESTEP_AU)*autimetosec*1e15 # fs/s

    TDCI_TEMPLATE = self.TDCI_TEMPLATE
    tdci_simulation_time = self.tdci_simulation_time
    TDCI_TEMPLATE["coordinates"] = "temp.xyz"
    TDCI_TEMPLATE["run"] = "tdci"
    TDCI_TEMPLATE["tdci_simulation_time"] = str(self.tdci_simulation_time)
    TDCI_TEMPLATE["tdci_nstep"] = str(self.nstep)
    TDCI_TEMPLATE["tdci_nfields"] = str(self.nfields)
    TDCI_TEMPLATE["tdci_grad_end"] = "no"
    TDCI_TEMPLATE["tdci_grad_init"] = "no"
    TDCI_TEMPLATE["tdci_grad_half"] = "yes"
    TDCI_TEMPLATE["tdci_grad_states"] = "no"
    TDCI_TEMPLATE["tdci_grad_states_select"] = ""
    TDCI_TEMPLATE["tdci_fieldfile0"] = "field0.bin"
    # Krylov subspace options
    TDCI_TEMPLATE["tdci_krylov_end"] = ("yes" if self.krylov_end else "no")
    TDCI_TEMPLATE["tdci_krylov_end_n"] = self.krylov_end_n
    TDCI_TEMPLATE["tdci_krylov_end_interval"] = self.krylov_end_interval
    # Options that tccontroller may remove on initial step
    TDCI_TEMPLATE["tdci_diabatize_orbs"]  = "yes"
    TDCI_TEMPLATE["tdci_write_binfiles"]  = "yes"
    TDCI_TEMPLATE["tdci_recn_readfile"]   = "recn_init.bin"
    TDCI_TEMPLATE["tdci_imcn_readfile"]  = "imcn_init.bin"
    TDCI_TEMPLATE["tdci_prevorbs_readfile"] = "PrevC.bin"
    TDCI_TEMPLATE["tdci_prevcoords_readfile"] = "PrevCoors.bin"

    self.FIELD_INFO = { "tdci_simulation_time": self.tdci_simulation_time,
                        "nstep"               : self.nstep,
                        "nfields"             : self.nfields,
                        "f0"                  : self.f0_values,
                        "krylov_end"          : self.krylov_end,
                        "krylov_end_n"        : self.krylov_end_n,
		        "atoms"               : self.atoms
                      }






# Below code from
# https://stackoverflow.com/questions/6811902/import-arbitrary-named-file-as-a-python-module-without-generating-bytecode-file
# This should allow end user to put their python input files in arbitrary locations and be sloppy with naming them
# also avoids creating a bytecode compiled version of the user's input file.
import imp, contextlib
@contextlib.contextmanager
def preserve_value(namespace, name):
    """ A context manager to preserve, then restore, the specified binding.

        :param namespace: The namespace object (e.g. a class or dict)
            containing the name binding.
        :param name: The name of the binding to be preserved.
        :yield: None.

        When the context manager is entered, the current value bound to
        `name` in `namespace` is saved. When the context manager is
        exited, the binding is re-established to the saved value.

        """
    saved_value = getattr(namespace, name)
    yield
    setattr(namespace, name, saved_value)


def make_module_from_file(module_name, module_filepath):
    """ Make a new module object from the source code in specified file.

        :param module_name: The name of the resulting module object.
        :param module_filepath: The filesystem path to open for
            reading the module's Python source.
        :return: The module object.

        The Python import mechanism is not used. No cached bytecode
        file is created, and no entry is placed in `sys.modules`.

        """
    py_source_open_mode = 'U'
    py_source_description = (".py", py_source_open_mode, imp.PY_SOURCE)

    with open(module_filepath, py_source_open_mode) as module_file:
        with preserve_value(sys, 'dont_write_bytecode'):
            sys.dont_write_bytecode = True
            module = imp.load_module(
                    module_name, module_file, module_filepath,
                    py_source_description)

    return module


def import_program_as_module(program_filepath):
    """ Import module from program file `program_filepath`.

        :param program_filepath: The full filesystem path to the program.
            This name will be used for both the source file to read, and
            the resulting module name.
        :return: The module object.

        A program file has an arbitrary name; it is not suitable to
        create a corresponding bytecode file alongside. So the creation
        of bytecode is suppressed during the import.

        The module object will also be added to `sys.modules`.

        """
    module_name = os.path.basename(program_filepath)

    module = make_module_from_file(module_name, program_filepath)
    sys.modules[module_name] = module

    return module






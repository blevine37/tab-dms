
import os, sys, shutil, time
import numpy as np
import h5py
########################################
# Constants
########################################

# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903
# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17



# Need to make sure hdf5 and job directories from previous runs don't get in the way
def clean_files(jobdir):
  if os.path.exists(jobdir+"oldrun/"):
    shutil.rmtree(jobdir+"oldrun/")
  os.makedirs(jobdir+"oldrun/")
  if os.path.exists(jobdir+"electronic"):
    shutil.move(jobdir+"electronic", jobdir+"oldrun/electronic")
  if os.path.exists(jobdir+"data.hdf5"):
    shutil.move(jobdir+"data.hdf5", jobdir+"oldrun/data.hdf5")



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
# h5py
########################################
# time in attoseconds
def h5py_update(x, v, a, pe, ke, Time, TCdata=None):

  # Get array dimension
  n = x.shape[0]
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

    h5f.create_dataset('x', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('v', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('a', (0, n, 3), maxshape=(None, n, 3), dtype='float64')
    h5f.create_dataset('pe', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('ke', (0,), maxshape=(None,), dtype='float64')
    h5f.create_dataset('time', (0,), maxshape=(None,), dtype='float64')

  # Resize
  for key in ['x','v','a','pe','ke','time', 'tdci_dir', 'recn', 'imcn']:
    dset = h5f[key]
    dset.resize(dset.len() + 1, axis=0)

  # Store data
  h5f['x'][-1] = x
  h5f['v'][-1] = v
  h5f['a'][-1] = a
  h5f['pe'][-1] = pe
  h5f['ke'][-1] = ke
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
    print(('{:25.17f}'*3).format(pot, kin, tot))
  print("")

  # Close
  h5f.close()









# Takes inputfile.py module and copies its namespace to this class instance
# Then modifies stuff that the user shouldn't touch.
# like TDCI_TEMPLATE options that tccontroller fiddles with.
class ConfigHandler:
  def __init__(self, config):
    self.JOBDIR = config.JOBDIR
    self.initial_electronic_state = config.initial_electronic_state
    self.RESTART = config.RESTART
    self.SCHEDULER = config.SCHEDULER
    self.TERACHEM = config.TERACHEM
    self.TIMESTEP_AU = config.TIMESTEP_AU # Dynamics time step in atomic units
    self.nstep = config.NSTEPS_TDCI
    self.nfields = config.nfields
    self.krylov_end = config.krylov_end
    self.krylov_end_n = config.krylov_end_n
    self.krylov_end_interval = config.krylov_end_interval
    self.TDCI_TEMPLATE = config.TDCI_TEMPLATE
    self.job_template_contents = config.job_template_contents
    self.JOB_TEMPLATE = config.job_template_contents
    self.f0_values = config.f0_values

    self.atoms, self.xyz = xyz_read(config.xyzpath)
    self.xyzpath = config.xyzpath
    # TDCI simulation time in femtoseconds
    self.tdci_simulation_time = float(config.TIMESTEP_AU)*autimetosec*1e15 # fs/s

    TDCI_TEMPLATE = self.TDCI_TEMPLATE
    tdci_simulation_time = self.tdci_simulation_time
    TDCI_TEMPLATE["coordinates"] = "temp.xyz"
    TDCI_TEMPLATE["run"] = "tdci"
    TDCI_TEMPLATE["tdci_simulation_time"] = str(self.tdci_simulation_time)
    TDCI_TEMPLATE["tdci_nstep"] = str(self.nstep)
    TDCI_TEMPLATE["tdci_nfields"] = str(self.nfields)
    TDCI_TEMPLATE["tdci_gradient"] = "yes" # TODO: only do grad at halfstep to save time
    TDCI_TEMPLATE["tdci_grad_init"] = "yes"
    TDCI_TEMPLATE["tdci_grad_half"] = "yes"
    TDCI_TEMPLATE["tdci_fieldfile0"] = "field0.bin"
    # Krylov subspace options
    TDCI_TEMPLATE["tdci_krylov_end"] = ("yes" if self.krylov_end else "no")
    TDCI_TEMPLATE["tdci_krylov_end_n"] = self.krylov_end_n
    TDCI_TEMPLATE["tdci_krylov_end_interval"] = self.krylov_end_interval
    # Options that tccontroller may remove on initial step
    TDCI_TEMPLATE["tdci_diabatize_orbs"]  = "yes"
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






# Below code stolen from
# https://stackoverflow.com/questions/6811902/import-arbitrary-named-file-as-a-python-module-without-generating-bytecode-file
# This should allow end user to put their python input files in arbitrary locations and be sloppy wih naming them
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






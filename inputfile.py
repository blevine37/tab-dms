
METHOD = 0 # Ehrenfest = 0, TAB = 1

JOBDIR = "./"
xyzpath = "benzene_perturbed.xyz"
velpath = ""
initial_electronic_state = 0  # S0 = 0, S1 = 1, etc.

RESTART = False
restart_frame = True  # True automatically detects last valid step
restart_hdf5 = "restart.hdf5"
TERACHEM = "/home/adurden/terachem/build/bin/" # directory containing terachem executable


TIMESTEP_AU = 4.0 # Nuclear timestep in Atomic Time units. 1 au_t ~= 24 attoseconds
# Note: TIMESTEP_AU may be altered slightly to make it a multiple of the electronic timestep.
DURATION = 1000 # Time to simulate in femtoseconds


WIGNER_PERTURB = True   # Perturb the initial position and velocity based on Wigner distribution
WIGNER_TEMP = 0.0       # Temperature for Wigner distribution
import random
WIGNER_SEED = random.randint(0, 2**32-1)     # Random seed for Wigner distribution sampling
# If you already ran a Hessian calculation, you can include the path to the Hessian.bin to skip the freq calculation
HESSIAN_FILE = None

SCHEDULER = False

########################################
# TDCI TeraChem Template
########################################

TIMESTEP_E_AS = 0.10           # Electronic timestep in attoseconds
nfields = 1              # number of distinct fields (generally for multichromatic floquet)
krylov_end = False       # Generate approximate eigenstates at end of calculation?
krylov_end_n = 5         # Number of steps to save wfn on to generate approx eigenstates with.
                         #   There will be 2*krylov_end_n approximate eigenstates returned.
krylov_end_interval = 20 # Number of steps between saved steps.
TDCI_TEMPLATE = {
  "gpus"                 : "1 0",
  "precision"            : "double",
  "threall"              : "1.0e-18",
  "convthre"             : "5.0e-10",
  "dciconvtol"           : "6.0e-13",
  "cphftol"              : "1.0e-10",
  #"basis"                : "6-311++g[2d,2p]",
  #"basis"                : "sto-3g",
  "basis"                : "3-21g",
  "method"               : "hf",
  "charge"               : "0",
  "spinmult"             : "1",
  "csf_basis"            : "no",
  "sphericalbasis"       : "no",
  "tdci_eshift"          : "gs",
  "tdci_stepprint"       : "1",
  "tdci_laser_freq"      : "0.0",
  "tdci_photoneng"       : "0.0",
  "tdci_fstrength"       : "0.0E+16",
  "tdci_fdirection"      : "x",
  "tdci_ftype"           : "cw",
  "tdci_corrfn_t"        : "p0",
  "tdci_write_field"     : "no",
  "tdci_floquet"         : "no",
  "tdci_floquet_photons" : "4",
  "cpcisiter"            : "350",
  "ci_solver"            : "direct",
  "dcimaxiter"           : "300",
  "dciprintinfo"         : "yes",
  "dcipreconditioner"    : "orbenergy",
  "closed"               : "18", # Ethylene 2/2
  "active"               : "6", 
  "cassinglets"          : "3",
  "castriplets"          : "0",
  "cascharges"           : "no", # Turning extra analyses off to save time
  "cas_ntos"             : "no",
  "cphfiter"             : "300", # MaxIters for cpXhf gradient calculation
  #"tdci_sanitytest"      : "no",

  # orbital options
  #"casci"                : "no",  # no if using CISNO or CASSCF, yes for FOMO or HF
  "fon"                  : "yes",
  "fon_method"           : "gaussian",
  "fon_temperature"      : "0.15",
  #"fon_mix"              : "no",
  #"cisno"                : "yes",
  #"cisnostates"          : "3",
  #"cisnumstates"         : "3",
  #"cisguessvecs"         : "3",
  #"cismaxiter"           : "500",
  #"cisconvtol"           : "1.0e-8",
  #"casscf"               : "yes",
  #"casweights"           : "[1,1,1]",
  #"dynamicweights"       : "yes" # dynamic state weighting in casscf
  #"casscfmaxiter"        : "300",   # Default: 50
  #"casscfnriter"         : "300",   # Default: 20 (max newton-raphson iterations)
  #"casscfconvthre"       : "1E-6",  # Default: 1E-6
  #"casscfenergyconvthre" : "1E-8",  # Default: 1E-8
  #"cpsacasscfmaxiter"    : "100",   # Default: 20
  #"cpsacasscfconvthre"   : "1E-4",  # Default: "1E-4"
  #"cpsacasscfsolver"     : "direct",

}

########################################
# External field
########################################

# Depending on the external field you want, you might have to write some
#   code here to generate the waveform, below is a CW tuned to ethylene
#   Function should accept np.arrays in units of AU time and return AU Electric field units.
#   This function then gets fed to tccontroller to generate the field on different TDCI steps
#  INPUT:  t is a numpy array
#  OUTPUT: out is a numpy array (same length as t)
import numpy as np
def f0_values(t):
  return 0*t # Turn off field completely.
  EPSILON_C = 0.00265316
  E_FIELD_AU = 5.142206707E+11
  HZtoAU = 2.418884E-17
  E_strength_Wm2 = 2.5E+16 # In W/m^2
  E_str = (np.sqrt(2.0*E_strength_Wm2 / EPSILON_C) )/E_FIELD_AU  # transform to au field units
  # Continuous field for Rabi flop
  # S0 <-> S1 gap at S0 geom:  0.28229097 au, 7.68152679 eV
  # S0 <-> S1 gap at S1 geom:  0.26018939 au, 7.08011240 eV
  # Need a range of frequencies to account for the changing gap
  field_au = np.linspace(0.26018939, 0.28229097, 50)
  #field_freq_hz = map(field_au, lambda x: (x/(2*np.pi*HZtoAU)) )
  out = 0.0*t
  for i in range(0,len(field_au)):
    out += np.sin(field_au[i]*t)
  return (E_str/len(field_au))*out





# Job Template: Used to make a shell script that executes terachem 
#   make sure you include temppath and tempname in your job template!
#   those keywords are search and replaced
job_template_contents = "#!/bin/bash\n\
source /home/adurden/.bashrc\n\
cd temppath\n\
"+TERACHEM+"terachem tempname.in > tempname.out\n"





















xyzpath = "h2o.xyz"
initial_electronic_state = 0  # S0 = 0, S1 = 1, etc.

RESTART = False
TERACHEM = "/home/adurden/terachem/build/bin/" # directory containing terachem executable

# Job Template: Used to make a shell script that executes terachem 
#   make sure you include temppath and tempname in your job template!
#   those keywords are search and replaced
job_template_contents = "#!/bin/bash\n\
                         source /home/adurden/.bashrc\n\
                         cd temppath\n\
                        "+TERACHEM+"terachem tempname.in > tempname.out\n"

TIMESTEP_AU = 5.0 # Nuclear timestep in Atomic Time units. 1 au_t ~= 24 attosecond

########################################
# TDCI TeraChem Template
########################################

NSTEPS_TDCI = 1000       # Number of electronic timesteps in one nuclear timestep.
                         #   Electronic timestep duration will be TIMESTEP_AU / NSTEPS_TDCI
nfields = 1              # number of distinct fields (generally for multichromatic floquet)
krylov_end = False       # Generate approximate eigenstates at end of calculation?
krylov_end_n = 6         # Number of steps to save wfn on to generate approx eigenstates with.
                         #   There will be 2*krylov_end_n approximate eigenstates returned.
krylov_end_interval = 20 # Number of steps between saved steps.
TDCI_TEMPLATE = {
  "gpus"                 : "1 0",
  "precision"            : "double",
  #"threall"              : "1.0e-20",
  #"convthre"             : "1.0e-6",
  #"basis"                : "6-311++g[2d,2p]",
  "basis"                : "sto-3g",
  "method"               : "hf",
  "charge"               : "0",
  "spinmult"             : "1",
  "csf_basis"            : "no",
  "sphericalbasis"       : "no", # sometimes this fixes norm problems, sometimes it causes them!
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
  #"cisno"                : "yes",
  #"cisnostates"          : "6",
  #"cisnumstates"         : "6",
  #"cisguessvecs"         : "8",
  #"cismaxiter"           : "500",
  #"cisconvtol"           : "1.0e-8",
  "cpcisiter"            : "350",
  "fon"                  : "no",
  "fon_method"           : "gaussian",
  "fon_temperature"      : "0.125",
  #"fon_mix"              : "no",
  #"casscf"               : "no",
  "casci"                : "yes",  # no if using CISNO or CASSCF
  "ci_solver"            : "direct",
  "dcimaxiter"           : "300",
  "dciprintinfo"         : "yes",
  "dcipreconditioner"    : "orbenergy",
  #"closed"               : "18",
  #"active"               : "6",
  #"closed"               : "6",   # Ethylene 4/4
  #"active"               : "4",
  #"closed"               : "4", # Ethylene 8/8
  #"active"               : "8", 
  #"closed"               : "7", # Ethylene 2/2
  #"active"               : "2", 
  "closed"               : "0", # H2O STO-3G full CI
  "active"               : "7",
  #"closed"               : "0", # H2O bigbasis
  #"active"               : "10",
  "cassinglets"          : "3",
  "castriplets"          : "0",
  "cascharges"           : "no", # Turning extra analyses off to save time
  "cas_ntos"             : "no",
  "cphfiter"             : "300", # MaxIters for cpXhf gradient calculation
  #"tdci_sanitytest"      : "no",

  # orbital options
  #"fon"                  : "yes",
  #"fon_method"           : "gaussian",
  #"fon_temperature"      : "0.25",

}

########################################
# External field
########################################

# Depending on the external field you want, you might have to write some
#   code here to generate the waveform, below is a CW tuned to ethylene
#   Function should accept np.arrays in units of AU time and return AU Electric field units.
#   This function then gets fed to tccontroller to generate the field on different TDCI steps
import numpy as np
def f0_values(t):
  return 0*t # Turn off field completely.
  EPSILON_C = 0.00265316
  E_FIELD_AU = 5.142206707E+11
  HZtoAU = 2.418884E-17
  E_strength_Wm2 = 2.5E+16 # In W/m^2
  E_str = (np.sqrt(2.0*E_strength_Wm2 / EPSILON_C) )/E_FIELD_AU  # transform to au field units
  # Continuous field for Rabi flop
  # S0 <-> S1 gap at S0 geom:  0.28229097  7.68152679 eV
  # S0 <-> S1 gap at S1 geom:  0.26018939 au, 7.08011240 eV
  # Need a range of frequencies to account for the changing gap
  field_au = np.linspace(0.26018939, 0.28229097, 50)
  #field_freq_hz = map(field_au, lambda x: (x/(2*np.pi*HZtoAU)) )
  out = 0.0*t
  for i in range(0,len(field_au)):
    out += np.sin(field_au[i]*t)
  return (E_str/len(field_au))*out





















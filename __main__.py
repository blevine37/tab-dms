#!/usr/bin/env python2.7
import sys, os, shutil
import tccontroller, utils
import ehrenfest
import tab

########################################
# Read input file
########################################

infile = None
# Is an input file specified as a parameter?
if len(sys.argv) > 1:
  if os.path.isfile(sys.argv[1]):
    infile = os.path.abspath(sys.argv[1])
# Is there a file named inputfile.py in the current directory?
if infile is None:
  if os.path.isfile("inputfile.py"):
    infile = os.path.abspath("inputfile.py")
# Default to using the example inputfile.py in the source code
if infile is None:
  print("No input file found in argument or cwd, using the inputfile.py in the source directory")
  infile = os.path.dirname(os.path.realpath(__file__)) + "/inputfile.py"
  if not os.path.isfile(infile): print("ERROR: Can't find input file!!")

########################################
# Initialize
########################################

#utils.clean_files()
l = tccontroller.logger()
# logprint(string) will write a timestamped string to the log, and to STDOUT
logprint = l.logprint

#import inputfile
input_module = utils.import_program_as_module(infile) # Hack for importing an arbitrary py file as a namespace
config = utils.ConfigHandler(input_module) # Include defaults

# Make sure we're not about to blow away data that already exists
if (os.path.exists(config.JOBDIR+"/electronic") and config.RESTART==False):
  logprint(config.JOBDIR+'/electronic already exists, but no RESTART requested? Resolve this before running.')
  sys.exit()

if os.path.abspath(config.JOBDIR)+"/inputfile.py" != infile:
  shutil.copy(infile, os.path.abspath(config.JOBDIR)+"/inputfile.py")

tc = tccontroller.tccontroller(config, logger=l)


########################################
# Announce and Run Dynamics!
########################################
# Print header
logprint("TDCI + TAB-DMS")

# Select propagation scheme (Ehrenfest, TAB)
if config.TAB == 0:
  ehrenfest_ = ehrenfest.Ehrenfest(config.TIMESTEP_AU, logprint, tc)
  logprint("Propagation scheme: Ehrenfest")
elif config.TAB == 1:
  ehrenfest_ = tab.TAB(config.TIMESTEP_AU, logprint, tc)
  if config.krylov_end == False:
    logprint("Propagation scheme: TAB")
  else:
    logprint("Propagation scheme: TAB-DMS")
else:
  print("ERROR: Choose valid propagation scheme!")
  sys.exit()

#   get commit number
#srcpath = os.path.dirname(os.path.realpath(__file__))
#commit = ""
#with open( srcpath+"/.git/refs/heads/main", 'r') as f:
  #commit = (f.read()).strip()[:8]

#logprint("Rev: "+str(commit))

if config.WIGNER_PERTURB:
  logprint("Wigner Random Seed: "+str(config.WIGNER_SEED))

logprint("=================")

# Time steps
logprint("Propagation time step in au: " + str(config.TIMESTEP_AU))
logprint("TDCI simulation time step in fs: " + str(config.tdci_simulation_time))
logprint("")

# Do Ehrenfest dynamics!
ehrenfest_.run_ehrenfest()



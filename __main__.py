#!/usr/bin/env python2.7
import sys, os, shutil
import tccontroller, utils
import ehrenfest

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
input_module = utils.import_program_as_module(infile)
config = utils.ConfigHandler(input_module)

if os.path.abspath(config.JOBDIR)+"/inputfile.py" != infile:
  shutil.copy(infile, os.path.abspath(config.JOBDIR)+"/inputfile.py")

tc = tccontroller.tccontroller(config, logger=l)

ehrenfest_ = ehrenfest.Ehrenfest(config.TIMESTEP_AU, logprint, tc)



########################################
# Announce and Run Dynamics!
########################################
# Print header
logprint("TDCI + TAB-DMS")
logprint("=================")

# Time steps
logprint("Propagation time step in au: " + str(config.TIMESTEP_AU))
logprint("TDCI simulation time step in fs: " + str(config.tdci_simulation_time))
logprint("")

# Do Ehrenfest dynamics!
ehrenfest_.run_ehrenfest()



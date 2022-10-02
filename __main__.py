
import tccontroller, utils
import ehrenfest
########################################
# Main function
########################################

utils.clean_files()
l = tccontroller.logger()
# logprint(string) will write a timestamped string to the log, and to STDOUT
logprint = l.logprint

import inputfile
config = utils.ConfigHandler(inputfile)

tc = tccontroller.tccontroller(config, logger=l)

ehrenfest_ = ehrenfest.Ehrenfest(config.TIMESTEP_AU, logprint, tc)



# Print header
logprint("TDCI + TAB-DMS")
logprint("=================")

# Time steps
logprint("Propagation time step in au: " + str(config.TIMESTEP_AU))
logprint("TDCI simulation half time step in fs: " + str(config.tdci_simulation_time))
logprint("")

# Do Ehrenfest dynamics!
ehrenfest_.run_ehrenfest()



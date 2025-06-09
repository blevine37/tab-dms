######################################################
# TAB code for TDCI
#
######################################################

import tccontroller
import ehrenfest
import numpy as np
import shutil, os, subprocess, time
import h5py
import utils
import math
import random

from copy import deepcopy

########################################
# Constants
########################################

# Atomic unit of length (Bohr radius, a_0) to angstrom (A)
bohrtoangs = 0.529177210903
# Atomic unit of time to seconds (s)
autimetosec = 2.4188843265857e-17


class TAB(ehrenfest.Ehrenfest):
  def propagate(self, x_init, v_init, t_init, ReCn_init, ImCn_init=None):
    realtime_start = time.time()  # For benchmarking
    it = int(t_init/(self.delta*autimetosec*1e+18))
    t = t_init
    x, v, ReCn, ImCn = x_init, v_init, ReCn_init, ImCn_init
    a = 0.0 # initial acceleration is not used
    TCdata = None
    #dcps = 6
    dcps = utils.getdcps(self.atoms)
    self.logprint("dcps are "+str(dcps))
    if self.tc.config.RESTART:
      gradout = self.tc.grad(x*bohrtoangs, ReCn, ImCn, DoGradStates=True) #initial grad call for restart
      grad_select3 = [i for i in range(len(gradout["states"]))] 
    while it < self.tc.config.MAXITERS: # go forever! :D
      t += self.delta * autimetosec * 1e+18 # Time in Attoseconds
      x_prev, v_prev, ReCn_prev, ImCn_prev, TCdata_prev = x, v, ReCn, ImCn, TCdata
      
      if it == 0:   #initial grad call
          gradout_int = self.tc.grad(x*bohrtoangs, ReCn, ImCn, DoGradStates=True)
          grad_select = [i for i in range(len(gradout_int["states"]))] #all gradients were calculated 
      else:         #reuse the data from end-of-the-loop call
          gradout_int = deepcopy(gradout)
          grad_select = deepcopy(grad_select3)

      ######store the old population to get its derivatives
      states = gradout_int["states"]
      self.logprint("There are "+str(len(states))+" states")
      ReCn, ImCn = gradout_int["recn"], gradout_int["imcn"]
      
      oldpop = ((np.dot(ReCn, np.transpose(states)))**2 + (np.dot(ImCn, np.transpose(states)))**2).real # ~adiabatic populations in "states"-basis
      self.logprint("Population before ehrenfest is "+str(oldpop))
      #oldpote = gradout_int["eng"]
      #oldkine = self.ke_calc(v)  #Kinetic energy before Ehrenfest
      
      #### Ehrenfest-TDCI propagation ######################
      x, v_timestep, v, a, stepnorm, steppop, TCdata = self.step(x, v, ReCn=ReCn, ImCn=ImCn) # Do propagation step
      
      oldpote = TCdata["eng"] 
      oldkine = self.ke_calc(v)  #Kinetic energy after Ehrenfest

      ReCn, ImCn = TCdata["recn"], TCdata["imcn"]   # New Ehrenfest wavefunctions
      ######################################################
      #TAB Gegins here - history based correction###########
      ######################################################
      
      ####------------TAB-parameters--------------#####
      dimH = len(states)  #number of states
      deltatn = self.delta #classical time-step
      
      nzthresh = 1.0e-10 # Threshold for considering numbers numerically zero
      errortol = 1.0e-6 #Tolerance for cumulative errors in collapse probabilities
      npthresh = 1.0e-7 #Tolerance for negative probabilities
      pehrptol = 1.0e-5 #If Ehrenfest matrix represents rho less than this deviation from one, skip the numerical optimization
      tolodotrho = 1.0e-5
      nta = 200 #steps of integration over simulation history
      dtw = 0.10 #discrete time step for integrating over simulation history
      zpop = 1.0e-6
      dgscale = 1.0e+5
      tabdecay = self.tc.config.TABdecay  #exp or gauss
     
      #States populated after diabatization, if pop>zpop calculate forces
      states = TCdata["states"]
      steppop = ((np.dot(ReCn, np.transpose(states)))**2 + (np.dot(ImCn, np.transpose(states)))**2).real

      self.logprint("Population after ehrenfest is "+str(steppop))

      if self.tc.config.GRADSELECT: 
        grad_select2=[]
        for i in range(len(states)):
          if (steppop[i] >= zpop*0.5):   #0.5 factor acts as a buffer since the state can cross the threshold by diabatization in the next step
            grad_select2.append(i)
      else:
        grad_select2 = grad_select

      #Get gradients x+dx
      gradout_mid = self.tc.grad(x*bohrtoangs, ReCn, ImCn, DoGradStates=True, GradStatesSelect=grad_select2)        ###everthing after progpagation before collapsing gradout_mid
      ReCn, ImCn = gradout_mid["recn"], gradout_mid["imcn"]
      states = gradout_mid["states"]
      self.logprint("There are "+str(len(states))+" states")
      ReCn_states = np.dot(ReCn,np.transpose(states))	# coefficient in the ~adiabatic basis('states' basis)
      ImCn_states = np.dot(ImCn,np.transpose(states))
      newpop = ReCn_states**2.0.real + ImCn_states**2.0.real
      self.logprint("There are "+str(len(gradout_int["forces"]))+ " initial forces")
      self.logprint("There are "+str(len(gradout_mid["forces"]))+ " mid forces")
      aforces = (np.array(gradout_mid["forces"])+np.array(gradout_int["forces"]))/2
      self.logprint("There are "+str(len(aforces))+" state forces")

      #If state becomes populated above zpop we need its gradient from the last step for averaging forces
      if self.tc.config.GRADSELECT:
        newlypopulated = [i for i in grad_select2 if i not in grad_select and steppop[i] >= zpop]       
        if newlypopulated:
          for i in newlypopulated:
            print i,' became suddenly populated - recalculating x(t-dt) forces'
          #recalculate force on the previous geometry in separate folder
          gradout_temp = self.tc.tempgrad(x_prev*bohrtoangs, ReCn_prev, ImCn_prev, GradStatesSelect=newlypopulated)
          for i in newlypopulated:
            aforces[i] = (np.array(gradout_mid["forces"][i])+np.array(gradout_temp["forces"][i]))/2 
      
      
      #########---For the restoration of wavefunctions--------#######
      ampdir = np.zeros((dimH),dtype=np.complex)	#Stores amplitude directions for each state
      amp = np.zeros((dimH),dtype=np.complex)		#Stores amplitudes for each state
      
      sw, sVR = gradout_mid["states_eng"], states
      
      temp1 = np.zeros((1),dtype=complex)
      
      norm2ct = sum(newpop)
     
      i = 0
      while i < dimH:
      	amp[i] = (ReCn_states[i]+1j*ImCn_states[i])/norm2ct
      	
      	if (newpop[i] == 0):
      	  ampdir[i] = 1.0
      	else:
      	  ampdir[i] = amp[i]/(newpop[i]**0.50)
      	pass
      	i = i + 1
      pass	
      
      ##################################################
      odotrho = (newpop - oldpop)/deltatn  #derivative of rho
       
      ###---Collapsing-------###########################
      npop, track = self.gcollapse(dimH,deltatn,aforces,newpop,dcps,nzthresh,errortol,npthresh,pehrptol,odotrho,tolodotrho,nta,dtw,zpop,dgscale,tabdecay)
      self.logprint("Population after collapsing is "+str(npop))
      if (track == 0):
        self.logprint("No collapsing at "+str(it)+" time step")
        gradout = gradout_mid
        grad_select3 = grad_select2
      else:
        ###---Restoring the wavefunctions--------#########
        self.logprint("TAB collapsed at"+str(it)+" time step")
        poparray = npop
        namp = np.zeros((dimH),dtype=complex)
        nct = np.zeros((dimH,1),dtype=complex)
      
        i = 0
        while i < dimH:
      	  namp[i] = ampdir[i]*(npop[i]**0.5)*norm2ct
      	  i = i + 1
        pass
      
        nct = np.dot(namp,sVR)
      
        ReCn = nct.real
        ImCn = nct.imag

        #States populated after collapse, is there a newly poopulated state? 
        if self.tc.config.GRADSELECT:
          grad_select3=[]
          for i in range(len(states)):
            if (npop[i] >= zpop*0.5):
              grad_select3.append(i)
          newlypopulated = [i for i in grad_select3 if i not in grad_select2]
          for i in newlypopulated:
            print i,' became populated after collapse'
          if len(newlypopulated) == 0:
              gradout = self.tc.grad(x*bohrtoangs,ReCn,ImCn,DoGradStates=False)
              gradout["forces"]=gradout_mid["forces"] #Reuse state gradients (geometry is unchanged)
              grad_select3 = [i for i in grad_select2] 
          else:
              gradout = self.tc.grad(x*bohrtoangs,ReCn,ImCn,DoGradStates=True,GradStatesSelect=newlypopulated)
              for i in grad_select2:
                gradout["forces"][i] = gradout_mid["forces"][i]
              grad_select3 = grad_select2 + newlypopulated #concatenate arrays
        else:
          #After collapse calculation
          gradout = self.tc.grad(x*bohrtoangs,ReCn,ImCn,DoGradStates=False)
          gradout["forces"]=gradout_mid["forces"]
          grad_select3 = grad_select2
        
        ##-----------Resclaing the Momentum to conseve total energy)----------#
        newpote = gradout["eng"]
      
        if (newpote > oldpote+oldkine):
      	  self.logprint("Not enough energy, jump back")
      	  ##Reverse momentum########---------
          v = -v
          #Restore WF
          ReCn, ImCn = gradout_mid["recn"], gradout_mid["imcn"]
          gradout = gradout_mid
          grad_select3 = grad_select2
          newpote = gradout["eng"]
        else:
      	  newkine = oldpote + oldkine - newpote
      	  scale = (newkine/oldkine)**0.50
          self.logprint("Kinetic energy is lifted by "+str(newkine-oldkine))
      	  ##Rescale the momentum and update the wave function##
          v *= scale 
      	  TCdata["recn"] = ReCn
      	  TCdata["imcn"] = ImCn
      norm, pop = self.getNormPop(TCdata["states"], TCdata["recn"], TCdata["imcn"])
      self.savestate(x, v_timestep, v, a, norm, pop, t, TCdata)
      self.logprint("Iteration " + str(it).zfill(4) + " finished")
      it+=1
    self.logprint("Completed TAB Propagation!")
    time_simulated = (t-t_init)/1000.
    import datetime
    realtime = str( datetime.timedelta( seconds=(time.time() - realtime_start) ))
    self.logprint("Simulated "+str(time_simulated)+" fs with "+str(it)+" steps in "+realtime+" Real time.")
  def gcollapse(self,dimH,deltatn,aforces,poparray,dcps,nzthresh,errortol,npthresh,pehrptol,odotrho,tolodotrho,nta,dtw,zpop,dgscale,tabdecay):
        
        import sys
        from scipy.optimize import lsq_linear
        import math
        # Rules ========================================================
	# i, j, k, l, m, and n are all reserved for integer incrementing
	
        #tseed=time.time()
        #print('Seed:',tseed)
	#random.seed(tseed)     #Make it as input parameter

	# General setup ================================================
	npop = np.zeros((dimH)) 	# Stores output electronic populations

	invtau = np.zeros((dimH,dimH)) 	# Stores inverse of state-pairwise decoherence times

	i = 0
	while i < dimH:
		j = i + 1
		while j < dimH:
                        k = 0 #iterate through atoms/degrees of freedom
                        while k < len(dcps):
				invtau[i][j] += ((np.sum(abs(aforces[i][3*k]-aforces[j][3*k])**2.0+abs(aforces[i][3*k+1]-aforces[j][3*k+1])**2.0+abs(aforces[i][3*k+2]-aforces[j][3*k+2])**2.0))/(8.0*dcps[k]))**0.50
                                k = k + 1
			#invtau[i][j] = (np.sum(abs(aforces[i]-aforces[j])**2.0)/(8.0*dcp))**0.50
			invtau[j][i] = invtau[i][j]
			j = j + 1
		i = i + 1
	pass
        self.logprint("invtau is "+str(invtau))

	# computing the weight function
	if tabdecay == "gauss":
	  w = np.zeros((dimH,nta+1))
	  zpoptime = np.zeros((dimH))

	  i = 0
	  while i < dimH:
		if (odotrho[i] >= tolodotrho):
			zpoptime[i] = poparray[i]/odotrho[i]
			k = 0
			sum = 0.0
			while k < nta:
				if ((sum + odotrho[i]*dtw) <= poparray[i]):
					w[i][k] = odotrho[i]*dtw/poparray[i]
				else: 
					if (sum < poparray[i]):
						w[i][k] = (poparray[i] - sum)/poparray[i]
					else:
						w[i][k] = 0.0
					pass
				pass
				sum = sum + odotrho[i]*dtw
				k = k + 1
			pass
			if (sum < poparray[i]):
				w[i][-1] = (poparray[i] - sum)/poparray[i]
			pass
		i = i + 1

#	Not super efficient, could put sooner, but here is the rank reduction 
#	to working with only populated electronic states -> add a tolerance to pass
#	from namd-main.py

	vstates = []

	i = 0
	while i < dimH:
		if (poparray[i] >= zpop):
			vstates.append(i)
		i = i + 1
	pass

	rank = len(vstates)

#	print 'odotrho'
#	print odotrho
#	print 'w'
#	print w

#	sys.exit()
	# constructing vectorized target density matrix
	# This target is rho**(d) in the original python manuscript
	# However it is organized into a column vector of the diagonal and
	# lower triangular elements so it can be used as the target in a 
	# library linear least squares optimization

	eseg = np.zeros((rank,rank))
	nblock = 0

	vtarget = []
	i = 0
	if tabdecay == "gauss":
	  while i < rank:
	        j = i
	        while j < rank:
# lots of logic in order to determine how integration will work based on current rules
			if (i == j):
				velem = dgscale*poparray[vstates[i]]
				eseg[i][i] = 1.0
			elif (odotrho[vstates[i]] < tolodotrho and odotrho[vstates[j]] < tolodotrho):
		                velem = ((poparray[vstates[i]]*poparray[vstates[j]])**(0.5))*(1.0+((math.exp(-1.0*((deltatn+dtw*nta)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])-math.exp(-1.0*((dtw*nta)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]]))/math.exp(-1.0*((dtw*nta)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])))
				eseg[i][j] = (1.0+((math.exp(-1.0*((deltatn+dtw*nta)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])-math.exp(-1.0*((dtw*nta)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]]))/math.exp(-1.0*((dtw*nta)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])))
				eseg[j][i] = eseg[i][j]
			elif (odotrho[vstates[i]] >= tolodotrho and odotrho[vstates[j]] < tolodotrho):
				k = 0
				nsum = 0.0
				dsum = 0.0
				while k <= nta:
					nsum = nsum + w[vstates[i]][k]*(math.exp(-1.0*((deltatn+dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])-math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]]))
					dsum = dsum + w[vstates[i]][k]*math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])
					k = k + 1
				velem = ((poparray[vstates[i]]*poparray[vstates[j]])**(0.5))*(1.0+nsum/dsum)
				eseg[i][j] = (1.0+nsum/dsum)
				eseg[j][i] = eseg[i][j]
			elif (odotrho[vstates[j]] >= tolodotrho and odotrho[vstates[i]] < tolodotrho):
                                k = 0
                                nsum = 0.0
                                dsum = 0.0
                                while k <= nta:
                                        nsum = nsum + w[vstates[j]][k]*(math.exp(-1.0*((deltatn+dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])-math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]]))
                                        dsum = dsum + w[vstates[j]][k]*math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])
                                        k = k + 1
                                velem = ((poparray[vstates[i]]*poparray[vstates[j]])**(0.5))*(1.0+nsum/dsum)
				eseg[i][j] = (1.0+nsum/dsum)
				eseg[j][i] = eseg[i][j]
			elif (odotrho[vstates[i]] >= tolodotrho and odotrho[vstates[j]] >= tolodotrho): # clean up notation and just use derivatives
				if (zpoptime[vstates[i]] > zpoptime[vstates[j]]):
	                                k = 0
	                                nsum = 0.0
	                                dsum = 0.0
	                                while k <= nta:
	                                        nsum = nsum + w[vstates[j]][k]*(math.exp(-1.0*((deltatn+dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])-math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]]))
	                                        dsum = dsum + w[vstates[j]][k]*math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])
	                                        k = k + 1
	                                velem = ((poparray[vstates[i]]*poparray[vstates[j]])**(0.5))*(1.0+nsum/dsum)
					eseg[i][j] = (1.0+nsum/dsum)
					eseg[j][i] = eseg[i][j]
				else:
                                        k = 0
                                        nsum = 0.0
                                        dsum = 0.0
                                        while k <= nta:
	                                        nsum = nsum + w[vstates[i]][k]*(math.exp(-1.0*((deltatn+dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])-math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]]))
	                                        dsum = dsum + w[vstates[i]][k]*math.exp(-1.0*((dtw*k)**(2.0))*invtau[vstates[i]][vstates[j]]*invtau[vstates[i]][vstates[j]])
	                                        k = k + 1
	                                velem = ((poparray[vstates[i]]*poparray[vstates[j]])**(0.5))*(1.0+nsum/dsum)
					eseg[i][j] = (1.0+nsum/dsum)
					eseg[j][i] = eseg[i][j]
			else:
				print 'something went horribly wrong'
				print 'collapse routine missed a case, check septab.py'
				sys.exit()
			pass
	                vtarget.append(velem)
	                j = j + 1
	        pass
	        i = i + 1
	elif tabdecay == "exp":
	  while i < rank:
	        j = i
	        while j < rank:
			  if (i == j):
				velem = dgscale*poparray[vstates[i]]
				eseg[i][i] = 1.0
			  else:
				velem = ((poparray[vstates[i]]*poparray[vstates[j]])**(0.5))*math.exp(-1.0*deltatn*invtau[vstates[i]][vstates[j]])
				eseg[i][j] = math.exp(-1.0*deltatn*invtau[i][j]) 
				eseg[j][i] = eseg[i][j]
			  pass
			  vtarget.append(velem)
			  j = j + 1
	        pass
	        i = i + 1


#	print 'vtarget'
#	print vtarget

#	print 'len vtarget'
#	print len(vtarget)

#	sys.exit()

        iter = 0        # Used to track what column is sent to minelem
        bcore = []      # block coordinates for fastest decaying elements
        listbank = []

        p, q, leave = self.minelem(eseg,rank,nzthresh)

        bcore.append(p)
        bcore.append(q)

#       print 'bcore'
#       print bcore

#       print 'pre-algorithm check before individual element stuff is made'
#       sys.exit()

        while leave == 'no':

                # Finding the largest possible coherent sub-block
                i = 0
                bstates = []
                while i < rank:         # starting by adding all possible states
                        bstates.append(i)
                        i = i + 1
                pass

                # checking for zeroes in eseg such that blocks can be removed from the coherent block
                icstates = []   # list of states with zero elemnts in rows or columns shared by the
                                # coherence indicated by bcore

                if (nblock > 0):
                        icstates = self.rstates(rank,eseg,bcore,nzthresh)    # function identifying the icoherent states
                pass

#               print 'icstates', icstates

                i = 0
                if (nblock == 0):
                        pass
                else:
#                       print 'is this getting killed by poor loop navigation?'
                        while i < len(icstates):
                                j = -(i + 1)
                                temp = bstates.pop(icstates[j])
                                i = i + 1
                        pass
                pass

                listbank.append(bstates[:])

#               print 'bstates for current block', bstates

                # projecting out the current block from eseg
                iter = len(bstates)
                i = 0
                temp4 = eseg[bcore[0]][bcore[1]]
                while i < iter:
                        j = 0
                        while j < iter:
                                eseg[bstates[i]][bstates[j]] = eseg[bstates[i]][bstates[j]] - temp4
                                j = j + 1
                        pass
                        i = i + 1
                pass

#               print 'updated eseg'
#               print eseg

                nblock = nblock + 1
                if (nblock > 500):
                        print 'nblock blew up'
                        print 'Pnc matrix'
#                        print Pnc
                        print 'current eseg'
                        print eseg
                        sys.exit()
                pass

                bcore = []      # block coordinates for fastest decaying elements

                p, q, leave = self.minelem(eseg,rank,nzthresh)

                bcore.append(p)
                bcore.append(q)

        pass

        i = 0
        while i < rank:
                if (abs(poparray[vstates[i]]*eseg[i][i]) > nzthresh):
                        listbank.append([])
                        listbank[-1].append(i)
                pass
                i = i + 1
        pass

#	print 'listbank'
#	print listbank

	# generating the vectorized set of coherent block density matrices
	# for the TAB wave function collapse step
	# These are determined using the current time step electronic populations,
	# and therefore must be recomputed at each collapse step. (Unlike the list of 
	# states populated in each block [listbank])

	A = [] 	#stores the block density matrices in form of A[block-ID][element]
	aindex = 0

        i = 0
        while i < len(listbank):
                k = 0
                A.append([])
                while k < rank:
                        l = k
                        while l < rank:
                                if (k in listbank[i] and l in listbank[i]):
                                        m = 0
                                        popsum = 0.0
                                        while m < len(listbank[i]):
                                                popsum = popsum + poparray[vstates[listbank[i][m]]]
                                                m = m + 1
                                        pass
                                        if (popsum <= nzthresh):
                                                aelem = 0.0
                                        else:
                                                if (k == l):
                                                        aelem = dgscale*((poparray[vstates[k]]*poparray[vstates[l]])**(0.5))/popsum
                                                else:
                                                        aelem = ((poparray[vstates[k]]*poparray[vstates[l]])**(0.5))/popsum
                                                pass
                                        pass
                                else:
                                        aelem = 0.0
                                A[aindex].append(aelem)
                                l = l + 1
                        pass
                        k = k + 1
                pass
                i = i + 1
                aindex = aindex + 1
        pass

#	print 'len(A)'
#	print len(A)

#	print A

#	Test to see if there is linear dependence, and if not will do the real
#	linear least squares optimization
	At = np.transpose(A)

#	print 'len At'
#	print len(At)

#	ehrv = []
#	ehrv.append([])

#	i = 0
#	while i < len(A[-1]):
#		ehrv[0].append(A[-1][i])
#		i = i + 1
#	pass

#	ehrvt = np.transpose(ehrv)
#	ehrw = lsq_linear(ehrvt, vtarget, bounds=(0.0, 2.0), method='bvls')

#	if (abs(ehrw.x[0]-1.0) <= pehrptol):
#		i = 0
#		return poparray
#	pass


#	print 'A[dimH-1]'
#	print A[dimH-1]

	# scipy linear least squares optimization
#	At = np.transpose(A)
	optw = lsq_linear(At, vtarget, bounds=(0.0, 2.0), method='bvls')

	# Error analysis of the linear least squares target wave function



	# Collapsing the wave function

	i = 0
	ptotal = 0.0
	while i < len(optw.x):
		ptotal = ptotal + optw.x[i]
		i = i + 1
	pass

#	print 'optw'
#	print optw.x

	check2 = random.random()
	check = check2*ptotal

	track = 0
	psum = 0.0
	i = 0
	j = 0
	while j < 2:
		psum = psum + optw.x[i]
		if (check <= psum):
			j = 3
			track = i
		else:
			i = i + 1
		pass
	pass

	# track is the collapsed into density matrix according to A
	# constructing npop from the density matrix

	if (track == 0):
		return poparray, track
	pass

	k = 0
	index = 0
	while k < rank:
		l = k
		while l < rank:
			if (k == l):
				npop[vstates[k]] = A[track][index]/dgscale
			index = index + 1
			l = l + 1
		k = k + 1
	pass

#	print 'npop', npop
#	print 'vectorized target'
#	print A[track]

#	print 'early stop'
#	sys.exit()

	return npop, track 
  
  
  
  
  def minelem (self,eseg,dimH,nzthresh): # For a given sub-block of the density matrix
	"""Returns the element with fastest coherence loss"""

	# Standard library imports ====================================
	import numpy as np
	import sys

	leave = 'no'

	i = 0

	p = 0 	# first coordinate of minimum element of eseg
	q = 0 	# second coordinate of min. element

	check = 1.1 	# placeholder value for scanning for minimum element of eseg
	while i < dimH - 1:
		j = i + 1
		while j < dimH:
			if (abs(eseg[i][j]) <= nzthresh):
				pass
			else:
				if (check > eseg[i][j]):
					check = eseg[i][j]
					p = i
					q = j
				pass
			pass
			j = j + 1
		pass
		i = i + 1
	pass

	if (p == q):
		leave = 'yes'
	pass

	return p, q, leave

  def rstates(self,dimH,eseg,bcore,nzthresh):
	import numpy as np
	import sys

	icstates = [] 	# list of unique states that have been found to have 0 elements
			# in eseg in line with the current bcore element

	p = bcore[0]
	q = bcore[1]
	sum = []

	# Though horribly inefficient, repeating the original algorithm to avoid errors, then appending this
	        # loop over the row (changes column index)
        rowstates = []
        j = 0
        while j < dimH:
                if (j != p):
                        if (eseg[p][j] <= nzthresh):
                                rowstates.append(j)
                        pass
                pass
                j = j + 1
        pass

        # loop over the column (changes row index)
        colstates = []
        i = 0
        while i < dimH:
                if (i != q):
                        if (eseg[i][q] <= nzthresh):
                                colstates.append(i)
                        pass
                pass
                i = i + 1
        pass

        sum = rowstates + colstates



	i = 0
	while i < dimH -1:
		j = i + 1
		while j < dimH:
			if (eseg[i][j] <= nzthresh):
				if (i == p or i == q):
					if (j != p and j != q):
						sum.append(j)
					pass
				else:
					if (j == p or j == q):
						sum.append(i)
					else:
						k = 0
						flag = 'none'
						check2 = len(sum)
						while k < check2:
							if (i == sum[k]):
								flag = 'i'
							pass
							if (j == sum[k]):
								flag = 'j'
							k = k + 1
						pass
						if (flag == 'none'):	
							check = random.random()
							if (check <= 0.5):
								sum.append(i)
							else:
								sum.append(j)
							pass
						elif (flag == 'i'):
							sum.append(i)
						elif (flag == 'j'):
							sum.append(j)
						pass
					pass
				pass
			pass
			j = j + 1
		pass
		i = i + 1
	pass

	temp3 = sum.sort()
	usum = []

	usum.append(sum[0])
	i = 1
	while i < len(sum):
		j = 0
		add = 0
		while j < len(usum):
			if (usum[j] == sum[i]):
				add = 1
			pass
			j = j + 1
		pass
		if (add == 0):
			usum.append(sum[i])
		pass
		i = i + 1
	pass

#	print 'sum in sremove'
#	print sum
#	print 'usum'
#	print usum
	icstates = usum[:]

#	print 'icstates in sremove'
#	print icstates

	return icstates

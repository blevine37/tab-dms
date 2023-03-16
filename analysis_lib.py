#!/usr/bin/python2.7


import numpy as np
import math
import sys, struct
import os, time, shutil, subprocess
import copy
import h5py
import matplotlib


import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


c=2.99792458*10**10 # in cm/s
h=4.1357*10**-15 # in eV*s
bohrtoangs = 0.529177210903


rms = lambda x_seq: (sum(x*x for x in x_seq)/len(x_seq))**(1/2)

# we may need to be careful about endian-ness if this runs on a different machine than TC
def read_bin_array(filepath, length):
  #print("l: "+str(length))
  if length == 0:
    return np.array([])
  f = open(filepath, 'rb')
  return np.array(struct.unpack('d'*length, f.read()))

def write_bin_array(array, filepath):
  if (len(np.shape(array)) != 1):
    print("WARNING: write_bin_array() is for 1D double arrays! Your array has shape:")
    print(np.shape(array))
  f = open(filepath, 'wb')
  f.write( struct.pack('d'*len(array), *array ))
  f.close(); del f

# quick and dirty way to get values from a tc output file
def scan_outfile(filed, key, pos):
  f = open(filed, 'r')
  l = f.readline()
  while l != "":
    if (len(l.split()) > len(key)):
      if (l.split()[:len(key)] == key):
        return l.split()[pos]
    l = f.readline()
  return None

# get fomo orbital energies and occupations
def scan_fomo(filed):
  output = []
  f = open(filed, 'r')
  l = f.readline()
  while l != "":
    #print(l.split())
    if l.split() != ["Orbital","Energy","Occupation"]:
      l = f.readline()
    else: break
  # l currently is header
  l = f.readline()
  l = f.readline()
  while len(l.split()) == 3:
    temp = l.split()
    #print((l, temp))
    temp[0] = int(temp[0])
    temp[1] = float(temp[1])
    temp[2] = float(temp[2])
    output.append(temp)
    l = f.readline()
  return output
    
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


def molden_orbs(filename):
  f = open(filename, 'r')
  orbs = []
  orb = []
  lines = []
  MO_starts = []
  for line in f:
    lines.append(line)
  for i in range(0,len(lines)):
    if len(lines[i].split()) > 0:
      if lines[i].split()[0] == "Occup=": MO_starts.append(i)
  for MO_start in MO_starts[:-1]:
    i = 1
    while lines[MO_start+i].split()[0] != "Ene=":
      orb.append(float(lines[MO_start+i].split()[1]))
      i+=1
    orbs.append(orb[:])
    orb = []
  i=1
  MO_start = MO_starts[-1]
  while MO_start+i < len(lines):
    if len(lines[MO_start+i].split()) == 2:
      orb.append(float(lines[MO_start+i].split()[1]))
    i+=1
  orbs.append(orb[:])
  orb = []
  return orbs

# Check if a float is in a list
def float_in(flt, l):
    delta = 1e-4
    for i in l:
        if np.abs(flt-i) < delta: return True
    return False

# acti_ - length of matrix (all indices must be same length)
# Uoo - rotation matrix (2-d array)
# Vtuvw - 4-index tensor to rotate (1-d array)
# This code was previously used to rotate 2-electron integrals, but we
#   scrapped this approach and rotate the CI vector instead.
def rotate_4index(acti_, Uoo, Vtuvw):

  # Rotate 2-electron integrals with Uoo
  # (i'j|kl) = Sum_i Uii' (ij|kl)
  # (i'j'|kl) = Sum_j Ujj' (i'j|kl)
  # (i'j'|k'l) = Sum_k Ukk' (i'j'|kl)
  # (i'j'|k'l') = Sum_l Ull' (i'j'|k'l)
  tempVtuvw = np.zeros( (acti_,acti_,acti_,acti_) )
  tempVtuvw2 = np.zeros( (acti_,acti_,acti_,acti_) )
  # 1. (i'j|kl) = Sum_i Uii' (ij|kl)
  for i_ in range(0, acti_):
    for j in range(0, acti_):
      for k in range(0, acti_):
        for l in range(0, acti_):
          tempVtuvw[i_][j][k][l] = 0;
          for i in range(0, acti_):
            tempVtuvw[i_][j][k][l] += Uoo[i][i_] * Vtuvw[i][j][k][l]


  # 2. (i'j'|kl) = Sum_j Ujj' (i'j|kl)
  for i_ in range(0, acti_):
    for j_ in range(0, acti_):
      for k in range(0, acti_):
        for l in range(0, acti_):
          tempVtuvw2[i_][j_][k][l] = 0;
          for j in range(0, acti_):
            tempVtuvw2[i_][j_][k][l] += Uoo[j][j_] * tempVtuvw[i_][j][k][l]


  # 3. (i'j'|k'l) = Sum_k Ukk' (i'j'|kl)
  for i_ in range(0, acti_):
    for j_ in range(0, acti_):
      for k_ in range(0, acti_):
        for l in range(0, acti_):
          tempVtuvw[i_][j_][k_][l] = 0;
          for k in range(0, acti_):
            tempVtuvw[i_][j_][k_][l] += Uoo[k][k_] * tempVtuvw2[i_][j_][k][l]


  # 4. (i'j'|k'l') = Sum_l Ull' (i'j'|k'l)
  for i_ in range(0, acti_):
    for j_ in range(0, acti_):
      for k_ in range(0, acti_):
        for l_ in range(0, acti_):
          tempVtuvw2[i_][j_][k_][l_] = 0;
          for l in range(0, acti_):
            tempVtuvw2[i_][j_][k_][l_] += Uoo[l][l_] * tempVtuvw[i_][j_][k_][l]
  return tempVtuvw2




rms = lambda x_seq: (sum(x*x for x in x_seq)/len(x_seq))**(1/2)

# Design question: How much stuff should we store in hdf5?
#                  So far we've kept stuff stored in hdf5 to a minimum.
#                  Overlap arrays that are used mainly as diagnostics?
#
# TODO: Rewrite this so that we always use the explicit time-stamp and directory
#           labeled by hdf5 in case of half-steps midway through.
#       Technically, half-steps would only be used if we did the FOMO FIX method of
#         skipping a gradient at a problematic xyz... but afaik that code is bugged
#         do we care about fixing it? (AD: 3/15/2023)
# TODO: make sure terachem's States_Cn.bin includes higher spin states and that their order makes sense



# This array makes a bunch of arrays you can plot.
#   Arrays:
#     self.time    - Time at the start of each frame in femtoseconds
#     self.pe      - Potential Energy in eV as calculated in TeraChem
#     self.ke      - Kinetic Energy in eV of Nuclei in Ehrenfest Dynamics
#     self.tot     - pe+ke
#     self.reltot  - Relative energies ( tot-min(tot) )
#     self.diff    - Total Energy change between step i and previous step.
#     self.rmsgrad - Root Mean Square of the gradient of the wfn calculated 
#                     halfway through TDCI prop.
#      Note: S refers to orbital overlap matrix between current and previous nuclear-step wfn
#     self.S_sq_oos      - Orbital norm rotation between active and non-active orbital space
#     self.S_sq_actidiag - Orbital norm unchanged in rotation
#     self.S_sq_actioffdiag - Orbital norm rotating within active space
#     self.S_sq_actisum  - Total norm in active space (should be equal to norbitals)
#     self.state_proj    - Requires DoStateProjections=True. Dictionary. state_proj[i] has 
#                           the projection of the wfn onto the i'th state at each step.
#                           provided that the i'th state was calculated by terachem.
#     self.rmsgrad_state - Dictionary. rmsgrad_state[i] contains Root Mean Square
#                           Gradient of i'th state.
#     self.fomo_eng - Dictionary. fomo_eng[i] is the fomo energy of orbital i at each step.
#     self.fomo_eng - Dictionary. fomo_occ[i] is the fomo occup of orbital i at each step.
#

class plottables:
  def init_params(self):
    start = time.time()
    #nmo, clsd, ndets, acti, nstates, natoms = 26, 7, 4, 2, 3, 6
    d = self.d
    dt = self.d + "electronic/1/"
    # we could save some time if we make a scan_outfile function does multiple of these...
    self.nmo = int(scan_outfile(dt+"test1.out", ["Number", "of", "molecular", "orbitals:"], 4))
    self.clsd = int(scan_outfile(dt+"test1.out", ["Number", "of", "closed", "orbitals:"], 4))
    self.ndets = int(scan_outfile(dt+"test1.out", ["Number", "of", "determinants:"], 3))
    self.acti = int(scan_outfile(dt+"test1.out", ["Number", "of", "active", "orbitals:"], 4))
    nsinglets = int(scan_outfile(dt+"test1.out", ["Number", "of", "singlet", "states:"], 4))
    ndoublets = int(scan_outfile(dt+"test1.out", ["Number", "of", "doublet", "states:"], 4))
    ntriplets = int(scan_outfile(dt+"test1.out", ["Number", "of", "triplet", "states:"], 4))
    nquartets = int(scan_outfile(dt+"test1.out", ["Number", "of", "quartet", "states:"], 4))
    nquintets = int(scan_outfile(dt+"test1.out", ["Number", "of", "quintet", "states:"], 4))
    self.nstates = int(nsinglets) + int(ndoublets) + int(ntriplets) + int(nquartets) + int(nquintets)
    self.natoms = 0
    with open(dt+"temp.xyz","r") as xyzf:
      self.natoms = int(xyzf.readline().strip())

    # check if we're running fomo
    self.DoFOMO = False
    fomo = scan_outfile(dt+"test1.in", ["fon"], 1)
    if fomo == "yes":
      self.DoFOMO = True

    # Check if we calculated gradients of each state
    self.DoGradStates = False
    tdci_grad_states = scan_outfile(dt+"test1.in", ["tdci_grad_states"], 1)
    if tdci_grad_states == "yes":
      self.DoGradStates = True
    runtime = "init_params runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()

  def get_state_grads(self):
    start = time.time()
    print("Starting get_state_grads, this may take several minutes for large systems with many steps")
    rmsgrad_state = {}
    for state in range(0, self.nstates):
      rmsgrad_state[state] = []
    for i in range(1,self.nstep):
      if i%500 == 0: print(i);sys.stdout.flush()
      for state in range(0, self.nstates):
        dt = self.d+"electronic/"+str(i)+"/"
        if not os.path.exists(dt+"gradstate"+str(state)+".bin"):
          rmsgrad_state[state].append(-1.0) # Why is this here? When does this happen?
        else:
          grad_tdci = read_bin_array(dt+"gradstate"+str(state)+".bin",3*self.natoms)
          rmsgrad_state[state].append(rms(grad_s1_tdci))
    runtime = "rmsgrad_state runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()
    return rmsgrad_state

  # Is something wrong with this approach? I was doing it a different and weird way before...
  def get_state_projections(self):
    start = time.time()
    state_proj = {}
    for state in range(0, self.nstates):
      state_proj[state] = []
    for i in range(1,self.nstep-2):
      dt = self.d+"electronic/"+str(i)+"/"
      recn = read_bin_array(dt+"ReCn_end.bin",self.ndets)
      imcn = read_bin_array(dt+"ImCn_end.bin",self.ndets)
      cn = read_bin_array(dt+"States_Cn.bin",self.nstates*self.ndets)
      cn.resize((self.nstates,self.ndets))
      for state in range(0, self.nstates):
        reproj = np.dot(cn[state], recn) 
        improj = np.dot(cn[state], imcn)
        projnorm = np.sqrt( reproj**2 + improj**2 )
        state_proj[state].append(projnorm)
    runtime = "get_state_projections runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()
    return state_proj

    
  def get_S_diagnostics(self):
    start = time.time()
    print("Starting get_S_diagnostics, this may take several minutes for large systems with many steps")
    self.S_sq_oos = [] # out of space rotation
    self.S_sq_actidiag = [] # sumsq active diagonals
    self.S_sq_actioffdiag = [] # sumsq active off-diagonals (in-space rotation)
    self.S_sq_actisum = [] # sum of active orbitals
    act_orbs = range(self.clsd, self.clsd+self.acti)
    
    for i in range(1,self.nstep):
      if (i%500 == 0): print(i)
      dt = self.d+"electronic/"+str(i)+"/"
      s = read_bin_array(dt+"S_MIXED_MO.bin",self.nmo**2)
      s.resize((self.nmo,self.nmo))
      s2 = s**2 # square element-wise
      oos = 0
      diag = 0
      offdiag = 0
      for i in act_orbs:
        diag += s2[i][i]
        for j in act_orbs:
          if (i != j): # active block offdiag
            offdiag += s2[i][j]

      for i in range(0,self.nmo):
        for j in range(0,self.nmo):
          #  i XOR j is in the active space
          # (rotations between active and non-active orbitals)
          if (np.abs(s[i][j]) > 0 and  ( ( (i in act_orbs) and not (j in act_orbs)) or  ((j in act_orbs) and not (i in act_orbs)) )):
            oos+=s2[i][j]

      self.S_sq_oos.append(oos)
      self.S_sq_actidiag.append(diag)
      self.S_sq_actioffdiag.append(offdiag)
      self.S_sq_actisum.append(diag+offdiag)
    runtime = "get_S_diagnostics runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()

  def get_h5data(self):
    start = time.time()
    d = self.d
    # Opening a h5py file while the simulation is running will crash it, so create a copy!
    p = subprocess.Popen('cp "'+d+'data.hdf5" "'+d+'data_read.hdf5"', shell=True)
    p.wait()
    time.sleep(1)
    h = h5py.File(d+"data_read.hdf5", 'r')
    self.nstep = nstep = len(h['time'][1:])
    p = h['pe'][1:]
    k = h['ke'][1:]
    self.tot = np.array(p)+np.array(k)
    totmin = min(self.tot[2:nstep])
    self.reltot = [(z-totmin)*27.2114 for z in self.tot]
    self.time = np.array(h['time'][1:])/1000. # in fs
    self.steptime = (h['time'][3] - h['time'][2])/1000. # in fs
    self.diff = [0.0]
    for i in range(1,len(self.reltot)):
      self.diff.append(abs(self.reltot[i-1]-self.reltot[i]))
    self.pe = 27.2114*np.array(h['pe'][1:])
    self.ke = 27.2114*np.array(h['ke'][1:])
    h.close()
    runtime = "get_h5data runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()

  # get FOMO orbital energies and occupations
  def get_fomodata(self):
    d = self.d
    start = time.time()
    self.fomo_eng = {}
    self.fomo_occ = {}
    for i in range(1, self.nstep):
      # scan_fomo returns a list of [ (orb_index, orb_energy, orb_occupation),... ]
      fomotemp = scan_fomo(d+"electronic/"+str(i)+"/test"+str(i)+".out")
      for j in fomotemp:
        if j[0] not in self.fomo_eng: # make sure there is a key for this orbital
          self.fomo_eng[j[0]] = [j[1]]
          self.fomo_occ[j[0]] = [j[2]]
        else:
          self.fomo_eng[j[0]].append(j[1])
          self.fomo_occ[j[0]].append(j[2])
    runtime = "get_fomodata runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()

  def get_rmsgrad(self):
    start = time.time()
    rmsgrad = []
    for i in range(1,self.nstep):
      dt = self.d+"electronic/"+str(i)+"/"
      grad = read_bin_array(dt+"tdcigrad_half.bin",3*self.natoms)
      rmsgrad.append(rms(grad))
    runtime = "get_rmsgrad runtime: {:10.6f} seconds\n".format(time.time() - start)
    print(runtime);sys.stdout.flush()
    return rmsgrad

  def __init__(self, d, label, DoStateProjections=False, DoSDiagnostic=True):
    start_ = time.time()
    self.label = label
    self.d = d
    self.DoStateProjections = DoStateProjections
    self.DoSDiagnostic = DoSDiagnostic
    self.get_h5data()
    self.init_params()
    if DoSDiagnostic: self.get_S_diagnostics() # sets self.S_sq_*, where *: oos, actidiag, actioffdiag, actisum
    if self.DoFOMO: self.get_fomodata() # sets self.fomo_eng and self.fomo_occ dictionaries
    if self.DoGradStates: self.rmsgrad_state = self.get_state_grads()     
    if DoStateProjections: self.state_proj = self.get_state_projections()
    self.rmsgrad = self.get_rmsgrad()
    runtime = "Total "+label+" runtime: {:10.6f} seconds\n".format(time.time() - start_)
    print(runtime);sys.stdout.flush()


    
  # Accepts a range of femtoseconds, converts to indices for arrays in plottables
  def fs2index_range(self, fs_start, fs_end):
    startt, endt = 0, 0
    while (self.time[startt] < fs_start):
        startt+=1
        if startt > len(self.time)-5: break
    while self.time[endt] < fs_end: 
        endt+=1
        if endt > len(self.time)-5: break
    return startt, endt






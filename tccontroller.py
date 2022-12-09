#!/usr/bin/python2.7
# Written by Andy Durden 2022

from __future__ import print_function
import numpy as np
import struct, shutil, os, sys, subprocess, time
import copy


# Constants
FStoAU = 41.341375
EPSILON_C = 0.00265316
E_FIELD_AU = 5.142206707E+11

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf)

# TCPB uses what i'm calling "tcstring" here
# TODO: add TCPB support :)
def tcstring_to_xyz(atoms,geom,filename):
  f = open(filename,'w')
  f.write(str(len(atoms))+"\n\n")
  for i in range(0,len(atoms)):
    f.write(str(atoms[i])+"   "+str(geom[3*i+0])+"    "+str(geom[3*i+1])+"    "+str(geom[3*i+2])+"\n")
  f.close()

def xyz_to_tcstring(filename):
  f = open(filename,'r')
  f.readline();f.readline()
  l = f.readline()
  atoms = []
  geom = []
  #print(l.split(" "))
  while len(l.split(" "))==4:
    atoms.append(l.split(" ")[0])
    geom.append(l.split(" ")[1]) 
    geom.append(l.split(" ")[2]) 
    geom.append(l.split(" ")[3])
    l = f.readline()


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



# Used to sort input file dictionaries into something a little more human-readable
# before writing them as a file
def dictkey(key):
  keylist = ["gpus", "timings", "precision", "threall", "convthre", "basis", 
             "coordinates", "method", "run", "to", "charge", "spinmult", "csf_basis",
             "tdci_simulation_time", "tdci_nstep", "tdci_eshift", "tdci_stepprint",
             "tdci_nfields", "tdci_laser_freq", "tdci_photoneng", "tdci_fstrength",
             "tdci_fdirection", "tdci_ftype", "tdci_corrfn_t", "tdci_write_field",
             "tdci_floquet", "tdci_floquet_photons", "tdci_krylov_end", "tdci_krylov_end_n",
             "tdci_krylov_end_interval", "tdci_diabatize_orbs", "tdci_recn_readfile",
             "tdci_imcn_readfile", "tdci_prevorbs_readfile", "tdci_prevcoords_readfile",
             "tdci_gradient", "tdci_grad_init", "tdci_grad_half", "tdci_fieldfile0",
             "tdci_fieldfile1", "tdci_fieldfile2", "tdci_fieldfile3", "tdci_fieldfile4",
             "casci", "fon", "fon_method", "fon_temperature", "ci_solver", "dcimaxiter",
             "dciprintinfo", "dcipreconditioner", "closed", "active", "cassinglets",
             "casdoublets", "castriples", "casquartets", "cascharges", "cas_ntos",]
  if key in keylist:
    return keylist.index(key)
  else:
    return len(keylist)+1

def dict_to_file(d, filepath):
  dkeys = d.keys()
  dkeys.sort(key=dictkey)
  f = open(filepath, 'w')
  for key in dkeys:
    f.write(str(key)+" "+str(d[key])+"\n")
  f.close()

def dicts_to_file(dlist, filepath):
  f = open(filepath, 'w')
  for d in dlist:
    dkeys = d.keys()
    dkeys.sort(key=dictkey)
    for key in dkeys:
      f.write(str(key)+" "+str(d[key])+"\n")
  f.close()

# we may need to be careful about endian-ness if this runs on a different machine than TC
def read_bin_array(filepath, length):
  #print("l: "+str(length))
  if length == 0:
    return np.array([])
  f = open(filepath, 'rb')
  return np.array(struct.unpack('d'*length, f.read()))

def read_csv_array(filepath):
  f = open(filepath, 'r')
  text = f.read()
  text = text.strip(", \n")
  return np.array(list(map(float, (text).split(","))))

# accepts ONE-DIMENSIONAL DOUBLE ARRAYS	
def write_bin_array(array, filepath):
  if (len(np.shape(array)) != 1):
    print("WARNING: write_bin_array() is for 1D double arrays! Your array has shape:")
    print(np.shape(array))
  f = open(filepath, 'wb')
  f.write( struct.pack('d'*len(array), *array ))
  f.close(); del f

def search_replace_file(filepath, search, replace):
  p = subprocess.Popen('sed -i -e "s+'+search+'+'+replace+'+g" '+str(filepath), shell=True)
  p.wait()
  return 0

# ensure every subdirectory of a path exists
def makedirs(dirstr):
  for i in range(2, dirstr.count("/")+1):
    if not os.path.exists("/".join(dirstr.split("/")[0:i])):
      os.mkdir("/".join(dirstr.split("/")[0:i]))
  return os.path.exists(dirstr)

class logger():
  def __init__(self):
    self.maxlogs = 20
    self.rotate_logs()
    f = open("log",'w');f.write("");f.close();del f
    
  def rotate_logs(self):
    i = self.maxlogs
    if os.path.exists("log."+str(i)):
      os.remove("log."+str(i))
    i += -1
    while i>0:
      if os.path.exists("log."+str(i)):
        os.rename("log."+str(i), "log."+str(i+1))
      i += -1
    if os.path.exists("log"):
      os.rename("log", "log.1")

  # logprint(string) will write a timestamped string to the log, and to STDOUT
  def logprint(self, string, PRINT=True):
    writestr = "["+time.asctime()+"] "+string+"\n"
    f = open("log",'a')
    f.write(writestr)
    f.close()
    print(writestr, end='')


class molden:
  def __init__(self, filename, clsd, acti):
    self.filename = filename
    self.lines = []
    f = open(filename, 'r')
    for line in f: self.lines.append(line)
    f.close(); del f
    self.clsd = clsd
    self.acti = acti
    self.dissect()
    self.write_activeMOs(clsd,acti)

  # separate parts of the molden file
  def dissect(self):
    i=0
    while "[MO]" not in self.lines[i]:
     i+=1
    self.header = self.lines[0:i]
    #print("header:\n")
    #print(self.header)
    self.MOs = []
    for j in range(0,len(self.lines)):
      if "Ene=" in self.lines[j]:
        self.MOs.append(j)

  def write_activeMOs(self, clsd, acti):
    f = open("/".join(self.filename.split("/")[0:-1])+"/tdciactive.molden",'w')
    for line in self.header:
      f.write(line)
    mo_len = self.MOs[1]-self.MOs[0]
    for i in range(clsd, clsd+acti):
      for j in range(0,mo_len):
        #print(i)
        #print((len(self.lines),self.MOs[i],j))
        f.write(self.lines[self.MOs[i]+j])
    f.close()

    

class job:
  def __init__(self, n, Natoms, Nkrylov, ReCn, ImCn, xyz, pjob, JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE, FIELD_INFO, logger=None, SCHEDULER=False, halfstep=False, noclean=False):
    self.n = n
    self.Natoms = Natoms
    self.Nkrylov = Nkrylov
    self.ReCn = ReCn # Initial value input by user
    self.ImCn = ImCn
    self.xyz = xyz
    self.pjob = pjob
    if halfstep:
      self.dir = JOBDIR+"electronic/"+str(n)+"h/"
    else:
      self.dir = JOBDIR+"electronic/"+str(n)+"/"
    self.halfstep = halfstep
    self.JOBDIR=JOBDIR
    self.JOB_TEMPLATE=JOB_TEMPLATE # contents of bash script that runs tc (with tempdir and tempname)
    self.TDCI_TEMPLATE=TDCI_TEMPLATE # dictionary of terachem options written to input file
    self.SCHEDULER=SCHEDULER # Unimplemented: will interface with SLURM 
    self.FIELD_INFO = FIELD_INFO
    self.ndets = 0
    self.nmo = 0
    self.nbf = 0
    self.restarts = 0
    self.gradjob = False
    self.logger = logger
    self.noclean = noclean
    if not noclean:
      self.clean_files()

  def start(self):
    p = subprocess.Popen( 'bash '+self.dir+'tdci.job', shell=True)
    return p

  def readmisc(self):
    f = open(self.dir+"misc.bin",'rb')
    self.ndets, self.nmo, self.nbf = struct.unpack('iii', f.read())
    f.close();del f

  def make_files(self):
    makedirs(self.dir)
    xyz_write(self.FIELD_INFO["atoms"], self.xyz, self.dir+"temp.xyz")
    with open(self.dir+"tdci.job", 'w') as templatefile:
      templatefile.write(self.JOB_TEMPLATE)
    time.sleep(1) # make sure file gets written and closed properly
    search_replace_file(self.dir+"tdci.job", "temppath", self.dir)
    search_replace_file(self.dir+"tdci.job", "tempname", "test"+str(self.n))
    
    tempname = "test"+str(self.n)+".in"
    dict_to_file(self.TDCI_TEMPLATE, self.dir+"/"+tempname)
    if self.gradjob:
      search_replace_file(self.dir+tempname, "tdci_fieldfile0 field0.bin", "")
      search_replace_file(self.dir+tempname, "tdci_fieldfile1 field1.bin", "")
      search_replace_file(self.dir+tempname, "tdci_fieldfile2 field2.bin", "")
    else:
      self.make_fieldfiles()
    if (self.gradjob) or (self.n==0):
      search_replace_file(self.dir+tempname, "tdci_diabatize_orbs yes", "tdci_diabatize_orbs no")
      if self.ReCn is None:
        search_replace_file(self.dir+tempname, "tdci_recn_readfile recn_init.bin", "")
      else:
        write_bin_array(self.ReCn,self.dir+"recn_init.bin")
      if self.ImCn is None:
        search_replace_file(self.dir+tempname, "tdci_imcn_readfile imcn_init.bin", "")
      else:
        write_bin_array(self.ImCn,self.dir+"imcn_init.bin")
      search_replace_file(self.dir+tempname, "tdci_prevorbs_readfile PrevC.bin", "")
      search_replace_file(self.dir+tempname, "tdci_prevcoords_readfile PrevCoors.bin", "")
      search_replace_file(self.dir+tempname, "tdci_krylov_init cn_krylov_init.bin", "")
      return 0

    else: # Copy Prev Orbitals and Coords (in double4) for orbital diabatization
      pjobd = self.pjob.dir
      shutil.copy(pjobd+"/NewCoors.bin", self.dir+"/PrevCoors.bin")
      shutil.copy(pjobd+"/NewC.bin", self.dir+"/PrevC.bin")
      if self.ReCn is None:
        shutil.copy(pjobd+"/ReCn_end.bin", self.dir+"/recn_init.bin")
      else:
        write_bin_array(self.ReCn,self.dir+"recn_init.bin")
      if self.ImCn is None:
        shutil.copy(pjobd+"/ImCn_end.bin", self.dir+"/imcn_init.bin")
      else:
        write_bin_array(self.ImCn,self.dir+"imcn_init.bin")
      if self.FIELD_INFO["krylov_end"]:
        shutil.copy(pjobd+"/Cn_krylov_end.bin", self.dir+"/cn_krylov_init.bin")
      return 0

  def clean_files(self):
    if os.path.exists(self.dir):
      shutil.rmtree(self.dir)

  def make_fieldfiles(self):
    # Field file should include values for half-steps, so the length of the array
    #   should be 2*nsteps!
    FStoAU = 41.341375
    T = float(self.TDCI_TEMPLATE["tdci_simulation_time"])
    N = int(self.TDCI_TEMPLATE["tdci_nstep"])
    # half is 0 if we start evenly on timestep, 0.5 if start on half timestep
    half = self.FIELD_INFO["half"]/2.
    t = np.linspace( (self.n+half)*T*FStoAU, (self.n+half+1)*T*FStoAU, 2*N)
    for i in range(0, self.FIELD_INFO["nfields"]):
      vals = self.FIELD_INFO["f"+str(i)](t)
      write_bin_array(vals,self.dir+"field"+str(i)+".bin")

  def check_output(self,output):
    logprint = self.logger.logprint
    outputgood = True
    norm = np.linalg.norm(output["recn"])**2 + np.linalg.norm(output["imcn"])**2
    logprint("Final wfn norm (MO basis): "+str(norm))
    if ((norm<0.7) or (norm>1.1) or (np.isnan(norm))):
      logprint("ERROR: Norm out of bounds")
      outputgood = False
    logprint("Sum of gradient elements: "+str(np.sum(output["grad"])))
    if (np.isnan(np.sum(output["grad"]))):
      logprint("ERROR: nan in gradient")
      outputgood = False
    if self.FIELD_INFO["krylov_end"]:
      norm = np.sum(output["recn_krylov"]**2) + np.sum(output["imcn_krylov"]**2)
      logprint("Final wfn norm (AES basis): "+str(norm))
      krylov_MO_Re = np.matmul(np.transpose(output["krylov_states"]), output["recn_krylov"])
      krylov_MO_Im = np.matmul(np.transpose(output["krylov_states"]), output["recn_krylov"])
      #print("Checking AES basis quality:")
      #print("ReCn:")
      #print(output["recn"])
      #print("ImCn:")
      #print(output["imcn"])
      #print("ReCn (AES->MO):")
      #print(krylov_MO_Re)
      #print("ImCn (AES->MO):")
      #print(krylov_MO_Im)
      overlap = np.dot(krylov_MO_Re,output["recn"])**2 + np.dot(krylov_MO_Im,output["imcn"])**2
      logprint("Overlap of AES-MO with MO:"+str(overlap))
    if outputgood:
      return True
    else:
      return False


  def run_safely(self):
    logprint = self.logger.logprint
    # Run the job
    retries = 0
    while (retries < 3):
      if not self.noclean:
        self.clean_files()
        self.make_files()
      p = self.start()
      logprint("Started "+str(self.dir)+"\n")
      finished = False
      # Periodically check if the process is finished
      i = 1
      status = "TC Running."
      print(status, end=""); sys.stdout.flush()
      while not finished:
        if self.check_status(p):
          print(""); sys.stdout.flush()
          finished = True
        else: # fun rotating dots
          if (i%10 == 0):
            print("\b"*9, end="");
            print(" "*9, end="");
            print("\b"*9, end=""); sys.stdout.flush()
          else:
            print(".", end=""); sys.stdout.flush()
        time.sleep(1)
        i+=1
      # Make sure output is good
      if self.gradjob: # if we're doing gradient only (not tdci) for initial step
        output = self.gradoutput()
      elif self.TDCI_TEMPLATE["run"] == "frequencies":
        #xyzname = os.path.basename(self.xyzpath) # remove directories from path
        #xyzname = xyzname[:xyzname.rindex(".")] # remove extension
        xyzname = "temp"
        scrdir = self.dir + "scr."+xyzname+"/"
        output = self.read_hessfile(scrdir+"Hessian.bin")
      else: # normal tdci
	output = self.output()
      if output: # Everything checks out!
        logprint("Output looks good!")
        return output
      else: # Outputs bad, try redoing the job!
        logprint("Output is bad. Restarting the job. See bad jobfiles in ./badjobs/"+str(self.n)+"_"+str(retries))
        makedirs("./badjobs/"+str(self.n)+"_"+str(retries))
        shutil.copytree( self.dir, "badjobs/"+str(self.n)+"_"+str(retries))
      retries+=1
    logprint("Went through 3 retries and output is still bad T_T\n")
    return output

      
  def check_status(self, p):
    if self.SCHEDULER:
      # TODO: Need to call squeue and decipher it's output
      pass
    else: 
      result = p.poll()
      #print(result)
      if result != None: # returns None if still running
        return True
      else:
        return False

  # key : list of words that match the beginning of line.split()
  # For ndets, ["Number", "of", "determinants:"]
  # pos : the index of the element to be returned from matching line.split()
  def scan_outfile(self, key, pos):
    logprint = self.logger.logprint
    f = open(self.dir+"test"+str(self.n)+".out", 'r')
    l = f.readline()
    while l != "":
      if (len(l.split()) > len(key)):
        if (l.split()[:len(key)] == key):
          return l.split()[pos]
      l = f.readline()
    logprint("key "+str(key)+" not found :( ")
    return None

  def scan_normpop(self):
    logprint = self.logger.logprint
    if not os.path.exists(self.dir+"Pop"):
      logprint("file Pop does not exist.")
    if not os.path.exists(self.dir+"norm"):
      logprint("file norm does not exist.")
    f = open(self.dir+"Pop",'r')
    l = f.readline()
    S0_start = float(l.split(",")[1])
    lp = None
    while l != "":
      lp = l
      l = f.readline()
    # lp should be last line of Pop
    S0_end = float(lp.split(",")[1])
    f.close()
    g = open(self.dir+"norm",'r')
    l = g.readline()
    norm_start = float(l.split(",")[1])
    lp = None
    while l != "":
      lp = l
      l = g.readline()
    norm_end = float(lp.split(",")[1])
    g.close()
    return S0_start, S0_end, norm_start, norm_end

  def gradoutput(self):
    logprint = self.logger.logprint
    filesgood = True
    files = ["gradinit.bin", "States_Cn.bin", "States_E.bin", "misc.bin"]
    for fn in files:
      if not os.path.exists(self.dir+fn):
        filesgood = False
        logprint("ERROR: "+fn+" missing")
    if not filesgood:
      return False
    grad = read_bin_array(self.dir+"gradinit.bin", 3*self.Natoms)
    grad.resize((self.Natoms,3))
    logprint("Grad:\n"+str(grad))
    #E = float(read_bin_array(self.dir+"Einit.bin", 1)[0])
    E = float(self.scan_outfile(["Initial", "energy:"], 2))
    nstates = int(self.TDCI_TEMPLATE["cassinglets"])
    if "casdoublets" in self.TDCI_TEMPLATE:
      nstates+= int(self.TDCI_TEMPLATE["casdoublets"])
    if "castriplets" in self.TDCI_TEMPLATE:
      nstates+= int(self.TDCI_TEMPLATE["castriplets"])
    if "casquartets" in self.TDCI_TEMPLATE:
      nstates+= int(self.TDCI_TEMPLATE["casquartets"])
    if (self.ndets == 0):
      self.readmisc()
    states = read_bin_array(self.dir+"States_Cn.bin", nstates*self.ndets)
    states.resize((nstates, self.ndets))
    states_eng = read_bin_array(self.dir+"States_E.bin", nstates)
    return { "grad"       : grad,
             "eng"        : E,
             "states"     : states,
             "states_eng" : states_eng,
             "recn" : self.ReCn,
             "imcn" : self.ImCn,
             "tdci_dir": self.dir
           }


  def read_hessfile(self, filepath):
    logprint = self.logger.logprint
    natoms = self.Natoms
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
      logprint("ERROR: Extra bytes in Hessian.bin, something isn't right...")
    #print( (len(first),len(second),len(third),len(rest)))
    hessian = np.array(struct.unpack('d'*((3*natoms)**2), second))
    hessian.resize((3*natoms,3*natoms))
    size = 3 # if RunPolarizability in TC, this should be 12. Not sure if we ever need that.
    dipolederiv = np.array(struct.unpack('d'*(size*3*natoms), third))
    return {"hessian": hessian, "dipolederiv": dipolederiv}

  def output(self):
    logprint = self.logger.logprint
    clsd, acti = int(self.TDCI_TEMPLATE["closed"]), int(self.TDCI_TEMPLATE["active"])
    filesgood = True
    files = ["ReCn_end.bin","ImCn_end.bin", "tdcigrad.bin", "tdcigrad_half.bin", "misc.bin", "tdci.molden"]
    if (self.n > 0) and (self.TDCI_TEMPLATE["tdci_diabatize_orbs"] == "yes"):
      files += ["S_MIXED_MO_active.bin"]
    if self.FIELD_INFO["krylov_end"]:
      files += ["ReCn_krylov_end.bin", "ImCn_krylov_end.bin", "Cn_krylov_end.bin", "E_krylov_end.bin", "tdcigrad_krylov.bin"]
    for fn in files:
      if not os.path.exists(self.dir+fn):
        filesgood = False
        logprint("ERROR: "+fn+" missing")
    if not filesgood:
      return False

    if (self.ndets == 0):
      self.readmisc()
    # rewrites the molden file with just the active orbitals
    moldenwriter = molden(self.dir+"tdci.molden", clsd, acti)
    del moldenwriter 
    
    # Format output structure
    eng = float(self.scan_outfile(["Final", "TDCI", "Energy:"], 3))

    if os.path.exists(self.dir+"gradinit.bin"):
      grad_init = read_bin_array(self.dir+"gradinit.bin", 3*self.Natoms)
      grad_init.resize((self.Natoms, 3))
      logprint("Grad_init:\n"+str(grad_init))

    grad = read_bin_array(self.dir+"tdcigrad.bin", 3*self.Natoms)
    grad.resize((self.Natoms, 3))
    grad_half = read_bin_array(self.dir+"tdcigrad_half.bin", 3*self.Natoms)
    #grad_half = read_bin_array(self.dir+"grad_S0.bin", 3*self.Natoms)
    grad_half.resize((self.Natoms, 3))
    logprint("Grad_half:\n"+str(grad_half))
    krylov_states = None
    krylov_energies = None
    krylov_gradients = None
    recn_krylov = None
    imcn_krylov = None
    recn = read_bin_array(self.dir+"ReCn_end.bin", self.ndets)
    imcn = read_bin_array(self.dir+"ImCn_end.bin", self.ndets)
    if self.FIELD_INFO["krylov_end"]:
      # just calculate these
      #recn_krylov = read_bin_array(self.dir+"ReCn_krylov_end.bin", self.Nkrylov)
      #imcn_krylov = read_bin_array(self.dir+"ImCn_krylov_end.bin", self.Nkrylov)
      # we dont have ndets, and everything else is in AES basis anyway.
      krylov_states = read_bin_array(self.dir+"Cn_krylov_end.bin", self.Nkrylov*self.ndets)
      krylov_states.resize((self.Nkrylov,self.ndets))
      recn_krylov = np.dot(krylov_states, recn)
      imcn_krylov = np.dot(krylov_states, imcn)
      krylov_energies = read_bin_array(self.dir+"E_krylov_end.bin", self.Nkrylov)
      krylov_gradients = read_bin_array(self.dir+"tdcigrad_krylov.bin", 3*self.Natoms*self.Nkrylov)
      krylov_gradients.resize((self.Nkrylov, self.Natoms, 3))

    output = { "recn": recn,  # 1d array, number of determinants
               "imcn": imcn,  # 1d array, number of determinants
               "eng": eng,    # float, Energy of current wfn
               "grad": grad,    # 2d array, Natoms x 3 dimensions.
               "grad_half": grad_half,    # 2d array, Natoms x 3 dimensions.
               "recn_krylov": recn_krylov,      # 1d array, 2*krylov_sub_n
               "imcn_krylov": imcn_krylov,      # 1d array, 2*krylov_sub_n
               "krylov_states": krylov_states,  # 2d array of CI vectors of each approx eigenstate
               "krylov_energies": krylov_energies, # 1d array of energies of each approx eigenstate
               "krylov_gradients": krylov_gradients, # 3d array of approx eigenstate gradients, Napprox x Natoms x 3 dim.
               "tdci_dir": self.dir
             }

    #print("TDCI job Output:\n")
    #print(output)

    # Check overlap matrices
    printS = False
    S_prediab = None
    nmo = self.nmo
    nbf = self.nbf
    if self.n > 0:
      # Error analysis
      # tdci end energy is in 'eng'
      eng_tdci_start = float(self.scan_outfile(["Initial", "energy:"], 2))
      prevstep_end_eng = float(self.pjob.scan_outfile(["Final", "TDCI", "Energy:"], 3))
      diaberr = 27.2114*(eng_tdci_start - prevstep_end_eng)
      tdcierr = 27.2114*(eng - eng_tdci_start)
      logprint("Error diab ("+str(self.n-1)+"  -> "+str(self.n)+"  ): "+"{: .8f}".format(diaberr)+" eV")
      logprint("Error tdci ("+str(self.n)+"i -> "+str(self.n)+"f ): "+"{: .8f}".format(tdcierr)+" eV")
      if diaberr > 0.5: printS = True
      if (self.TDCI_TEMPLATE["tdci_diabatize_orbs"] == "yes"):
        printS = True
        S_prediab = read_bin_array(self.dir+"S_MIXED_MO_active.bin", acti**2)
        S_prediab.resize((acti,acti))
        for i in range(0,acti):
          if np.abs(S_prediab[i][i]) < 0.5: logprint("WARNING: S_prediab["+str(i)+"]["+str(i)+"] = "+str(S_prediab[i][i]))
          if np.linalg.norm(S_prediab[i]) < 0.9: logprint("WARNING: norm(S_prediab["+str(i)+"]) = "+str(np.linalg.norm(S_prediab[i])))
      else: printS = False

    #normpop = self.scan_normpop()
    #S0_start, S0_end, norm_start, norm_end = normpop[0], normpop[1], normpop[2], normpop[3]
    S0_start, S0_end, norm_start, norm_end = self.scan_normpop()
    #print((S0_start, S0_end, norm_start, norm_end))
    logprint("S0   start->end: "+str(S0_start)+" -> "+str(S0_end)+" ("+str(S0_end-S0_start)+")")
    logprint("norm start->end: "+str(norm_start)+" -> "+str(norm_end)+" ("+str(norm_end-norm_start)+")")
    if printS:
      logprint("S Prediab  : \n"+str(S_prediab))
      logprint("S Prediab row norms: "+str(map(lambda x: np.linalg.norm(x), S_prediab)))
      

    # ugh this is bad. error checking should have access to more information than we're outputting.
    if self.check_output(output):
      #self.sanity_test(output)
      return output
    else:
      return False

  def sanity_test(self, output):
    logprint = self.logger.logprint
    logprint("Sanity test on output...")
    Cn_approx_end = None
    Qt_end = None
    if (os.path.exists(self.dir+"Cn_approx_end.bin")) and (os.path.exists(self.dir+"Qt_end.bin")):
      logprint("Extra debug files present:")
      Cn_approx_end = read_bin_array(self.dir+"Cn_approx_end.bin", self.Nkrylov**2)
      Cn_approx_end.resize((self.Nkrylov,self.Nkrylov))
      Qt_end = read_bin_array(self.dir+"Qt_end.bin", self.Nkrylov*self.ndets)
      Qt_end.resize((self.Nkrylov,self.ndets))
      Cn_krylov = np.matmul(Cn_approx_end, Qt_end)
      if np.allclose(Cn_krylov, output["krylov_states"]):
        logprint("Pass! Cn_approx_end * Qt_end = krylov_states")
      else:
        logprint("Fail! Cn_approx_end * Qt_end != krylov_states")
    else:
      logprint("Extra debug files not present.")
    if np.allclose(np.matmul(output["krylov_states"],output["recn"]),output["recn_krylov"]):
      logprint("Pass! recn_krylov == krylov_states * recn")
    else:
      logprint("Fail! recn_krylov != krylov_states * recn")

    import pickle
    with open("data.pickle", 'wb') as f:
      pickle.dump([Cn_approx_end, Qt_end, output], f, pickle.HIGHEST_PROTOCOL)
    logprint("Sanity test finished.")
      


class tccontroller:
  def __init__(self, config, logger):
    self.config = config
    self.N = 0
    # halfstep is True if the next job should start on a half-timestep
    #   Since Ehrenfest does a halfstep first and then only fullsteps, 
    #   this should be true during the entire propagation loop 
    self.jobs = []
    self.prevjob = None
    self.JOBDIR=config.JOBDIR
    if self.JOBDIR[-1] != "/": # make sure theres a leading slash on the directory
      self.JOBDIR+="/"
    self.JOB_TEMPLATE=config.JOB_TEMPLATE
    self.TDCI_TEMPLATE=config.TDCI_TEMPLATE
    self.SCHEDULER=config.SCHEDULER
    self.FIELD_INFO=config.FIELD_INFO
    self.FIELD_INFO["half"] = 0 
    self.Natoms = len(config.FIELD_INFO["atoms"])
    self.Nkrylov = 2*config.FIELD_INFO["krylov_end_n"]
    self.logger = logger
    self.RESTART = config.RESTART
    #if self.RESTART:
    #  self.restart()

  # Need to populate self.prevjob so orbital diabatization takes place
  # start job with noclean so that we can use the files setup by reading hdf5
  def restart_setup(self):
    self.FIELD_INFO["half"] = 1 
    print("Setting prevjob: "+str(self.N-1))
    xyz = np.zeros((self.Natoms,3)) # need one to make a job instance, should be fine since we're not running it.
    j = job( self.N-1, self.Natoms, self.Nkrylov, None, None, xyz, 
	     None, self.JOBDIR, self.JOB_TEMPLATE, self.TDCI_TEMPLATE, self.FIELD_INFO, 
	     logger=self.logger, SCHEDULER=self.SCHEDULER, noclean=True )
    self.prevjob = j
    
    

  # find the last valid TDCI calculation for continuity, return step number.
  def restart(self):
    print("Detecting last valid TDCI calculation in "+str(self.JOBDIR)+"electronic/")
    joblist = [f for f in os.listdir(self.JOBDIR+"electronic/") if os.path.isdir(f) ]
    joblist.sort(reverse=True) # highest numbered directories will be at the start of the list
    prevjob = None
    i=0
    while ((prevjob == None) and (i < len(joblist))): # loop over the subdirectories
      if joblist[i].isdigit(): # make sure directory is numbered in case of 'grad'
        xyz = np.zeros((self.Natoms,3)) # need one to make a job instance, should be fine since we're not running it.
        j = job( int(joblist[i]), self.Natoms, self.Nkrylov, None, None, xyz, 
                 None, self.JOBDIR, self.JOB_TEMPLATE, self.TDCI_TEMPLATE, self.FIELD_INFO, 
                 logger=self.logger, SCHEDULER=self.SCHEDULER )
        tcdata = j.output() # Check if the job completed and has good output
        print(tcdata)
        if tcdata:
          prevjob = j
        else:
          del j
      i+=1
    if prevjob is None:
      print("Can't find any valid jobs in "+str(JOBDIR)+"electronic/ ... Wrong jobdir?")
      os.kill()
    print("Last valid job is "+str(JOBDIR)+"electronic/"+str(prevjob.N))
    self.N = prevjob.N+1
    self.prevjob = prevjob
    self.jobs.append(prevjob)
    return prevjob.N

  def grad(self, xyz, ReCn=None, ImCn=None):
    grad_template = copy.deepcopy(self.TDCI_TEMPLATE)
    # overwrite template to do gradient stuff instead of tdci
    #grad_template["run"] = "gradient"
    #grad_template["cassavevectors"] = "yes"
    #grad_template["casgradmult"] = "1 1 1"
    #grad_template["casgradstate"] = "0 1 2"
    grad_template["tdci_grad_init"] = "yes"
    grad_template["tdci_grad_half"] = "no"
    grad_template["tdci_fstrength"] = "0.0"
    grad_template["tdci_simulation_time"] = "0.01"
    grad_template["tdci_nstep"] = "1"
    grad_template["tdci_krylov_end"] = "no"
    j = job(self.N, self.Natoms, self.Nkrylov, ReCn, ImCn, xyz, None, self.JOBDIR, self.JOB_TEMPLATE, grad_template, self.FIELD_INFO, logger=self.logger, SCHEDULER=self.SCHEDULER)
    j.gradjob = True
    j.dir = self.JOBDIR+"electronic/grad/"
    self.prevjob = j
    return j.run_safely()

  def hessian(self, xyz, temp):
    hess_template = copy.deepcopy(self.TDCI_TEMPLATE)
    hess_template["run"] = "frequencies"
    hess_template["to"] = str(temp)
    j = job(self.N, self.Natoms, self.Nkrylov, None, None, xyz, None, self.JOBDIR, self.JOB_TEMPLATE, hess_template, self.FIELD_INFO, logger=self.logger, SCHEDULER=self.SCHEDULER)
    j.dir = self.JOBDIR+"electronic/hessian"+str(self.N)+"/"
    return j.run_safely()
    

  def nextstep(self, xyz, ReCn=None, ImCn=None):
    print("creating job step: "+str(self.N))
    j = job(self.N, self.Natoms, self.Nkrylov, ReCn, ImCn, xyz, self.prevjob, self.JOBDIR, self.JOB_TEMPLATE, self.TDCI_TEMPLATE, self.FIELD_INFO, logger=self.logger, SCHEDULER=self.SCHEDULER)
    output = j.run_safely()
    self.N+=1
    self.prevjob = j
    return output
    

  def halfstep(self, xyz, ReCn=None, ImCn=None):
    halfstep_template = copy.deepcopy(self.TDCI_TEMPLATE)
    halfstep_template["tdci_simulation_time"] = str(float(halfstep_template["tdci_simulation_time"])/2.)
    halfstep_template["tdci_nstep"] = str( int( int(halfstep_template["tdci_nstep"])/2))
    j = job(self.N, self.Natoms, self.Nkrylov, ReCn, ImCn, xyz, self.prevjob, self.JOBDIR, self.JOB_TEMPLATE, halfstep_template, self.FIELD_INFO, logger=self.logger, SCHEDULER=self.SCHEDULER, halfstep=True)
    output = j.run_safely()
    if self.FIELD_INFO["half"]:
      self.N+=0
    else:
      self.N+=1
    self.FIELD_INFO["half"] = (self.FIELD_INFO["half"] + 1) % 2 # Align field with halfsteps
    self.prevjob = j
    return output
    

    





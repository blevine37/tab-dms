#!/usr/bin/python2.7

from __future__ import print_function
import numpy as np
import struct, shutil, os, subprocess, time

FStoAU = 41.341375
EPSILON_C = 0.00265316
E_FIELD_AU = 5.142206707E+11



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
  print(l.split(" "))
  while len(l.split(" "))==4:
    atoms.append(l.split(" ")[0])
    geom.append(l.split(" ")[1]) 
    geom.append(l.split(" ")[2]) 
    geom.append(l.split(" ")[3])
    l = f.readline()
  


def dict_to_file(d, filepath):
  f = open(filepath, 'w')
  for key, value in d.iteritems():
    f.write(str(key)+" "+str(value)+"\n")
  f.close()

def dicts_to_file(dlist, filepath):
  f = open(filepath, 'w')
  for d in dlist:
    for key, value in d.iteritems():
      f.write(str(key)+" "+str(value)+"\n")
  f.close()

# we may need to be careful about endian-ness if this runs on a different machine than TC
def read_bin_array(filepath, length):
  print("l: "+str(length))
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
    print("WARNING: write_bin_array() is for ONE-DIMENSION DOUBLE ARRAYS!! Why are you trying to write the following as one?:")
    print(array)
  f = open(filepath, 'wb')
  f.write( struct.pack('d'*len(array), *array ))
  f.close(); del f

def search_replace_file(filepath, search, replace):
  p = subprocess.Popen('sed -i -e "s+'+search+'+'+replace+'+g" '+str(filepath), shell=True)
  p.wait()
  return 0

def makedirs(dirstr):
  for i in range(2, dirstr.count("/")+1):
    if not os.path.exists("/".join(dirstr.split("/")[0:i])):
      os.mkdir("/".join(dirstr.split("/")[0:i]))
  return os.path.exists(dirstr)


class job:
  def __init__(self, n, Natoms, Nkrylov, ReCn, ImCn, xyzpath, pjob, JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE, FIELD_INFO, SCHEDULER=False):
    self.n = n
    self.Natoms = Natoms
    self.Nkrylov = Nkrylov
    self.ReCn = ReCn # Initial value input by user
    self.ImCn = ImCn
    self.xyzpath = xyzpath
    self.pjob = pjob
    self.dir = JOBDIR+"electronic/"+str(n)+"/"
    self.JOBDIR=JOBDIR
    self.JOB_TEMPLATE=JOB_TEMPLATE
    self.TDCI_TEMPLATE=TDCI_TEMPLATE
    self.SCHEDULER=SCHEDULER
    self.FIELD_INFO = FIELD_INFO
    self.ndets = 0
    self.restarts = 0

  def start(self):
    p = subprocess.Popen( 'bash '+self.dir+'tdci.job', shell=True)
    return p

  def readmisc(self):
    f = open(self.dir+"misc.bin",'rb')
    self.ndets = struct.unpack('i', f.read())[0]
    f.close();del f

  def make_files(self):
    makedirs(self.dir)
    shutil.copy(self.xyzpath, self.dir+self.xyzpath.split("/")[-1]) # copy xyzfile
    shutil.copy(self.JOB_TEMPLATE, self.dir+"/tdci.job")
    time.sleep(2) # make sure file gets copied
    search_replace_file(self.dir+"tdci.job", "temppath", self.dir)
    search_replace_file(self.dir+"tdci.job", "tempname", "test"+str(self.n))
    tempname = "test"+str(self.n)+".in"
    shutil.copy(self.TDCI_TEMPLATE, self.dir+"/"+tempname)
    search_replace_file(self.dir+tempname, "coords.xyz", self.xyzpath.split("/")[-1]) 
    self.make_fieldfiles()
    if self.n==0:
      search_replace_file(self.dir+tempname, "tdci_diabatize_orbs yes", "tdci_diabatize_orbs no")
      if type(self.ReCn) == type(None):
	search_replace_file(self.dir+tempname, "tdci_recn_readfile recn_init.bin", "")
      else:
	write_bin_array(self.ReCn,self.dir+"recn_init.bin")
      if type(self.ImCn) == type(None):
	search_replace_file(self.dir+tempname, "tdci_imcn_readfile imcn_init.bin", "")
      else:
	write_bin_array(self.ImCn,self.dir+"imcn_init.bin")

      search_replace_file(self.dir+tempname, "tdci_prevorbs_readfile PrevC.bin", "")
      search_replace_file(self.dir+tempname, "tdci_prevcoords_readfile PrevCoors.bin", "")
      search_replace_file(self.dir+tempname, "tdci_krylov_init cn_krylov_init.bin", "")

    else: # Copy Prev Orbitals and Coords (in double4) for orbital diabatization
      pjobd = self.pjob.dir
      shutil.copy(pjobd+"/NewCoors.bin", self.dir+"/PrevCoors.bin")
      shutil.copy(pjobd+"/NewC.bin", self.dir+"/PrevC.bin")
      if type(self.ReCn) == type(None):
	shutil.copy(pjobd+"/ReCn_end.bin", self.dir+"/recn_init.bin")
      else:
	write_bin_array(self.ReCn,self.dir+"recn_init.bin")
      if type(self.ImCn) == type(None):
	shutil.copy(pjobd+"/ImCn_end.bin", self.dir+"/imcn_init.bin")
      else:
	write_bin_array(self.ImCn,self.dir+"imcn_init.bin")
      
      if self.FIELD_INFO["krylov_end"]:
        shutil.copy(pjobd+"/Cn_krylov_end.bin", self.dir+"/cn_krylov_init.bin")

  def clean_files(self):
    if os.path.exists(self.dir):
      shutil.rmtree(self.dir)

  def make_fieldfiles(self):
    # Field file should include values for half-steps, so the length of the array
    #   should be 2*nsteps!
    FStoAU = 41.341375
    T = self.FIELD_INFO["tdci_simulation_time"]
    N = self.FIELD_INFO["nstep"]
    t = np.linspace( self.n*T*FStoAU, (self.n+1)*T*FStoAU, 2*N)
    for i in range(0, self.FIELD_INFO["nfields"]):
      vals = self.FIELD_INFO["f"+str(i)](t)
      #fieldfile = open(self.dir+"field"+str(i)+".csv", "w")
      #for v in vals:
      #  fieldfile.write( '{:11.8e}'.format(v) + "," )
      #fieldfile.close()
      write_bin_array(vals,self.dir+"field"+str(i)+".bin")

  def check_output(self,output):
    outputgood = True
    norm = np.linalg.norm(output["recn"])**2 + np.linalg.norm(output["imcn"])**2
    print("Final wfn norm (MO basis): "+str(norm))
    if ((norm<0.7) or (norm>1.1) or (np.isnan(norm))):
      print("ERROR: Norm out of bounds")
      outputgood = False
    print("Sum of gradient elements: "+str(np.sum(output["grad"])))
    if (np.isnan(np.sum(output["grad"]))):
      print("ERROR: nan in gradient")
      outputgood = False
    if self.FIELD_INFO["krylov_end"]:
      norm = np.sum(output["recn_krylov"]**2) + np.sum(output["imcn_krylov"]**2)
      print("Final wfn norm (AES basis): "+str(norm))
      krylov_MO_Re = np.matmul(np.transpose(output["krylov_states"]), output["recn_krylov"])
      krylov_MO_Im = np.matmul(np.transpose(output["krylov_states"]), output["recn_krylov"])
      print("Checking AES basis quality:")
      print("ReCn:")
      #print(output["recn"])
      #print("ImCn:")
      #print(output["imcn"])
      #print("ReCn (AES->MO):")
      #print(krylov_MO_Re)
      #print("ImCn (AES->MO):")
      #print(krylov_MO_Im)
      overlap = np.dot(krylov_MO_Re,output["recn"])**2 + np.dot(krylov_MO_Im,output["imcn"])**2
      print("Overlap of AES-MO with MO:"+str(overlap))
    if outputgood:
      return True
    else:
      return False


  def run_safely(self):
    # Run the job
    retries = 0
    while (retries < 20):
      self.clean_files()
      self.make_files()
      p = self.start()
      print("Started "+str(self.dir)+"\n")
      finished = 0
      # Periodically check if the process is finished
      while not finished:
        time.sleep(5)
        if self.check_status(p):
          finished = True
      # Make sure output is good
      output = self.output()
      if output: # Everything checks out!
        print("Output looks good!")
        return output
      else: # Outputs bad, try redoing the job!
        print("Output is bad. Restarting the job.")
        makedirs("badjobs/"+str(self.n)+"_"+str(retries))
        shutil.copytree( self.dir, "badjobs/"+str(self.n)+"_"+str(retries))
      retries+=1
    print("Went through 20 retries and output is still bad T_T\n")
    return output

      
  def check_status(self, p):
    if self.SCHEDULER:
      # TODO: Need to call squeue and decipher it's output
      pass
    else: 
      result = p.poll()
      print(result)
      if result != None: # returns None if still running
        print("Done!\n")
        return True
      else:
        print("Still running...\n")
        return False

  # key : list of words that match the beginning of line.split()
  # For ndets, ["Number", "of", "determinants:"]
  # pos : the index of the element to be returned from matching line.split()
  def scan_outfile(self, key, pos):
    f = open(self.dir+"test"+str(self.n)+".out", 'r')
    l = f.readline()
    while l != "":
      if (len(l.split()) > len(key)):
        if (l.split()[:len(key)] == key):
          return l.split()[pos]
      l = f.readline()
    print("key "+str(key)+" not found :( ")
    return None


  def output(self):
    filesgood = True
    files = ["ReCn_end.bin","ImCn_end.bin", "tdcigrad.bin", "misc.bin"]
    if self.FIELD_INFO["krylov_end"]:
      files += ["ReCn_krylov_end.bin", "ImCn_krylov_end.bin", "Cn_krylov_end.bin", "E_krylov_end.bin", "tdcigrad_krylov.bin"]
    for fn in files:
      if not os.path.exists(self.dir+fn):
	filesgood = False
	print("ERROR: "+fn+" missing")
    if not filesgood:
      return False

    self.readmisc()
    eng = self.scan_outfile(["Final", "TDCI", "Energy:"], 3)
    grad = read_bin_array(self.dir+"tdcigrad.bin", 3*self.Natoms)
    grad.resize((self.Natoms, 3))
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
               "recn_krylov": recn_krylov,      # 1d array, 2*krylov_sub_n
               "imcn_krylov": imcn_krylov,      # 1d array, 2*krylov_sub_n
               "krylov_states": krylov_states,  # 2d array of CI vectors of each approx eigenstate
               "krylov_energies": krylov_energies, # 1d array of energies of each approx eigenstate
               "krylov_gradients": krylov_gradients # 3d array of approx eigenstate gradients, Napprox x Natoms x 3 dim.
             }

    print("TDCI job Output:\n")
    print(output)

    if self.check_output(output):
      #self.sanity_test(output)
      return output
    else:
      return False

  def sanity_test(self, output):
    print("Sanity test on output...")
    Cn_approx_end = None
    Qt_end = None
    if (os.path.exists(self.dir+"Cn_approx_end.bin")) and (os.path.exists(self.dir+"Qt_end.bin")):
      print("Extra debug files present:")
      Cn_approx_end = read_bin_array(self.dir+"Cn_approx_end.bin", self.Nkrylov**2)
      Cn_approx_end.resize((self.Nkrylov,self.Nkrylov))
      Qt_end = read_bin_array(self.dir+"Qt_end.bin", self.Nkrylov*self.ndets)
      Qt_end.resize((self.Nkrylov,self.ndets))
      Cn_krylov = np.matmul(Cn_approx_end, Qt_end)
      if np.allclose(Cn_krylov, output["krylov_states"]):
        print("Pass! Cn_approx_end * Qt_end = krylov_states")
      else:
        print("Fail! Cn_approx_end * Qt_end != krylov_states")
    else:
      print("Extra debug files not present.")
    if np.allclose(np.matmul(output["krylov_states"],output["recn"]),output["recn_krylov"]):
      print("Pass! recn_krylov == krylov_states * recn")
    else:
      print("Fail! recn_krylov != krylov_states * recn")

    import pickle
    with open("data.pickle", 'wb') as f:
      pickle.dump([Cn_approx_end, Qt_end, output], f, pickle.HIGHEST_PROTOCOL)
    print("Sanity test finished.")
      


class tccontroller:
  def __init__(self, JOBDIR, JOB_TEMPLATE, TDCI_TEMPLATE, FIELD_INFO, SCHEDULER=False):
    self.N = 0
    self.jobs = []
    self.prevjob = None
    self.JOBDIR=JOBDIR
    self.JOB_TEMPLATE=JOB_TEMPLATE
    self.TDCI_TEMPLATE=TDCI_TEMPLATE
    self.SCHEDULER=SCHEDULER
    self.FIELD_INFO=FIELD_INFO
    self.Natoms = None
    self.Nkrylov = 2*FIELD_INFO["krylov_end_n"]


  def nextstep(self, xyzpath, ReCn=None, ImCn=None):
    #print("nextstep: "+str(self.N)+"\n")
    if self.N == 0:
      f = open(xyzpath, 'r')
      self.Natoms = int(f.readline()) # Get the number of atoms so we know the dims of the gradient
      f.close();del f
      j = job(self.N, self.Natoms, self.Nkrylov, ReCn, ImCn, xyzpath, None, self.JOBDIR, self.JOB_TEMPLATE, self.TDCI_TEMPLATE, self.FIELD_INFO, self.SCHEDULER)
    else:
      j = job(self.N, self.Natoms, self.Nkrylov, ReCn, ImCn, xyzpath, self.jobs[-1], self.JOBDIR, self.JOB_TEMPLATE, self.TDCI_TEMPLATE, self.FIELD_INFO, self.SCHEDULER)
    self.jobs.append(j)
    self.N+=1
    #print("appended N, it's "+str(self.N)+" now :D\n")
    #j.run_safely()
    #return j.output()
    return j.run_safely()
    

    

    





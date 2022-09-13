#!/usr/bin/python2.7

import numpy as np
import math
import sys, struct
import os, time, shutil, subprocess

import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

c=2.99792458*10**10 # in cm/s
h=4.1357*10**-15 # in eV*s
bohrtoangs = 0.529177210903


# we may need to be careful about endian-ness if this runs on a different machine than TC
def read_bin_array(filepath, length):
  #print("l: "+str(length))
  if length == 0:
    return np.array([])
  f = open(filepath, 'rb')
  return np.array(struct.unpack('d'*length, f.read()))

def dist(w, v):
  if (len(w) != len(v)):
    print("dist() ERROR: "+str(w)+" and "+str(v)+" lens do not match")
    return 0
  tot = 0
  for i in range(0,len(w)):
    tot += (w[i] - v[i])**2
  return np.sqrt(tot)

# Takes the distance between atoms in subsequent frames, sums it over all atoms
#   as a general metric of how much the structure has changed in one step
def calc_xyz_distances(xyzs):
  output = [0]
  for i in range(0,len(xyzs)-1):
    tot = 0
    for atom in range(0,len(xyzs[0])):
      tot += dist( xyzs[i][atom] , xyzs[i+1][atom] )
    output.append(tot)
  #output.append(0.0)
  print(len(output))
  return output




# plot hd5f stuff
def h5py_plot(steptime):

    print("Plotting PE/KE in ehrenfest.png")
    # Open h5py file
    h5f = h5py.File('data2.hdf5', 'r')

    #import pdb; pdb.set_trace()
    # Get number of iterations
    niters = h5f['geom'].shape[0]

    # Get number atoms
    natoms = h5f['geom'].shape[1]


    xyzdist = calc_xyz_distances(h5f['geom'])

    # Iterate and print energies
    poten = h5f['poten']
    kinen = h5f['kinen']
    tot = []
    #print(('{:>25s}'*3).format('Potential', 'Kinetic', 'Total'))
    for it in range(0, niters):
        pot = poten[it]
        kin = kinen[it]
        tot.append(pot + kin)
        #print(('{:25.17f}'*3).format(pot, kin, tot[-1]))
    #print("")

    eV = 27.2114
    test = poten
    asdf = [np.abs(test[i]-test[i+1]) for i in range(0,len(test)-1)]
    m = max(asdf)
    print("max poten deviation at step "+str(asdf.index(m))+": "+str(m*eV))
    poten = np.array(poten)
    kinen = np.array(kinen)
    tot = np.array(tot)
    emin = min(poten)
    poten = eV*(poten-emin)
    kinen = eV*(kinen-min(kinen))
    tot = eV*(tot-emin)

    test = tot
    asdf = [np.abs(test[i]-test[i+1]) for i in range(0,len(test)-1)]
    m = max(asdf)
    print("max totenergy deviation at step "+str(asdf.index(m))+": "+str(m))
        
    #print(poten)
    #print(kinen)
    #print(tot)

    X = np.array(range(0,niters)) * steptime/1000.  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Rel. E (eV)')
    ax.plot(X, poten, label="Potential", marker="^", linewidth=1.2)
    ax.plot(X, kinen, label="Kinetic", marker="v", linewidth=1)
    ax.plot(X, tot, label="Total", marker="o", linewidth=1)
    ax.plot([], [], label=r'$\sum \Delta x_i$', marker="o", color="tab:red") # Need this for legend
    ax_dist = ax.twinx() 
    ax_dist.set_ylabel("Distance (Angstrom)", color="tab:red")
    ax_dist.plot(X, xyzdist, label=r'$\sum \Delta x_i$', marker="o", markersize=4, linewidth=1, color="tab:red")
    ax.legend()
    #ax_dist.legend()
    plt.tight_layout()
    plt.savefig("ehrenfest.png", dpi=800, bbox_inches = "tight")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Rel. E (eV)')
    ax2.plot(X, tot, label="Total Energy", marker="o", linewidth=0.4)
    ax2.legend()
    plt.tight_layout()
    #plt.xlim([0,12])
    plt.savefig("ehrenfest_toteng.png", dpi=800, bbox_inches="tight")

    # Close
    h5f.close()
    return( (X, poten, kinen, tot) )

# should return a red->blue scale in rgb tuples
def rgb_linspace(n):
  a = np.linspace(0, 2, n)
  rgbs = []
  for i in range(0,n):
    if a[i] == 0:
      rgbs.append((1.,0.,0.))
    elif a[i] < 1:
      rgbs.append((1-a[i],a[i],0))
    elif a[i] == 1:
      rgbs.append((0.,1.,0.))
    elif a[i] < 2:
      rgbs.append((0.,2-a[i],a[i]-1))
    elif a[i] == 2:
      rgbs.append((0.,0.,1.))
  return rgbs

def plot_populations(nstates, nsteps, steptime):
  print("Plotting state populations...")
  pops = []
  for i in range(0,nstates):
    pops.append([])
  # Read populations from first step of each TDCI calc
  for i in range(0,nsteps):
    f = open("electronic/"+str(i)+"/Pop", 'r')
    l = (f.readline()).split(",")
    #steptime = float(l[0])
    for j in range(0,nstates):
      pops[j].append(float(l[j+1])) # first element in l is time
    f.close()
  # Plot 'em
  fig = plt.figure()
  ax = fig.add_subplot(111)
  rgbs = rgb_linspace(nstates)
  print(rgbs)
  print("len(pops[0]): "+str(len(pops[0])))
  X = np.array(range(0,len(pops[0]))) * steptime/1000.  
  print(len(X))
  print(max(X))
  for i in range(0,nstates):
    ax.plot(X,pops[i], label="S"+str(i), marker="o", markersize="3", linewidth=1.2, color=rgbs[i])
  
  ax.set_ylabel('Population')
  ax.set_xlabel("Time (fs)")
  #plt.xlim([0,12])
  ax.legend(loc="upper right", fontsize="xx-small")
  plt.savefig("Pops.png", dpi=800, bbox_inches='tight')
  

# For MOLDEN 
def make_xyz_series(nsteps):
  print("Making trajectory.xyz...")
  outstring = ""
  for i in range(0,nsteps,2):
    f = open("electronic/"+str(i)+"/temp.xyz",'r')
    for line in f:
      outstring += line
    f.close()
  f = open("trajectory.xyz",'w')
  f.write(outstring)

# VMD, ffmpeg
def render_trajectory(nsteps):
  print("Rendering trajectory to mp4...")
  if os.path.exists("xyzs/"):
    shutil.rmtree("xyzs/")
    os.makedirs("xyzs/")
  else:
    os.makedirs("xyzs/")
  if os.path.exists("bmp/"):
    shutil.rmtree("bmp/")
    os.makedirs("bmp/")
  else:
    os.makedirs("bmp/")

  # copy xyz files to directory
  j = 0
  for i in range(0,nsteps,2):
    shutil.copy("electronic/"+str(i)+"/temp.xyz", "xyzs/"+"{:04d}".format(j)+".xyz")
    j+=1

  # write VMD script
  # Keeping the script in here so we can modify camera parameters from python if we want
  vmdtxt = "# Script from Arshad Mehmood\n"+\
	    "color Display Background white\n"+\
	    "display resize 960 960\n"+\
	    "set isoval 0.25\n"+\
	    "axes location Off\n"+\
	    "for {set i 0} {$i<="+str(nsteps)+"} {incr i} {\n"+\
	    "set name [format %04d $i]\n"+\
	    "puts \"Processing $name.xyz...\"\n"+\
	    "mol default style CPK\n"+\
	    "mol new ./xyzs/$name.xyz\n"+\
	    "scale to 0.45\n"+\
	    "rotate y by 0.00000\n"+\
	    "translate by 0.000000 0.00000 0.000000\n"+\
	    "mol modstyle 0 top CPK 0.500000 0.300000 50.000000 50.000000\n"+\
	    "mol modcolor 0 top Element\n"+\
	    "color Element C gray\n"+\
	    "mol addrep top\n"+\
	    "mol modstyle 1 top Isosurface $isoval 0 0 0 1 1\n"+\
	    "mol modcolor 1 top ColorID 1\n"+\
	    "mol modmaterial 1 top AOShiny\n"+\
	    "material change opacity AOShiny 0.350000\n"+\
	    "material change transmode AOShiny 1.000000\n"+\
	    "display cuemode Linear\n"+\
	    "display cuestart 3.000000\n"+\
	    "render TachyonInternal bmp/$name.bmp\n"+\
	    "mol delete top\n"+\
	    "}"
  f = open("vmd.tcl", 'w')
  f.write(vmdtxt)
  f.close()
  # render xyz's into bmps
  vmdp = subprocess.Popen("vmd -dispdev text -eofexit < vmd.tcl > output.log", shell=True)
  vmd_endcode = vmdp.wait()
  print("vmd endcode: "+str(vmd_endcode))
  # render bmps into an mp4
  ffmpegp = subprocess.Popen("ffmpeg -y -r 20 -i bmp/%04d.bmp -c:v libx264 -preset slow -crf 18 "+
                             " -force_key_frames source -x264-params keyint=4:scenecut=0 -pix_fmt yuv420p trajectory.mp4", shell=True)
  ffmpeg_endcode = ffmpegp.wait()
  print("ffmpeg endcode: "+str(ffmpeg_endcode))
  # clean up files
  #os.remove("vmd.tcl")
  shutil.rmtree("bmp/")
  shutil.rmtree("xyzs/")
  return 0

def plot_state_energies(nstates, nsteps,steptime):
  print("Getting state energy data...")
  energies = []
  energies_postdiab = []
  for i in range(0,nstates):
    energies.append([])
    energies_postdiab.append([])
  for i in range(0, nsteps):
    e = read_bin_array("electronic/"+str(i)+"/States_E.bin", nstates)
    if (i==0):
      e_post = e
    else:
      e_post = read_bin_array("electronic/"+str(i)+"/States_E_postdiab.bin", nstates)
    for j in range(0, nstates):
      energies[j].append(e[j])
      energies_postdiab[j].append(e_post[j])

  X = np.array(range(0,nsteps)) * steptime/1000.  

  """ # Use this to make a pre and post diab plot for each state, basically checks validity of states after diabatization
  # shift energy and change to eV
  energies_shifted = np.copy(energies)
  energies_postdiab_shifted = np.copy(energies_postdiab)
  for i in range(0, nstates):
    emin = min(energies[i])
    epmin = min(energies_postdiab[i])
    for j in range(0, nsteps):
      #print((i,j))
      #print np.array(energies).shape
      energies_shifted[i][j] = 27.2114*(energies[i][j]-emin)
      energies_postdiab_shifted[i][j] = 27.2114*(energies_postdiab[i][j]-epmin)

  for i in range(0, nstates):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X,energies[i], label="Pre Diab", marker="^", linewidth=0.7)
    ax.plot(X,energies_postdiab[i], label="Post Diab", marker="v", linewidth=0.4)
    ax.legend()
    plt.savefig("energies"+str(i)+".png", dpi=800, bbox_inches='tight')
  """
  # TODO: Do a standalone CASCI every nth step to generate these plots, while propagation will only solve for 1 state.


  MakeEStatesPlot = True
  if MakeEStatesPlot:
    print("Plotting state energies...")
    # shift to relative energy
    emin = min(energies_postdiab[0])
    # Plot all state energies
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rgbs = rgb_linspace(nstates)
    print(rgbs)
    for i in range(0,nstates):
      ax.plot(X, np.array(energies_postdiab[i])-emin, label="S"+str(i), marker="o", markersize="3", linewidth=1.2, color=rgbs[i])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Rel. E (V)')
    ax.legend()
    plt.savefig("Estates.png", dpi=800, bbox_inches='tight')
  
  MakeEDiffPlot = True
  if MakeEDiffPlot:
    print("Plotting state energy differences...")
    # Plot Gaps (Are we at a concial intersection/avoided crossing?)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1,nstates):
      ax.plot(X, np.array(energies_postdiab[i])-np.array(energies_postdiab[i-1]), label="S"+str(i)+"-S"+str(i-1), marker="o", markersize="3", linewidth=1.2, color=rgbs[i])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('E (eV)')
    ax.legend()
    plt.savefig("Egaps.png", dpi=800, bbox_inches='tight')


# detects if the molecule is h2o and calculates the O-H bond distance over time
def h2o_bond(steptime):

  from mpl_toolkits.axes_grid1 import host_subplot
  import mpl_toolkits.axisartist as AA

  # Calculate bond distance
  """
  f = open("trajectory.xyz", 'r')
  lines = []
  for l in f: lines.append(l)
  t = int(lines[0])
  if t != 3:
    print("not h2o!! !=3")
    return False
  test = [lines[2].split()[0], lines[3].split()[0], lines[4].split()[0]]
  print(test)
  if test != ["O", "H", "H"]:
    print("not h2o!!")
    return False
  # ok its h2o
  steps = len(lines)/5


  X = np.array(range(0,steps)) * steptime/1000.  
  Y = []
  for i in range(0,steps):
    print((len(lines), i, i*5+2, i*5+5))
    x0,y0,z0 = map(float, lines[(i)*5+2].split()[1:] )
    x1,y1,z1 = map(float, lines[(i)*5+3].split()[1:] )
    d = np.sqrt( (x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2 )
    Y.append(d)

  """

  f = h5py.File('data2.hdf5', 'r')
  steps = f['geom'].shape[0]
  natoms = f['geom'].shape[1]
  if natoms != 3:
    print("not h2o!! !=3")
    return False
  
  X = np.array(range(0,steps)) * steptime/1000.  
  Y = [] # O-H bond distance
  Ya = [] # relative acceleration between O and H
  for i in range(0, steps):
    xO,yO,zO = f['geom'][i][0]*bohrtoangs
    xH,yH,zH = f['geom'][i][1]*bohrtoangs
    dvec = [ xH-xO, yH-yO, zH-zO ]
    d = np.sqrt( (xH-xO)**2 + (yH-yO)**2 + (zH-zO)**2 )
    Y.append(d)
    axO,ayO,azO = f['accs'][i][0]*bohrtoangs
    axH,ayH,azH = f['accs'][i][1]*bohrtoangs
    avec = [ axH-axO, ayH-ayO, azH-azO ]
    a = np.sqrt( (axH-axO)**2 + (ayH-ayO)**2 + (azH-azO)**2 )
    Ya.append(a)
    print(avec, a)
    
  # Plot
  #fig = plt.figure()
  #ax = fig.add_subplot(111)
  ax = host_subplot(111, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)
  # Bond distance
  ax.plot(X, np.array(Y), label="O-H distance", marker="o", markersize="2", linewidth=0.8)
  #ax.plot([], [], label="O-H rel acceleration", marker="^", markersize="2", linewidth=0.8, color="tab:blue")
  #ax.plot([],[], label="Potential", marker="v", color="tab:red") # for legend
  ax.set_xlabel('Time (fs)')
  ax.set_ylabel('O-H Bond Distance (Angstrom)')
  ax2 = ax.twinx()
  ax2.set_ylabel("E (eV)", color="tab:red")
  ax2.plot(X, 27.2114*(np.array(f['poten'])-min(f['poten'])), label="Potential", marker="v", markersize="2", linewidth=0.8, color="tab:red")
  ax2.axis["right"].label.set_color("tab:red")

  offset = 0
  new_fixed_axis = ax2.get_grid_helper().new_fixed_axis
  ax2.axis["right"] = new_fixed_axis(loc="right",
				      axes=ax2,
				      offset=(offset, 0))
  ax3 = ax.twinx()
  offset = 60
  new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
  ax3.axis["right"] = new_fixed_axis(loc="right",
				      axes=ax3,
				      offset=(offset, 0))

  ax3.axis["right"].toggle(all=True)

  ax3.set_ylabel("Relative Accel (Angstrom)", color="tab:blue")
  ax3.plot(X, np.array(Ya), label="O-H rel acceleration", marker="^", markersize="2", linewidth=0.8, color="tab:blue")
  ax.legend()
  plt.tight_layout()
  plt.savefig("OHbond.png", dpi=800, bbox_inches='tight')
  f.close()
  
    
  

h5f = h5py.File('data2.hdf5', 'r')
#import pdb; pdb.set_trace()
steptime = h5f['time'][1] - h5f['time'][0]
h5f.close()
print("Steptime: "+str(steptime))

#steptime= 241.8/2 # delta * autimetosec * 10+18, attoseconds
nsteps = len(os.listdir("electronic/"))-2 # grad folder + last one might not have finished if you killed the job
nstates = 3
print("nsteps: "+str(nsteps))

X, poten, kinen, tot = h5py_plot(steptime)

make_xyz_series(nsteps)
h2o_bond(steptime)

plot_state_energies(nstates, nsteps,steptime)
plot_populations(nstates,nsteps,steptime)



render_trajectory(nsteps)




#!/usr/bin/python2.7

import numpy as np
import math
import sys, struct
import os, shutil, subprocess

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
def h5py_plot():

    # Open h5py file
    h5f = h5py.File('data.hdf5', 'r')

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
    print("")

    steptime= 241.8 # delta * autimetosec * 10+18, attoseconds
    eV = 27.2114
    poten = np.array(poten)
    kinen = np.array(kinen)
    tot = np.array(tot)
    emin = min(poten)
    poten = eV*(poten-emin)
    kinen = eV*(kinen-min(kinen))
    tot = eV*(tot-emin)
        
    print(poten)
    print(kinen)
    print(tot)

    X = np.array(range(0,niters)) * steptime/1000.  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Rel. E (eV)')
    ax.plot(X, poten, label="Potential", marker="^", linewidth=1.2)
    ax.plot(X, kinen, label="Kinetic", marker="v", linewidth=1)
    ax.plot(X, tot, label="Total", marker="o", linewidth=1)
    ax.plot([], [], label=r'$\sum \Delta x_i$', marker="o", color="tab:red")
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
    plt.savefig("ehrenfest_toteng.png", dpi=800, bbox_inches="tight")

    # Close
    h5f.close()

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

def plot_populations(nstates, nsteps):
  pops = []
  steptime= 241.8
  for i in range(0,nstates):
    pops.append([])
  # Read populations from first step of each TDCI calc
  for i in range(0,nsteps):
    f = open("electronic/"+str(i)+"/Pop", 'r')
    l = (f.readline()).split(",")
    steptime = float(l[0])
    for j in range(0,nstates):
      pops[j].append(float(l[j+1])) # first element in l is time
    f.close()
  # Plot 'em
  fig = plt.figure()
  ax = fig.add_subplot(111)
  rgbs = rgb_linspace(nstates)
  print(rgbs)
  for i in range(0,nstates):
    ax.plot(steptime*np.array(range(0,nsteps)),pops[i], label="S"+str(i), marker="o", markersize="3", linewidth=1.2, color=rgbs[i])
  
  ax.set_ylabel('Population')
  ax.set_xlabel("Time (fs)")
  plt.xlim([0,12])
  ax.legend(loc="upper right", fontsize="xx-small")
  plt.savefig("Pops.png", dpi=800, bbox_inches='tight')
  

# For MOLDEN 
def make_xyz_series(nsteps):
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
  ffmpegp = subprocess.Popen("ffmpeg -y -r 20 -i bmp/%04d.bmp -c:v libx264 -preset slow -crf 18 trajectory.mp4", shell=True)
  ffmpeg_endcode = ffmpegp.wait()
  print("ffmpeg endcode: "+str(ffmpeg_endcode))
  # clean up files
  #os.remove("vmd.tcl")
  shutil.rmtree("bmp/")
  shutil.rmtree("xyzs/")
  return 0

def plot_state_energies(nstates, nsteps):
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
  #print(energies)
  #print(np.array(energies).shape)
  # shift energy and change to eV
  for i in range(0, nstates):
    emin = min(energies[i])
    epmin = min(energies[i])
    for j in range(0, nsteps):
      #print((i,j))
      #print np.array(energies).shape
      energies[i][j] = 27.2114*(energies[i][j]-emin)
      energies_postdiab[i][j] = 27.2114*(energies_postdiab[i][j]-epmin)

  X = range(0, nsteps)
  for i in range(0, nstates):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X,energies[i], label="Pre Diab", marker="^", linewidth=0.7)
    ax.plot(X,energies_postdiab[i], label="Post Diab", marker="v", linewidth=0.4)
    ax.legend()
    plt.savefig("energies"+str(i)+".png", dpi=800, bbox_inches='tight')


nsteps = len(os.listdir("electronic/"))-2 # last one might not have finished if you killed the job
nstates = 8 
print("nsteps: "+str(nsteps))

make_xyz_series(nsteps)

plot_state_energies(nstates, nsteps)
plot_populations(nstates,nsteps)

h5py_plot()

render_trajectory(nsteps)




#!/usr/bin/python2.7

from analysis_lib import *
import rmsd


matplotlib.use("Agg")


np.set_printoptions(precision=8, linewidth=999999999)

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

maxstep = 5166

datas_process = [
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/1/", "1", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/2/", "2", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/3/", "3", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/4/", "4", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/5/", "5", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/6/", "6", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/7/", "7", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/8/", "8", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/9/", "9", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/10/", "10", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/11/", "11", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/12/", "12", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/13/", "13", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/14/", "14", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/15/", "15", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/16/", "16", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/17/", "17", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/18/", "18", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/19/", "19", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/20/", "20", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
#plottables( "/home/adurden/4tb/seawulf_jobs/benzene_300K2/21/", "21", DoStateProjections=True, DoSDiagnostic=True, maxsteps=maxstep ),
]


plt.clf()
# Window range in femtoseconds
#start, end = 25.8,26
start, end = 0, 1000


for data in datas_process:
  data.starti, data.endi = data.fs2index_range(start,end)
  print((data.starti, data.endi))
  print((data.time[data.endi]))

print(datas_process[0].fs2index_range(240,250))

#print((datas[0].starti, datas[0].endi))


def plot_traj(data, outname):
  plt.clf()
  fig = plt.figure()
  fig.set_figheight(14)
  fig.set_figwidth(8)
  ax1 = fig.add_subplot(611)
  ax2 = fig.add_subplot(612)
  ax3 = fig.add_subplot(613)
  ax4 = fig.add_subplot(614)
  ax5 = fig.add_subplot(615)
  ax6 = fig.add_subplot(616)



  #ax4.title.set_text('RMS gradient')
  #ax4.plot(data.time[data.starti:data.endi], data.rmsgrad[data.starti:data.endi], '-', label=data.label)
  ax1.title.set_text('Relative Potential vs Kinetic Energy')
  # Make PE relative
  data.pe = np.array(data.pe) - min(data.pe)
  ax1.plot(data.time[data.starti:data.endi], data.pe[data.starti:data.endi], '-', label='PE')
  ax1.plot(data.time[data.starti:data.endi], data.ke[data.starti:data.endi], '-', label='KE')
  ax1.set_ylabel("eV")
	       

  ax2.title.set_text("FOMO Orbital Energies")
  if data.DoFOMO:
    for i in range(data.clsd, data.clsd+data.acti+2): # Include an extra orbital on either side
      ax2.plot(data.time[data.starti:data.endi], data.fomo_eng[i][data.starti:data.endi], '-', label=str(i))
  #ax2.legend()
  ax2.set_ylabel("eV")

  ax3.title.set_text("State Energies")
  min_e = min([min(data.state_eng[i][data.starti:data.endi]) for i in range(0,3)])
  for i in range(0,3):
    data.state_eng[i] = 27.2114*(np.array(data.state_eng[i])-min_e) 
    ax3.plot(data.time[data.starti:data.endi], data.state_eng[i][data.starti:data.endi], '-', label=r'S$_'+str(i)+'$')
  #ax3.legend()
  ax3.set_ylabel("eV")


  ax4.title.set_text("State Projections")
  if data.DoStateProjections:
    for i in range(0,data.nstates):
      ax4.plot(data.time[data.starti:data.endi], data.state_proj[i][data.starti:data.endi], '-', label=r'S$_'+str(i)+'$')
  ax4.legend(loc="upper right")

  ax5.title.set_text('Relative Total Energy')
  shifted_reltot = data.reltot[data.starti:data.endi]-data.reltot[data.starti]
  #ax5.plot(data.time[data.starti:data.endi], data.reltot[data.starti:data.endi], 'o', markersize=1)
  ax5.plot(data.time[data.starti:data.endi], shifted_reltot, 'o', markersize=1)
  #ax5.legend()
  ax5.set_ylabel("eV")

  ax6.title.set_text(r"$S^2$ Closed-Active Off-Diagonals")
  if data.DoSDiagnostic:
    ax6.plot(data.time[data.starti:data.endi], data.S_sq_oos[data.starti:data.endi], '-')
  #ax6.legend()     
  ax6.set_xlabel("Time (fs)")


  matplotlib.use("Agg")
  plt.tight_layout()
  plt.savefig(outname, dpi=200)




def plot_ci_rmsd(data, outname):
  rmsd_s0x1 = []
  ci_0x1_path = "/home/adurden/conda/benzene/ciopt_0_1/ci_0x1.xyz"
  ci_0x1 = np.array(xyz_read(ci_0x1_path)[1])
  for i in range(data.starti+1,data.endi+1):
    temp = np.array(xyz_read( data.d+"/electronic/"+str(i)+"/temp.xyz")[1])
    rmsd_s0x1.append( rmsd.rmsd( ci_0x1, temp) )
  rmsd_s0x1 = np.array(rmsd_s0x1)

  rmsd_s1x2 = []
  ci_1x2_path = "/home/adurden/conda/benzene/ciopt_1_2/ci_1x2.xyz"
  ci_1x2 = np.array(xyz_read(ci_1x2_path)[1])
  for i in range(data.starti+1,data.endi+1):
    temp = np.array(xyz_read( data.d+"/electronic/"+str(i)+"/temp.xyz")[1])
    rmsd_s1x2.append( rmsd.rmsd( ci_1x2, temp) )
  rmsd_s1x2 = np.array(rmsd_s1x2)

  plt.clf()
  fig = plt.figure()
  fig.set_figheight(7)
  fig.set_figwidth(4)
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax1.title.set_text("RMSD to Conical Intersections")
  ax1.plot(data.time[data.starti:data.endi], rmsd_s0x1[data.starti:data.endi], '-', label=r'RMSD $CI(S_0,S_1)$')
  ax1.plot(data.time[data.starti:data.endi], rmsd_s1x2[data.starti:data.endi], '-', label=r'RMSD $CI(S_1,S_2)$')
  ax1.set_ylabel("RMSD (Angstrom)")
  ax1.legend()

  ax2.title.set_text("State Projections")
  if data.DoStateProjections:
    for i in range(0,data.nstates):
      ax2.plot(data.time[data.starti:data.endi], data.state_proj[i][data.starti:data.endi], '-', label=r'S$_'+str(i)+'$')
  ax2.legend(loc="upper right")
  ax2.set_xlabel("Time (fs)")

  plt.tight_layout() 
  plt.savefig(outname, dpi=255)

  print(data.endi)
  print(data.state_proj[0][data.endi])
  print(sum([data.state_proj[0][data.endi], data.state_proj[1][data.endi], data.state_proj[2][data.endi] ]))


def plot_average_statepop(datas, outname):
  # This plot wont work if you don't have the same timesteps on all your trajectories...

  # Plot averages
  fig = plt.figure()
  fig.set_figheight(4)
  fig.set_figwidth(7)

  last_endi = min([x.endi for x in datas] )
  print(last_endi)

  avgTot = np.mean( np.array([a.tot[a.starti:last_endi] for a in datas]), axis=0 )
  avgPE = np.mean( np.array([a.pe[a.starti:last_endi] for a in datas]), axis=0 )
  avgKE = np.mean( np.array([a.ke[a.starti:last_endi] for a in datas]), axis=0 )

  if len(datas) > 1:
    min_e = min( [min(x) for x in [z.tot for z in datas] ]  )
    #print(min_e)
    for data in datas:
      data.reltot = [ (z-min_e)*27.2114 for z in data.tot ]
    avgTot = [ (z-min_e)*27.2114 for z in avgTot ]


  avgstate_proj = []
  for i in range(0,datas[0].nstates):
    avgstate_proj.append( np.mean( np.array( [ a.state_proj[i][a.starti:last_endi] for a in datas] ), axis=0 ) )

  rgb = rgb_linspace( datas[0].nstates)

  ax2 = fig.add_subplot(111)
  ax2.title.set_text("State Projections")
  for data in datas:
    if data.DoStateProjections:
      for i in range(0,data.nstates):
	ax2.plot(data.time[data.starti:data.endi], data.state_proj[i][data.starti:data.endi], '-', alpha=0.3, color=rgb[i])
  for i in range(0,datas[0].nstates):
    ax2.plot(data.time[data.starti:last_endi], avgstate_proj[i][data.starti:last_endi], 'o', color=rgb[i], label=r'Avg $S_{i}$'.format(i=i))
      
  ax2.legend()

  plt.show() 
  plt.savefig(outname, dpi=255)


import os, shutil, h5py
from PIL import Image


# VMD, ffmpeg
def render_trajectory(data):
  """
  SMALL_SIZE = 8
  MEDIUM_SIZE = 10
  BIGGER_SIZE = 12
  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  """
  d = "./"
  start = time.time()
  #d = self.d
  # Opening a h5py file while the simulation is running will crash it, so create a copy!
  p = subprocess.Popen('cp "'+d+'data.hdf5" "'+d+'data_read.hdf5"', shell=True)
  p.wait()
  time.sleep(1)
  h = h5py.File(data.d+"data_read.hdf5", 'r')
  print("data.maxsteps: "+str(data.maxsteps))
  nstep = min(len(h['time'])-4, data.maxsteps)
  print("nstep: "+str(nstep))
  #nstep = 200
  # Clean directories
  if os.path.exists("plot_anim/"):
    shutil.rmtree("plot_anim/")
    time.sleep(1)
    os.makedirs("plot_anim/")
  else:
    os.makedirs("plot_anim/")
  if os.path.exists("xyzs/"):
    shutil.rmtree("xyzs/")
    time.sleep(1)
    os.makedirs("xyzs/")
  else:
    os.makedirs("xyzs/")
  if os.path.exists("bmp/"):
    shutil.rmtree("bmp/")
    time.sleep(1)
    os.makedirs("bmp/")
  else:
    os.makedirs("bmp/")

  # copy xyz files to directory
  print("Preparing xyz files...")
  j = 0
  for i in range(2,nstep,5):
    shutil.copy(h["tdci_dir"][i]+"/temp.xyz", "xyzs/"+"{:04d}".format(j)+".xyz")
    j+=1

  # write VMD script
  # Keeping the script in here so we can modify camera parameters from python if we want
  vmdtxt = "# Script from Arshad Mehmood\n"+\
	    "color Display Background white\n"+\
	    "display resize 960 960\n"+\
	    "set isoval 0.25\n"+\
	    "axes location Off\n"+\
	    "for {set i 0} {$i<="+str(nstep)+"} {incr i} {\n"+\
	    "set name [format %04d $i]\n"+\
	    "puts \"Processing $name.xyz...\"\n"+\
	    "mol default style CPK\n"+\
	    "mol new ./xyzs/$name.xyz\n"+\
	    "scale to 0.45\n"+\
	    "rotate x by 50.0000\n"+\
	    "rotate z by 50.0000\n"+\
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
  print("Rendering xyz to bmp...")
  vmdp = subprocess.Popen("vmd -dispdev text -eofexit < vmd.tcl > output.log", shell=True)
  vmd_endcode = vmdp.wait()
  print("vmd endcode: "+str(vmd_endcode))

  print("Preparing plots...")
  j = 0
  for i in range(2,nstep,5):
    plt.clf()
    my_dpi = 200
    fig = plt.figure(figsize=(1000/my_dpi, 1600/my_dpi), dpi=my_dpi)
    #ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.title.set_text("State Populations")
    ax2.set_xlabel("Time (fs)")
    ax2.set_xlim(left=data.time[data.starti], right=data.time[data.endi])
    for k in range(0,data.nstates):
      ax2.plot(data.time[data.starti:i], data.state_proj[k][data.starti:i], '-', label=r'$S_{p}$'.format(p=k))

    img = np.asarray(Image.open(d+"bmp/{:04d}.bmp".format(j)))
    ax1 = fig.add_subplot(211)
    ax1.axis('off')
    ax1.imshow(img)
    
    ax2.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(d+"plot_anim/{:04d}.png".format(j), dpi=my_dpi)

    j+=1


  # render bmps into an mp4
  print("Rendering mp4...")
  ffmpegp = subprocess.Popen("ffmpeg -y -r 60 -i plot_anim/%04d.png -c:v libx264 -preset slow -crf 24 "+
                             " -force_key_frames source -x264-params keyint=4:scenecut=0 -pix_fmt yuv420p "+data.filelabel+".mp4", shell=True)
  ffmpeg_endcode = ffmpegp.wait()
  print("ffmpeg endcode: "+str(ffmpeg_endcode))
  # clean up files
  os.remove("vmd.tcl")
  shutil.rmtree("bmp/")
  shutil.rmtree("xyzs/")
  return 0


"""
outdir = "plots_300K2/"

if os.path.isdir(outdir):
  shutil.rmtree(outdir)

os.mkdir(outdir)





for data in datas_process:
  print("Starting "+data.filelabel)
  plot_traj(data, outdir+data.filelabel+"_traj.png")
  plot_ci_rmsd(data, outdir+data.filelabel+"_cirmsd.png")

print("Starting avg_pop.png")
plot_average_statepop(datas_process, outdir+"avg_pop.png")
"""

render_trajectory(datas_process[0])








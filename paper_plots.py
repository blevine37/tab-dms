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


datas_process = [
#plottables( "/home/adurden/conda/ethylene/fomo_benfix2_read/", "benfix2", DoStateProjections=True, DoSDiagnostic=True ),
#plottables( "/home/adurden/conda/ethylene/engtest2/", "engtest", DoStateProjections=False, DoSDiagnostic=False ),
#plottables( "/home/adurden/conda/ethylene/engtest4/", "fix", DoStateProjections=False, DoSDiagnostic=False ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/1/", "1", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/2/", "2", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/3/", "3", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/4/", "4", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/5/", "5", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/6/", "6", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/7/", "7", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/8/", "8", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/9/", "9", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
plottables( "/home/adurden/jobs/seawulf_jobs/benzene/10/", "10", DoStateProjections=True, DoSDiagnostic=True, maxsteps=10000 ),
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
  fig.set_figheight(12)
  fig.set_figwidth(16)
  ax1 = fig.add_subplot(511)
  ax4 = fig.add_subplot(512)
  ax7 = fig.add_subplot(513)
  ax8 = fig.add_subplot(514)
  ax9 = fig.add_subplot(515)
  ax1.title.set_text('Total Energy')
  ax1.plot(data.time[data.starti:data.endi], data.reltot[data.starti:data.endi], 'o', markersize=1, label=data.label)
  ax1.legend()
  ax1.set_ylabel("eV")


  ax4.title.set_text('RMS gradient')
  ax4.plot(data.time[data.starti:data.endi], data.rmsgrad[data.starti:data.endi], '-', label=data.label)
	       

  ax7.title.set_text("FOMO Orbital Energies")
  if data.DoFOMO:
    for i in range(data.clsd, data.clsd+data.acti+1):
      ax7.plot(data.time[data.starti:data.endi], data.fomo_eng[i][data.starti:data.endi], '-', label=str(i))
      pass
  ax7.legend()

  ax9.title.set_text(r"$S^2$ Closed-Active Off-Diagonals")
  if data.DoSDiagnostic:
    ax9.plot(data.time[data.starti:data.endi], data.S_sq_oos[data.starti:data.endi], '-', label=data.label)
  ax9.legend()     

  ax8.title.set_text("State Projections")
  if data.DoStateProjections:
    for i in range(0,data.nstates):
      ax8.plot(data.time[data.starti:data.endi], data.state_proj[i][data.starti:data.endi], '-', label='S'+str(i))
  ax8.legend()

  matplotlib.use("Agg")
  plt.tight_layout()
  plt.savefig(outname, dpi=255)




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


import os, shutil


if os.path.isdir("plots"):
  shutil.rmtree("plots")

os.mkdir("plots")

for data in datas_process:
  print("Starting "+data.filelabel)
  plot_traj(data, "plots/"+data.filelabel+"_traj.png")
  plot_ci_rmsd(data, "plots/"+data.filelabel+"_cirmsd.png")

print("Starting avg_pop.png")
plot_average_statepop(datas_process, "plots/avg_pop.png")










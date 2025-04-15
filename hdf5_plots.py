import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# === CONFIGURATION ===
filename = "data.hdf5"

# === LOAD DATA ===
with h5py.File(filename, "r") as f:
    print("Available keys:", list(f.keys()))

    time = f["time"][:]    
    ke_data = f["ke"][:]
    pe_data = f["pe"][:]
    state_enes = f["state_enes"][:] # 2D array: shape (steps, states)
    norm = f["norm"][:]           # 1D array: shape (steps,)
    pop = f["pop"][:]             # 2D array: shape (steps, states)

# === TOTAL ENERGY ===
etot = ke_data + pe_data

# === STATS ===
# Summary
n_steps = len(time)
dt_fs = (time[1] - time[0])/1000 if n_steps > 1 else 0.0
total_time_fs = (n_steps - 1) * dt_fs 

print("=== Simulation Info ===")
print(f"  Number of steps:     {n_steps}")
print(f"  Nuclear time step:   {dt_fs:.8f} fs")
print(f"  Total simulation:    {total_time_fs:.2f} fs")

# Energy drift
print("Total energy stats:")
print(f"  Mean: {np.mean(etot):.6f}")
print(f"  Std:  {np.std(etot):.6f}")

hartree_to_ev = 27.2114
drift = etot[-1] - etot[0]
drift_ev = drift * hartree_to_ev

print("Energy drift:")
print(f"  Start: {etot[0]:.6f} Ha,  End:   {etot[-1]:.6f} Ha")
print(f"  Drift: {drift:.6e} Ha  ({drift_ev:.6f} eV)")


# === PLOTS ===
# KE, PE, SUM as subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(pe_data, color='blue', label='Potential Energy')
axes[0].set_ylabel("PE")
axes[0].legend(loc='upper right')

axes[1].plot(ke_data, color='red', label='Kinetic Energy')
axes[1].set_ylabel("KE")
axes[1].legend(loc='upper right')

axes[2].plot(etot, color='black', label='Total Energy')
axes[2].set_ylabel("Total Energy")
axes[2].set_xlabel("Step")
axes[2].legend(loc='upper right')
axes[2].get_yaxis().get_major_formatter().set_useOffset(False)

for ax in axes:
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

plt.tight_layout()
plt.show()

# PES plot
plt.figure(figsize=(8, 4))
print(state_enes)
for i in range(state_enes.shape[1]):
    plt.plot(state_enes[:, i], label=f'State {i}')

plt.xlabel("Step")
plt.ylabel("Energy [Ha]")
plt.title("State Energies Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Norm plot
plt.figure(figsize=(8, 4))
plt.plot(norm, label='Norm', color='green')
plt.xlabel("Step")
plt.ylabel("Norm")
plt.legend()
plt.title("Norm Over Time")
plt.ticklabel_format(style='plain', useOffset=False, axis='both')
plt.tight_layout()
plt.show()

# Population plot
plt.figure(figsize=(8, 4))
plt.style.library['tableau-colorblind10']
for i in range(pop.shape[1]):
    plt.plot(pop[:, i], label=f'State {i}')
plt.xlabel("Step")
plt.ylabel("Population")
plt.legend()
plt.title("Population Dynamics")
plt.ticklabel_format(style='plain', useOffset=False, axis='both')
plt.tight_layout()
plt.show()


#!/usr/bin/python3
#Interface that takes Terachem input, converts it to SHARC input
#Runs SHARC to get LVC energies/forces/...
#Converts the output to TC outputs
#passes that to TAB-DMS

#Made for testing, certainly not the way how to properly implement this

import os
import random
import subprocess
import numpy as np
import struct
import scipy.linalg

print("TAB LVC-SHARC interface")


###formulate QM.in
# Read coordinates from temp.xyz
with open('temp.xyz', 'r') as f:
    lines = f.readlines()

num_atoms = int(lines[0].strip())
coords = lines[2:]  # skip blank line and atom count

# Get number of states
with open("tc.in", "r") as file:
    for line in file:
        if "cassinglets" in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                number_of_cassinglets = int(parts[1])
            break
nstates = number_of_cassinglets

print("Number of cassinglets:", nstates)

# Format atom lines with three trailing zeros
formatted_atoms = []
for line in coords:
    parts = line.split()
    atom = parts[0]
    x, y, z = map(float, parts[1:4])
    formatted_atoms.append(f"{atom:<2} {x:>12.7f} {y:>12.7f} {z:>12.7f} {0.0:>14.7f} {0.0:>11.7f} {0.0:>11.7f}")

# Generate random integer
rand_int = random.randint(100000000, 999999999)

# Get current working directory
savedir = os.getcwd()

# Build content of QM.in
qm_content = [
    f"{num_atoms}",
    f"{rand_int}",
    *formatted_atoms,
    "init",
    "unit angstrom",
    f"states  {nstates}",
    f"savedir {savedir}",
    "SOC",
    "NACDR",
    "GRAD all"
]

# Write to QM.in
with open('QM.in', 'w') as f:
    f.write('\n'.join(qm_content))

print("QM.in file successfully written.")

### link the LVC model
target = "../../LVC.template"
link_name = "LVC.template"

# Create the symbolic link
try:
    os.symlink(target, link_name)
    print(f"Symbolic link created: {link_name} -> {target}")
except FileExistsError:
    print(f"Link {link_name} already exists.")
except OSError as e:
    print(f"Error creating symlink: {e}")

### link the LVC model
target = "../../LVC.resources"
link_name = "LVC.resources"

# Create the symbolic link
try:
    os.symlink(target, link_name)
    print(f"Symbolic link created: {link_name} -> {target}")
except FileExistsError:
    print(f"Link {link_name} already exists.")
except OSError as e:
    print(f"Error creating symlink: {e}")

###run the calc
sharc_script = os.path.expandvars("$SHARC/SHARC_LVC.py")
# Open log files
with open("QM.log", "a") as log_file, open("QM.err", "a") as err_file:
    subprocess.run(
        ["python3", sharc_script, "QM.in"],
        stdout=log_file,
        stderr=err_file
    )


###read in the results
#energies
# Each line should contain 2*n_states entries: real1 imag1 real2 imag2 ...

#gradients
def parse_qm_out_fixed(filename, n_states=25, n_atoms=9, dims=3):
    hamiltonian = np.zeros((n_states, n_states), dtype=complex)
    gradients = np.zeros((n_states, n_atoms, dims))
    nac_couplings = np.zeros((n_states, n_states, n_atoms, dims))

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract full Hamiltonian matrix
    matrix_started = False
    row_count = 0
    for i, line in enumerate(lines):
        if '1 Hamiltonian Matrix' in line:
            matrix_started = True
            continue
        if matrix_started:
            parts = line.strip().split()
            if len(parts) >= 2 * n_states:
                # Extract real and imaginary parts alternately: real1 imag1 real2 imag2 ...
                for j in range(n_states):
                    real_part = float(parts[2*j])
                    imag_part = float(parts[2*j + 1])
                    hamiltonian[row_count, j] = complex(real_part, imag_part)
                row_count += 1
                if row_count >= n_states:
                    break
    
    # Extract diagonal energies for compatibility
    energies = np.real(np.diag(hamiltonian))

    # Extract diabatic gradient vectors
    found_states = 0
    for i, line in enumerate(lines):
        if '! 3 Gradient Vectors' in line:
            idx = i + 1
            while found_states < n_states and idx < len(lines):
                line = lines[idx].strip()
                # Skip empty lines and lines starting with '!'
                if not line or line.startswith('!'):
                    idx += 1
                    continue
                # Try to read a block of 9 lines (9 atoms)
                try:
                    for a in range(n_atoms):
                        parts = lines[idx + a].split()
                        gradients[found_states, a, :] = list(map(float, parts))
                    found_states += 1
                    idx += n_atoms
                except (ValueError, IndexError):
                    idx += 1  # Skip problematic block
            break

    # Extract non-adiabatic couplings (NAC)
    print('>>>Extracting NAC vectors')
    nac_pairs_found = 0
    for i, line in enumerate(lines):
        if '! 5 Non-adiabatic couplings' in line:
            idx = i + 1
            while nac_pairs_found < n_states * n_states and idx < len(lines):
                line = lines[idx].strip()
                # Skip empty lines
                if not line:
                    idx += 1
                    continue
                # Look for state pair header: "9 3 ! m1 1 s1 X ms1 0   m2 1 s2 Y ms2 0"
                if str(n_atoms)+' 3 !' in line and 's1' in line and 's2' in line:
                    try:
                        # Extract state indices from header
                        parts = line.split()
                        s1_idx = int(parts[6]) - 1  # Convert to 0-based indexing
                        s2_idx = int(parts[12]) - 1
                        
                        # Read n_atoms lines (9 atoms) for this state pair
                        for a in range(n_atoms):
                            coord_line = lines[idx + 1 + a].split()
                            nac_couplings[s1_idx, s2_idx, a, :] = list(map(float, coord_line))
                        #print('NAC',s1_idx,s2_idx,nac_couplings[s1_idx,s2_idx])
                        
                        nac_pairs_found += 1
                        idx += n_atoms + 1  # Skip the 9 atom lines
                    except (ValueError, IndexError):
                        print('Error: NAC reading in QM.out')
                        sys.exit()
                        idx += 1
                else:
                    idx += 1
            break

    return energies, gradients, hamiltonian, nac_couplings


energies, gradients, hamiltonian, nac_couplings = parse_qm_out_fixed('QM.out', n_states=nstates)

print('E:',energies)
print('Hamiltonian matrix shape:', hamiltonian.shape)
print('NAC couplings shape:', nac_couplings.shape)
#print('Max NAC coupling:', np.max(np.abs(nac_couplings)))
#print('Hamiltonian matrix diagonal (real):', np.real(np.diag(hamiltonian)))
#print('Hamiltonian matrix first off-diagonal element:', hamiltonian[0,1])
#print('Grad[0]:',gradients[0])


### export bin files
states_energy_bin_path = './States_E.bin'
with open(states_energy_bin_path, 'wb') as f:
    f.write(struct.pack('d' * len(energies), *energies))

states_cn_bin_path = './States_Cn.bin'
identity_matrix = np.eye(nstates, dtype=np.float64)
with open(states_cn_bin_path, 'wb') as f:
    f.write(struct.pack('d' * identity_matrix.size, *identity_matrix.flatten()))

misc_bin_path = './misc.bin'
with open(misc_bin_path, 'wb') as f:
    f.write(struct.pack('iii', nstates, nstates, nstates)) #ndets, nmo, nbf

# Export Hamiltonian matrix and NACs to binary file (for TAB adiabatic collapse)
hamiltonian_bin_path = './Hamiltonian.bin'
hamiltonian_real = np.real(hamiltonian).flatten()
hamiltonian_imag = np.imag(hamiltonian).flatten()
with open(hamiltonian_bin_path, 'wb') as f:
    f.write(struct.pack('d' * hamiltonian_real.size, *hamiltonian_real))
    f.write(struct.pack('d' * hamiltonian_imag.size, *hamiltonian_imag))
print(f'Hamiltonian exported to {hamiltonian_bin_path}')

# Export Hamiltonian derivatives dH/dR (diagonal + off-diagonal forces)
# Construct full dH/dR for each atomic coordinate
dH_dR = np.zeros((nstates, nstates, num_atoms, 3))

# Diagonal: diabatic gradients
for i in range(nstates):
    dH_dR[i, i, :, :] = gradients[i, :, :]

# Off-diagonal: NAC * energy_gap
diagonal_energies = np.real(np.diag(hamiltonian))
for i in range(nstates):
    for j in range(i+1, nstates):
        energy_gap = diagonal_energies[j] - diagonal_energies[i]
        if abs(energy_gap) > 1e-10:
            dH_dR[i, j, :, :] = nac_couplings[i, j, :, :] * energy_gap   #From Hellman Feynman <i|dH/dR|j> =...= (E_j - E_i) <i|d/dR|j>
            dH_dR[j, i, :, :] = -dH_dR[i, j, :, :]  # Antisymmetric NACs

# Export flattened dH/dR
dH_dR_bin_path = './dH_dR.bin'
dH_dR_flat = dH_dR.flatten()
with open(dH_dR_bin_path, 'wb') as f:
    f.write(struct.pack('d' * dH_dR_flat.size, *dH_dR_flat))
print(f'Hamiltonian derivatives dH/dR exported to {dH_dR_bin_path}')

# Load initial ReCn and ImCn coefficients from previous step
def load_coefficients(recn_path='./recn_init.bin', imcn_path='./imcn_init.bin', n_states=25):
    """Load ReCn and ImCn coefficients from binary files"""
    try:
        # Load ReCn coefficients
        with open(recn_path, 'rb') as f:
            recn_data = f.read()
            recn = np.array(struct.unpack('d' * (len(recn_data) // 8), recn_data))
        
        # Load ImCn coefficients 
        try:
            with open(imcn_path, 'rb') as f:
                imcn_data = f.read()
                imcn = np.array(struct.unpack('d' * (len(imcn_data) // 8), imcn_data))
        except FileNotFoundError:
            print(f"Warning: {imcn_path} not found, initializing ImCn to zeros")
            imcn = np.zeros(n_states, dtype=np.float64)
            
        # Ensure proper size
        if len(recn) != n_states:
            print(f"Warning: ReCn size {len(recn)} doesn't match n_states {n_states}")
            if len(recn) < n_states:
                recn = np.pad(recn, (0, n_states - len(recn)))
            else:
                recn = recn[:n_states]
                
        if len(imcn) != n_states:
            print(f"Warning: ImCn size {len(imcn)} doesn't match n_states {n_states}")
            if len(imcn) < n_states:
                imcn = np.pad(imcn, (0, n_states - len(imcn)))
            else:
                imcn = imcn[:n_states]
                
    except FileNotFoundError:
        print(f"Warning: {recn_path} not found, initializing to ground state")
        recn = np.zeros(n_states, dtype=np.float64)
        recn[0] = 1.0  # Ground state
        imcn = np.zeros(n_states, dtype=np.float64)
    
    return recn, imcn

# Check if current folder name is an integer
current_folder = os.path.basename(os.getcwd())
try:
    int(current_folder)
    do_propagation = True
    print(f'Folder "{current_folder}" is integer - will do quantum propagation')
except ValueError:
    do_propagation = False
    print(f'Folder "{current_folder}" is not integer - will copy previous coefficients')

# Load coefficients (always needed for weights calculation)
recn_current, imcn_current = load_coefficients(n_states=nstates)
print('Loaded ReCn:', recn_current)
print('Loaded ImCn:', imcn_current)
print('Initial population:', recn_current**2 + imcn_current**2)

if do_propagation:
    # Schrödinger equation propagation 
    # i d/dt |C⟩ = H |C⟩  =>  |C(t+dt)⟩ = exp(-i H dt) |C(t)⟩

    def schrodinger_propagate(recn, imcn, hamiltonian, dt, hbar=1.0):
        """
        Propagate wavefunction using time evolution operator
        H and C are already in the same basis
        """
        # Combine real and imaginary parts into complex coefficients
        #print('Recn:', recn)
        #print('Imcn:', imcn)
 
        c_current = recn + 1j * imcn
        
        # Apply time evolution operator: |C(t+dt)⟩ = exp(-i H dt / ħ) |C(t)⟩
        # Use matrix exponential directly since H and C are in same basis
        time_evolution_operator = scipy.linalg.expm(-1j * hamiltonian * dt / hbar)
        c_evolved = np.dot(time_evolution_operator, c_current)
        
        # Extract real and imaginary parts
        recn_new = np.real(c_evolved)
        imcn_new = np.imag(c_evolved)
        #print('Recn_new:', recn_new)
        #print('Imcn_new:', imcn_new)
        
        return recn_new, imcn_new

    # Get time step from ../../log file
    def read_timestep_from_log(log_path='../../log', default_dt=4.96096480022):
        try:
            with open(log_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    if 'Propagation time step in au:' in line:
                        #print('Propagation time step in au (read from log):',float(line.split(':')[-1].strip()))
                        return float(line.split(':')[-1].strip())
        except:
            pass
        #print('Propagation time step in au (using default):',default_dt)
        return default_dt

    dt_au = read_timestep_from_log()

    print(f'Time step: {dt_au:.6f} au')
    #print('Initial population:', recn_current**2 + imcn_current**2)
    print('Initial coefficients norm:', np.sqrt(np.sum(recn_current**2 + imcn_current**2)))
    
    # Calculate initial energy
    c_initial = recn_current + 1j * imcn_current
    initial_energy = np.real(np.conj(c_initial).T @ hamiltonian @ c_initial)
    print(f'Initial energy: {initial_energy:.12f} au')

    # Propagate using SSO method
    recn_final, imcn_final = schrodinger_propagate(
        recn_current, imcn_current, hamiltonian, dt_au
    )

    #print('Final population:', recn_final**2 + imcn_final**2)
    print('Final coefficients norm:', np.sqrt(np.sum(recn_final**2 + imcn_final**2)))
    
    # Calculate final energy and check conservation
    c_final_check = recn_final + 1j * imcn_final
    final_energy = np.real(np.conj(c_final_check).T @ hamiltonian @ c_final_check)
    energy_drift = final_energy - initial_energy
    print(f'Final energy: {final_energy:.12f} au')
    print(f'Energy drift: {energy_drift:.2e} au')
    print(f'Relative energy error: {abs(energy_drift/initial_energy):.2e}')
    

else:
    # Just copy previous coefficients (original behavior)
    print('Copying previous coefficients without propagation')
    recn_final, imcn_final = recn_current, imcn_current
    
    # Still calculate energy for reference
    c_final_check = recn_final + 1j * imcn_final
    energy = np.real(np.conj(c_final_check).T @ hamiltonian @ c_final_check)
    print(f'Current energy: {energy:.12f} au')

# Write coefficients to output files
imcn_end_bin_path = './ImCn_end.bin'
recn_end_bin_path = './ReCn_end.bin'

if do_propagation:
    # Write propagated coefficients
    with open(imcn_end_bin_path, 'wb') as f:
        f.write(struct.pack('d' * imcn_final.size, *imcn_final))
    with open(recn_end_bin_path, 'wb') as f:
        f.write(struct.pack('d' * recn_final.size, *recn_final))
else:
    # Copy previous files (original behavior)
    with open('./imcn_init.bin', 'rb') as src, open(imcn_end_bin_path, 'wb') as dst:
        dst.write(src.read())
    
    with open('./recn_init.bin', 'rb') as src, open(recn_end_bin_path, 'wb') as dst:
        dst.write(src.read())

#gradients
# Write gradient vectors into {nstates} separate files: gradstate.0.bin to gradstate.24.bin
gradient_bin_paths = []

for state_index in range(nstates):
    filename = f'./gradstate{state_index}.bin'
    gradient_data = gradients[state_index].flatten()  # shape (9, 3) → (27,)
    with open(filename, 'wb') as f:
        f.write(struct.pack('d' * gradient_data.size, *gradient_data))
    gradient_bin_paths.append(filename)

#Ehrenfest force - NAC = 0, weighted average
# Use the final ReCn and ImCn coefficients to calculate population weights
weights = recn_final**2 + imcn_final**2  # Population weights from |Cn|^2
print('Population weights:', weights)
print('Total population:', np.sum(weights))

# Compute diabatic Ehrenfest force: F = -⟨Ψ|∇H|Ψ⟩ = -Σᵢⱼ Cᵢ* Cⱼ ∂Hᵢⱼ/∂R
# QM.out gives: gradients = ∂E/∂R, NAC = (∂Hᵢⱼ/∂R)/(Ej-Ei) for off-diagonals
# Need to rescale NAC by energy gaps to get true diabatic derivatives

c_final = recn_final + 1j * imcn_final
ehrenfest_force = np.zeros((num_atoms, 3))

# Get diagonal energies for energy gap calculation
diagonal_energies = np.real(np.diag(hamiltonian))

# Ehrenfest force: F = -Σᵢⱼ Cᵢ* Cⱼ ∂Hᵢⱼ/∂R
for i in range(nstates):
    for j in range(nstates):
        rho_ij = np.conj(c_final[i]) * c_final[j]  # Cᵢ* Cⱼ
        
        if i == j:
            # Diagonal: ∂Hᵢᵢ/∂R from gradients (already ∂E/∂R)
            hamiltonian_derivative = gradients[i]
        else:
            # Off-diagonal: Convert NAC back to diabatic derivative
            # NAC_output = (∂H_ij/∂R) / (E_j - E_i)
            # So: ∂H_ij/∂R = NAC_output × (E_j - E_i)
            energy_gap = diagonal_energies[j] - diagonal_energies[i]
            if abs(energy_gap) > 1e-10:  # Avoid division by zero
                hamiltonian_derivative = nac_couplings[i, j] * energy_gap
            else:
                hamiltonian_derivative = np.zeros((num_atoms, 3))  # Zero for degenerate states
            #print(hamiltonian_derivative)
            #test: ignore off-diag
            #print('IGNORING OFF-DIAG')
            #hamiltonian_derivative = np.zeros((num_atoms, 3))
        
        # Sum all contributions: Σᵢⱼ Cᵢ* Cⱼ ∂Hᵢⱼ/∂R
        ehrenfest_force += np.real(rho_ij) * hamiltonian_derivative

# This gives us ⟨∇H⟩ = Σᵢⱼ Cᵢ* Cⱼ ∂Hᵢⱼ/∂R (gradients, not forces)
ehrenfest_gradient = ehrenfest_force  # Rename for clarity - this is actually a gradient

print('Ehrenfest gradient shape:', ehrenfest_gradient.shape)
print('Max Ehrenfest gradient component:', np.max(np.abs(ehrenfest_gradient)))

# Write gradients to bin files (as expected by the dynamics code)
weighted_gradient = ehrenfest_gradient

# Compute weighted average energy using full Hamiltonian: ⟨Ψ|H|Ψ⟩
weighted_avg_energy = np.real(np.conj(c_final).T @ hamiltonian @ c_final)

# Flatten and prepare for writing
flattened_weighted_gradient = weighted_gradient.flatten()  # shape: (27,)

# Write the same result into the three output files
tdci_paths = []
for label in ['init', 'half', 'end']:
    path = f'./tdci_grad_{label}.bin'
    with open(path, 'wb') as f:
        f.write(struct.pack('d' * flattened_weighted_gradient.size, *flattened_weighted_gradient))
    tdci_paths.append(path)

#Fake tc.out
tc_out_path = './tc.out'

with open(tc_out_path, 'w') as f:
    f.write(f"Initial energy: {weighted_avg_energy:.12f}\n")
    f.write(f"Final TDCI Energy: {weighted_avg_energy:.12f}\n")
    f.write("DONE\n")

#norm
calculated_norm = np.sqrt(np.sum(recn_final**2 + imcn_final**2))
norm_out_path = './norm'
with open(norm_out_path, 'w') as f:
  f.write(f"1.0, {calculated_norm:.12f}")

pop_out_path = './Pop'
pop_header = ['Time (as)'] + [f'State {i}' for i in range(nstates)]
line = '1.0, ' + ', '.join(f'{w:.12f}' for w in weights)
with open(pop_out_path, 'w') as f:
   f.write(','.join(pop_header)+'\n')
   f.write(line+'\n')

#empty bins
empty_files = ['./NewCoors.bin', './NewC.bin']
for path in empty_files:
    with open(path, 'wb') as f:
        pass

print("Done")

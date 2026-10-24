import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


# =====
# import numpy as np

# data = np.loadtxt('fno_sindy_ensemble_summary.txt', skiprows=1)
# t         = data[:, 0]
# true_real = data[:, 1]
# true_imag = data[:, 2]
# std_real  = data[:, 4]
# std_imag  = data[:, 6]

# sigma_ensemble = np.sqrt(std_real**2 + std_imag**2)

# # Peak A: Look exclusively on the NEGATIVE Real side (Left hump in your 3D plot)
# left_mask = (true_real < 0)
# idx_peak_left = np.where(left_mask)[0][np.argmax(sigma_ensemble[left_mask])]

# # Peak B: Look exclusively on the POSITIVE Real side (Right hump in your 3D plot)
# right_mask = (true_real > 0)
# idx_peak_right = np.where(right_mask)[0][np.argmax(sigma_ensemble[right_mask])]

# print("--- CORRECTED GEOMETRIC PEAK LOCATIONS ---")
# print(f"Left-Side Peak occurs at t = {t[idx_peak_left]:.5f} s")
# print(f"   Coordinates: Re(z) = {true_real[idx_peak_left]:.2f}, Im(z) = {true_imag[idx_peak_left]:.2f}")
# print(f"   Uncertainty (Sigma): {sigma_ensemble[idx_peak_left]:.2f}")
# print("-" * 50)
# print(f"Right-Side Peak occurs at t = {t[idx_peak_right]:.5f} s")
# print(f"   Coordinates: Re(z) = {true_real[idx_peak_right]:.2f}, Im(z) = {true_imag[idx_peak_right]:.2f}")
# print(f"   Uncertainty (Sigma): {sigma_ensemble[idx_peak_right]:.2f}")
# ======

# --- 1. ENVIRONMENT SETUP & TARGET TIMESTAMPS ---
mat_path = os.path.join('jet_data', 'sp7.mat')
Dj = 0.0169  # Jet diameter (m)

t_peak1 = 0.03480  # Right-Side Peak
t_peak2 = 0.05403  # Left-Side Peak

print(f"Loading HDF5 v7.3 jet database from: {mat_path}...")

# Use h5py instead of scipy.io
with h5py.File(mat_path, 'r') as f:
    # --- 2. PREPROCESSING & TRANSPOSE HANDLING ---
    # h5py reads dimensions in reverse order: [nt, nx, nr] or [nt, nx, nr, 1]
    # Let's read the data directly
    Pa_n0 = f['Pa_n0'][:]
    t_inp = f['t_inp'][:].flatten()
    x_ij  = f['x_ij'][:]
    y_ij  = f['y_ij'][:]

# Strip any trailing singleton dimensions if they exist
p_raw = np.squeeze(Pa_n0)  

# MATLAB shape was [1, nr, nx, nt] -> [51, 100, 5348] after squeeze
# h5py reads it flipped as [5348, 100, 51] -> [nt, nx, nr]
# We want to reorder it pythonically to [nt, nr, nx] -> [5348, 51, 100]
p = np.transpose(p_raw, (0, 2, 1))

# Handle coordinate transposes from HDF5 reader
x_norm = x_ij / Dj  # Grid coordinates
r_norm = y_ij / Dj

# Calculate fluctuations (subtract time-mean)
p_mean = np.mean(p, axis=0)
p_fluct = p - p_mean

# Establish uniform timeline
dt = np.mean(np.diff(t_inp))
t_uniform = t_inp[0] + np.arange(p.shape[0]) * dt

print(f"Fluctuation tensor shape: {p_fluct.shape} -> [nt, nr, nx]")

# --- 3. TEMPORAL INDEX MATCHING ---
idx_peak1 = np.argmin(np.abs(t_uniform - t_peak1))
idx_peak2 = np.argmin(np.abs(t_uniform - t_peak2))

print(f"\nMatched Peak 1 (t={t_peak1}s) to frame index: {idx_peak1}")
print(f"Matched Peak 2 (t={t_peak2}s) to frame index: {idx_peak2}")

# --- 4. FLOW FIELD SNAPSHOT VISUALIZATION ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Peak 1 Snapshot (Right-Side Peak) - Added .T to match grid dimensions
ax1 = axes[0]
snapshot_1 = p_fluct[idx_peak1, :, :].T  
im1 = ax1.pcolormesh(x_norm, r_norm, snapshot_1, cmap='RdBu_r', shading='gouraud', vmin=-150, vmax=150)
ax1.set_ylabel(r'$r / D_j$', fontsize=11)
ax1.set_title(f'Flow Snapshot 1 (Right Peak) | $t$ = {t_uniform[idx_peak1]:.5f} s', fontsize=12, fontweight='bold')
ax1.axis('tight')
fig.colorbar(im1, ax=ax1, label='Pressure Fluctuation (Pa)')

# Peak 2 Snapshot (Left-Side Peak) - Added .T to match grid dimensions
ax2 = axes[1]
snapshot_2 = p_fluct[idx_peak2, :, :].T  
im2 = ax2.pcolormesh(x_norm, r_norm, snapshot_2, cmap='RdBu_r', shading='gouraud', vmin=-150, vmax=150)
ax2.set_xlabel(r'$x / D_j$', fontsize=11)
ax2.set_ylabel(r'$r / D_j$', fontsize=11)
ax2.set_title(f'Flow Snapshot 2 (Left Peak) | $t$ = {t_uniform[idx_peak2]:.5f} s', fontsize=12, fontweight='bold')
ax2.axis('tight')
fig.colorbar(im2, ax=ax2, label='Pressure Fluctuation (Pa)')

plt.tight_layout()
plt.savefig('jet_flow_snapshots_verification.png', dpi=300)
print("\n[SUCCESS] Saved HDF5-compatible plot as 'jet_flow_snapshots_verification.png'.")
plt.show()
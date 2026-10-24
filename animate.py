"""
animate_fno_reconstruction.py
─────────────────────────────────────────────────────────────────────────────
Three-panel premium dark-themed layout:
  TOP    — True SPOD reconstruction  : Re[φ₁ · a₁_true(t)]
  MIDDLE — Hybrid FNO-ODE prediction : Re[φ₁ · â₁(t)]
  BOTTOM — Dynamic tracking of modal coefficients over time

Usage:  python animate_fno_reconstruction.py [--fps 24] [--dpi 150] [--stride 2]
"""

import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects

parser = argparse.ArgumentParser()
parser.add_argument('--fps',     type=int,   default=24)
parser.add_argument('--dpi',     type=int,   default=150)
parser.add_argument('--stride',  type=int,   default=2)
parser.add_argument('--summary', default='fno_sindy_ensemble_summary.txt')
parser.add_argument('--mat',     default='/Users/icarus/Desktop/SPODcode/spod_matlab/spod_complete_continuous_database.mat')
parser.add_argument('--output',  default='fno_spod_reconstruction.mp4')
parser.add_argument('--st',      type=float, default=2.0)
args = parser.parse_args()

S  = args.stride
Dj = 0.0169

# ─── 1. TEMPORAL COEFFICIENTS ────────────────────────────────────────────────
print("Loading temporal coefficients …")
summary  = np.loadtxt(args.summary, skiprows=1)
t_full   = summary[::S, 0]

# Ground Truth Real & Imag
true_q   = summary[::S, 1] + 1j * summary[::S, 2]

# Extract Pure ODE Phase and FNO Mean Amplitude to form Hybrid prediction
ode_real = summary[::S, 5]
ode_imag = summary[::S, 6]
theta_ode = np.arctan2(ode_imag, ode_real)
r_fno = summary[::S, 7]
pred_q = r_fno * np.exp(1j * theta_ode)

N_frames = len(t_full)
print(f"  {N_frames} frames  (stride={S})")

# ─── 2. SPATIAL MODE ─────────────────────────────────────────────────────────
print(f"Loading spatial mode St≈{args.st} …")
with h5py.File(args.mat, 'r') as file:
    st_vector = np.squeeze(file['St_vector'][:])
    target_fi = int(np.argmin(np.abs(st_vector - args.st)))
    actual_st = float(st_vector[target_fi])
    print(f"  St = {actual_st:.4f}")

    x_grid = file['x_ij'][:].T           
    y_grid = file['y_ij'][:].T           
    raw    = file['P'][0, :, :, target_fi]
    phi    = raw['real'] + 1j * raw['imag']  

x_norm = x_grid / Dj
y_norm = y_grid / Dj

# ─── 3. RECONSTRUCT ──────────────────────────────────────────────────────────
print("Reconstructing spatial fields …")
p_true = np.real(phi[np.newaxis] * true_q[:, np.newaxis, np.newaxis])  
p_pred = np.real(phi[np.newaxis] * pred_q[:, np.newaxis, np.newaxis])  

vmax = float(np.quantile(np.abs(p_true), 0.995))
norm = Normalize(vmin=-vmax, vmax=vmax)

# Text glow helper
txt_glow = [path_effects.withStroke(linewidth=3, foreground='#0a0a0a')]

# ─── 4. FIGURE BUILDER ───────────────────────────────────────────────────────
def build_figure():
    fig = plt.figure(figsize=(13, 8.5))
    fig.patch.set_facecolor('#0a0a0a')
    
    # 3-Row Grid Spec (Top & Middle fields share space, Bottom is line tracking)
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 0.6], hspace=0.15)
    
    ax_true = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1], sharex=ax_true)
    ax_line = fig.add_subplot(gs[2])
    
    for ax in (ax_true, ax_pred, ax_line):
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#888888', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#222222')
            
    # Labels for snapshot windows
    ax_true.text(0.015, 0.88, 'TRUE SPOD RECONSTRUCTION', transform=ax_true.transAxes,
                 color='#00ffcc', fontsize=9, fontweight='bold', va='top', path_effects=txt_glow)
    ax_pred.text(0.015, 0.88, 'HYBRID FNO-ODE PREDICTION', transform=ax_pred.transAxes,
                 color='#ff007f', fontsize=9, fontweight='bold', va='top', path_effects=txt_glow)
    
    ax_true.set_ylabel(r'$y/D_j$', color='#aaaaaa', fontsize=10)
    ax_pred.set_ylabel(r'$y/D_j$', color='#aaaaaa', fontsize=10)
    plt.setp(ax_true.get_xticklabels(), visible=False)
    
    # Render fields
    im_true = ax_true.pcolormesh(x_norm, y_norm, p_true[0], cmap='RdBu_r', norm=norm, shading='gouraud', rasterized=True)
    im_pred = ax_pred.pcolormesh(x_norm, y_norm, p_pred[0], cmap='RdBu_r', norm=norm, shading='gouraud', rasterized=True)
    
    # Colorbar positioning alongside field axes
    cbar = fig.colorbar(im_true, ax=[ax_true, ax_pred], fraction=0.015, pad=0.02, aspect=30)
    cbar.set_label("p' [Pa]", color='#aaaaaa', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#333333', labelcolor='#888888', labelsize=8)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor('#222222')
        
    # Bottom Time-Series Tracker Configurations
    ax_line.plot(t_full * 1e3, np.real(true_q), color='#00ffcc', alpha=0.4, linewidth=1, label='True')
    ax_line.plot(t_full * 1e3, np.real(pred_q), color='#ff007f', alpha=0.8, linewidth=1.2, label='FNO-ODE')
    time_indicator = ax_line.axvline(t_full[0]*1e3, color='#ffffff', linestyle='--', alpha=0.7, linewidth=1)
    
    ax_line.set_xlim(t_full[0]*1e3, t_full[-1]*1e3)
    ax_line.set_xlabel('Time [ms]', color='#aaaaaa', fontsize=10)
    ax_line.set_ylabel(r'$Re(a_1)$', color='#aaaaaa', fontsize=9)
    ax_line.grid(True, color='#1c1c1c', linestyle=':', linewidth=0.5)
    
    leg = ax_line.legend(loc='upper right', frameon=True, facecolor='#0a0a0a', edgecolor='#222222', fontsize=8)
    for text in leg.get_texts():
        text.set_color('#aaaaaa')
        
    ax_pred.set_xlabel(r'$x/D_j$', color='#aaaaaa', fontsize=10)
    
    # Global Title
    suptitle = fig.suptitle("", color='white', fontsize=12, fontweight='semibold', y=0.96)
    
    return fig, im_true, im_pred, time_indicator, suptitle

# ─── 5. ANIMATION RUNNER ─────────────────────────────────────────────────────
print("Building master animation …")
fig, im_true, im_pred, time_indicator, suptitle = build_figure()

_pbar = tqdm(total=N_frames, desc="Rendering Video", unit="frame",
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

def _update(frame):
    im_true.set_array(p_true[frame].ravel())
    im_pred.set_array(p_pred[frame].ravel())
    
    current_t_ms = t_full[frame] * 1e3
    time_indicator.set_xdata([current_t_ms, current_t_ms])
    
    suptitle.set_text(
        f"Scientific ML Rollout  │  Timestep: {frame * S:04d}  │  "
        f"t = {current_t_ms:.2f} ms  │  St = {actual_st:.3f}"
    )
    _pbar.update(1)
    return im_true, im_pred, time_indicator, suptitle

ani = animation.FuncAnimation(fig, _update, frames=N_frames, interval=1000 // args.fps, blit=True)

writer = animation.FFMpegWriter(fps=args.fps, bitrate=8000, extra_args=['-crf', '16', '-pix_fmt', 'yuv420p'])
ani.save(args.output, writer=writer, dpi=args.dpi, savefig_kwargs={'facecolor': '#0a0a0a'})
_pbar.close()
plt.close(fig)
print(f"Done! → {os.path.abspath(args.output)}")

# ─── 6. STATIC DIAGNOSTIC FRAMES GENERATOR ───────────────────────────────────
print("Exporting publication-ready static snapshots …")
for fi, label in [(0, 'frame0'), (N_frames // 2, 'midpoint'), (N_frames - 1, 'final')]:
    fig_static, im_t, im_p, t_ind, title_s = build_figure()
    
    # Advance to the target snapshot values
    im_t.set_array(p_true[fi].ravel())
    im_p.set_array(p_pred[fi].ravel())
    current_t_ms = t_full[fi] * 1e3
    t_ind.set_xdata([current_t_ms, current_t_ms])
    
    title_s.set_text(
        f"Analysis Snapshot ({label.upper()})  │  Timestep: {fi * S:04d}  │  "
        f"t = {current_t_ms:.2f} ms  │  St = {actual_st:.3f}"
    )
    
    out_png = args.output.replace('.mp4', f'_{label}.png')
    fig_static.savefig(out_png, dpi=args.dpi, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close(fig_static)
    print(f"  → Saved {out_png}")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from find_spectrum import hann2, pwelch2

# =====================================================================
# LOAD DATA FROM ENSEMBLE EXPORTS
# =====================================================================
summary_data = np.loadtxt(
    'fno_sindy_ensemble_summary.txt', skiprows=1
)
dist_data = np.loadtxt(
    'fno_sindy_ensemble_distributions.txt', skiprows=1
)

t_model_slice = summary_data[:, 0]
true_real = summary_data[:, 1]
true_imag = summary_data[:, 2]

true_complex = true_real + 1j * true_imag
num_snapshots = len(t_model_slice)
num_realizations = 30

matrix_real = dist_data[:, 0].reshape(
    num_realizations, num_snapshots
)
matrix_imag = dist_data[:, 1].reshape(
    num_realizations, num_snapshots
)
matrix_complex = matrix_real + 1j * matrix_imag

dt = np.mean(np.diff(t_model_slice))
fs = 1.0 / dt

# =====================================================================
# SPECTRUM AND METRICS — PER REALIZATION
# =====================================================================
nFFT = 64
window = hann2(nFFT)

Pxx_true, F_true = pwelch2(
    true_complex - np.mean(true_complex),
    nFFT, fs, window
)
f_peak_true = F_true[np.argmax(Pxx_true)]

all_psd_realizations = []
all_phases_pred = []
all_envelopes = []

for i in range(num_realizations):
    z_i = matrix_complex[i, :]
    z_i_centered = z_i - np.mean(z_i)

    Pxx_i, F_pred = pwelch2(
        z_i_centered, nFFT, fs, window
    )

    all_psd_realizations.append(Pxx_i)
    all_phases_pred.append(np.unwrap(np.angle(z_i)))
    all_envelopes.append(np.abs(z_i))

matrix_psd = np.array(all_psd_realizations)
matrix_phases = np.array(all_phases_pred)
matrix_envelopes = np.array(all_envelopes)

true_phase_unwrapped = np.unwrap(np.angle(true_complex))

# Physical constants
Dj, Uj = 0.0169, 302.0

Strouhal_true = F_true * Dj / Uj
Strouhal_pred = F_pred * Dj / Uj

kx_normalized = 0.8602
kx_physical = kx_normalized / Dj

# =====================================================================
# ENSEMBLE STATISTICS
# =====================================================================

# Phase statistics at each time
phase_median = np.median(matrix_phases, axis=0)
phase_mean = np.mean(matrix_phases, axis=0)
phase_lower = np.percentile(matrix_phases, 2.5, axis=0)
phase_upper = np.percentile(matrix_phases, 97.5, axis=0)

# PSD statistics in dB space at each frequency
psd_db_matrix = 10 * np.log10(matrix_psd)
psd_median_db = np.median(psd_db_matrix, axis=0)
psd_lower_db = np.percentile(psd_db_matrix, 2.5, axis=0)
psd_upper_db = np.percentile(psd_db_matrix, 97.5, axis=0)

f_peak_pred = F_pred[np.argmax(psd_median_db)]

# Convection velocity evaluated per realization
individual_uc_tracks = [
    np.gradient(matrix_phases[i, :], dt) / kx_physical
    for i in range(num_realizations)
]

mean_uc_val = np.median([
    np.median(uc_track)
    for uc_track in individual_uc_tracks
])
uc_ratio_pct = (mean_uc_val / Uj) * 100

# =====================================================================
# FIGURE STYLE AND LAYOUT
# =====================================================================
style_name = (
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)
plt.style.use(style_name)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

band_label = '95% ensemble percentile band'

# =====================================================================
# (a) PHASE-SPACE PORTRAIT
# =====================================================================
ax = axes[0]

ax.plot(
    true_real, true_imag,
    label='LES reference',
    color='black',
    alpha=0.25,
    lw=1.2,
    zorder=1
)

for i in range(num_realizations):
    ax.plot(
        matrix_real[i, :],
        matrix_imag[i, :],
        color='#1f77b4',
        alpha=0.08,
        lw=0.5,
        label='Hybrid FNO–ODE ensemble' if i == 0 else None,
        zorder=2
    )

ax.set_title('(a) Phase-space portrait', fontweight='bold')
ax.set_xlabel(r'$\mathrm{Re}(z)$')
ax.set_ylabel(r'$\mathrm{Im}(z)$')
ax.axis('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(
    frameon=True,
    facecolor='white',
    loc='upper right'
)

# =====================================================================
# (b) WAVE-ENVELOPE DENSITY
# =====================================================================
ax = axes[1]

for i in range(num_realizations):
    sns.kdeplot(
        matrix_envelopes[i, :],
        ax=ax,
        color='#1f77b4',
        alpha=0.2,
        lw=1.0,
        bw_adjust=1.7,
        label='Ensemble realizations' if i == 0 else None,
        zorder=1
    )

sns.kdeplot(
    np.abs(true_complex),
    ax=ax,
    label=r'LES reference $|z|$',
    fill=False,
    color='black',
    lw=1.8,
    linestyle='--',
    zorder=3
)

ax.set_title('(b) Wave-envelope density', fontweight='bold')
ax.set_xlabel(r'Envelope magnitude $|z|$')
ax.set_ylabel('Density')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(
    frameon=True,
    facecolor='white',
    loc='upper right'
)

# =====================================================================
# (c) UNWRAPPED PHASE
# =====================================================================
ax = axes[2]

for phase_track in matrix_phases:
    ax.plot(
        t_model_slice,
        phase_track,
        color='#1f77b4',
        ls='-',
        lw=0.5,
        alpha=0.08,
        zorder=1
    )

ax.plot(
    t_model_slice,
    phase_median,
    color='#d62728',
    lw=2.5,
    label='Ensemble median phase',
    zorder=3
)

ax.plot(
    t_model_slice,
    phase_mean,
    color='#2ca02c',
    lw=2.0,
    ls=':',
    label='Ensemble mean phase',
    zorder=3
)

ax.fill_between(
    t_model_slice,
    phase_lower,
    phase_upper,
    color='#d62728',
    alpha=0.25,
    label=band_label,
    zorder=2
)

ax.plot(
    t_model_slice,
    true_phase_unwrapped,
    color='black',
    lw=1.5,
    ls='--',
    label='LES reference',
    zorder=4
)

ax.set_title('(c) Unwrapped phase', fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unwrapped phase (rad)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(
    frameon=True,
    facecolor='white',
    loc='lower right'
)

# # ADDED: Continuous phase tracking metrics
# ax.text(
#     0.05, 0.95,
#     (
#         f'Median Convection Velocity: {mean_uc_val:.2f} m/s\n'
#         f'Convection Velocity Ratio ($U_c/U_j$): {uc_ratio_pct:.2f}%'
#     ),
#     transform=ax.transAxes,
#     fontsize=9,
#     va='top',
#     ha='left',
#     zorder=5,
#     bbox={
#         'facecolor': 'white',
#         'edgecolor': 'gray',
#         'alpha': 0.85,
#         'boxstyle': 'round,pad=0.3'
#     }
# )

# =====================================================================
# (d) POWER SPECTRAL DENSITY
# =====================================================================
ax = axes[3]

for i in range(matrix_psd.shape[0]):
    ax.semilogx(
        Strouhal_pred,
        psd_db_matrix[i, :],
        color='#1f77b4',
        alpha=0.08,
        lw=0.5,
        zorder=1
    )

ax.semilogx(
    Strouhal_pred,
    psd_median_db,
    color='#d62728',
    lw=2.5,
    label='Ensemble median PSD',
    zorder=3
)

ax.fill_between(
    Strouhal_pred,
    psd_lower_db,
    psd_upper_db,
    color='#d62728',
    alpha=0.25,
    label=band_label,
    zorder=2
)

ax.semilogx(
    Strouhal_true,
    10 * np.log10(Pxx_true),
    color='black',
    lw=1.5,
    ls='--',
    label='LES reference',
    zorder=4
)

ax.set_title('(d) Power spectral density', fontweight='bold')
ax.set_xlabel(r'Strouhal number $St$')
ax.set_ylabel(r'PSD ($10\log_{10}P_{xx}$) [dB]')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(
    frameon=True,
    facecolor='white',
    loc='lower left'
)

# MOVED: PSD discrete peak metrics
# ax.text(
#     0.95, 0.95,
#     (
#         f'LES spectral peak: {f_peak_true:.2f} Hz\n'
#         f'Ensemble median PSD peak: {f_peak_pred:.2f} Hz'
#     ),
#     transform=ax.transAxes,
#     fontsize=9,
#     va='top',
#     ha='right',
#     zorder=5,
#     bbox={
#         'facecolor': 'white',
#         'edgecolor': 'gray',
#         'alpha': 0.85,
#         'boxstyle': 'round,pad=0.3'
#     }
# )

# =====================================================================
# SAVE AND REPORT
# =====================================================================
fig.tight_layout(pad=1.5)

output_fig_name = 'fno_sindy_pure_ensemble_evaluation.png'
fig.savefig(output_fig_name, dpi=300, bbox_inches='tight')

print(f'Plotting complete! Figure saved as: {output_fig_name}')
print(
    f'[METRIC] True Target Dominant Frequency: '
    f'{f_peak_true:.2f} Hz'
)
print(
    f'[METRIC] Ensemble Median Spectral Peak: '
    f'{f_peak_pred:.2f} Hz'
)
print(
    f'[METRIC] Median Convection Velocity: '
    f'{mean_uc_val:.2f} m/s'
)
print(
    f'[METRIC] Convection Velocity Ratio (Uc/Uj): '
    f'{uc_ratio_pct:.2f}%'
)

plt.show()
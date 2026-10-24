import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
# Import your professor's custom spectrum functions
from find_spectrum import hann2, pwelch2

# 1. LOAD PRISTINE TRAJECTORY DATA (Updated for 7 columns)
data_file = 'fno_sindy_autonomous_rollout.txt'
data = np.loadtxt(data_file, skiprows=1)  # skip header row

t_model_slice   = data[:, 0]
true_real       = data[:, 1]
true_imag       = data[:, 2]
pred_real_fixed = data[:, 3]  # FNO-corrected for spatial tracking
pred_imag_fixed = data[:, 4]  # FNO-corrected for spatial tracking
pred_real_pure  = data[:, 5]  # Pure phase-pure ODE for spectral metrics
pred_imag_pure  = data[:, 6]  # Pure phase-pure ODE for spectral metrics

# Reconstruct complex numbers
true_complex      = true_real + 1j * true_imag
pred_complex_fixed = pred_real_fixed + 1j * pred_imag_fixed
pred_complex_pure  = pred_real_pure + 1j * pred_imag_pure

# Calculate sampling parameters
dt = np.mean(np.diff(t_model_slice))
fs = 1.0 / dt

# Unwrapped phase tracking (Use pure ODE values to ensure zero FNO jitter)
true_phase_unwrapped = np.unwrap(np.angle(true_complex))
pred_phase_unwrapped = np.unwrap(np.angle(pred_complex_pure))


# Calculate mean linear slope trends
slope_true, _ = np.polyfit(t_model_slice, true_phase_unwrapped, 1)
slope_pred, _ = np.polyfit(t_model_slice, pred_phase_unwrapped, 1)
f0_true = slope_true / (2 * np.pi)
f0_pred = slope_pred / (2 * np.pi)

# =====================================================================
# SPECTRUM EVALUATION FOR PEAK ESTIMATION (DECOUPLED FROM FNO JITTER)
# =====================================================================
nFFT = 128
window = hann2(nFFT)
idx_test = 0

pure_test_true = true_complex[idx_test:]
pure_test_pred = pred_complex_pure[idx_test:] # Isolated pure ODE path

Pxx_true, F_true = pwelch2(pure_test_true, nFFT, fs, window)
Pxx_pred, F_pred = pwelch2(pure_test_pred, nFFT, fs, window)


f_peak_true = F_true[np.argmax(Pxx_true)]
f_peak_pred = F_pred[np.argmax(Pxx_pred)]

Dj, Uj = 0.0169, 302.0
Strouhal_true = F_true * Dj / Uj
Strouhal_pred = F_pred * Dj / Uj

# =====================================================================
# SET UP THE PLOTTING ENVIRONMENT
# =====================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10, 
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))

# SUBPLOT 1: PHASE SPACE PORTRAIT ATTRACTOR (Uses FNO Fixed for spatial matching)
ax = axes[0]
ax.plot(true_real, true_imag, label='True Jet Attractor', color='black', alpha=0.3, lw=1)
ax.plot(pred_real_fixed, pred_imag_fixed, label='Hybrid FNO-ODE Rollout', color='#1f77b4', alpha=0.8, lw=1.2)
ax.set_title("Phase Space Portrait Alignment", fontweight='bold')
ax.set_xlabel(r"$\mathrm{Re}(a_1)$")
ax.set_ylabel(r"$\mathrm{Im}(a_1)$")
ax.axis('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', loc='upper right')

# SUBPLOT 2: KDE PROBABILITY DENSITY OF WAVE ENVELOPE (Uses FNO Fixed for amplitude distribution)
ax = axes[1]
sns.kdeplot(np.abs(true_complex), ax=ax, label='True Data Envelope $|z|$', fill=True, alpha=0.2, color='black', lw=2)
sns.kdeplot(np.abs(pred_complex_fixed), ax=ax, label='Hybrid FNO-ODE Envelope', fill=True, alpha=0.2, color='#1f77b4', lw=2)
ax.set_title("Wave Envelope Probability Density", fontweight='bold')
ax.set_xlabel("Envelope Magnitude")
ax.set_ylabel("Density")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', loc='upper right')

# SUBPLOT 3: UNWRAPPED PHASE VELOCITY ACCUMULATION (Uses Pure ODE)
ax = axes[2]
ax.plot(t_model_slice, true_phase_unwrapped, color='black', lw=2, label='True Phase')
ax.plot(t_model_slice, pred_phase_unwrapped, color='#1f77b4', ls='--', lw=1.5, label='FNO-ODE Phase')
ax.set_title("Unwrapped Phase Growth", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accumulated Phase (rad)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', loc='upper left')

ax.text(0.05, 0.45, f"True Peak: {f_peak_true:.2f} Hz\nPred Peak: {f_peak_pred:.2f} Hz", 
        transform=ax.transAxes, fontsize=9.5, va='top',
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.3'))

# SUBPLOT 4: POWER SPECTRAL DENSITY (Clean, sharp peaks from unpolluted ODE data)
ax = axes[3]
ax.semilogx(Strouhal_true, 10 * np.log10(Pxx_true), color='black', linewidth=2, label='True Jet State')
ax.semilogx(Strouhal_pred, 10 * np.log10(Pxx_pred), color='#1f77b4', linestyle='--', linewidth=1.5, label='FNO-ODE Rollout')
ax.set_title("State Power Spectral Density", fontweight='bold')
ax.set_xlabel('Strouhal Number ($St$)')
ax.set_ylabel('PSD ($10 \log_{10}(P_{xx})$) [dB]')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='white', loc='lower left')

plt.tight_layout()
output_fig_name = 'fno_sindy_autonomous_evaluation_with_spectral_phase.png'
plt.savefig(output_fig_name, dpi=300)
print(f"Metrics compiled across full step sequences! Saved to: {output_fig_name}")
plt.show()

# PRINT RESULTS
print('\n' + '='*50)
print('          AEROACOUSTIC FREQUENCY EVALUATION          ')
print('='*50)
print(f'True Welch Peak Freq : {f_peak_true:.2f} Hz')
print(f'Pred Welch Peak Freq : {f_peak_pred:.2f} Hz')
print(f'True Acoustic Peak Error: {abs(f_peak_true - f_peak_pred):.2f} Hz')
print('='*50 + '\n')
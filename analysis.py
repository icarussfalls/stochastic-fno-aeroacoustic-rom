import numpy as np
import pandas as pd
from find_phase import *
from find_spectrum import hann2, pwelch2
import matplotlib.pyplot as plt

def load_complex_signal(real_file, imag_file, index):
    real_data = np.loadtxt(real_file)[index:]
    imag_data = np.loadtxt(imag_file)[index:]
    t = real_data[:, 0]
    signal = real_data[:, 1] + 1j * imag_data[:, 1]  # Combine real and imaginary parts
    return t, signal

real_file = '09042025_TowneSPOD_St0_1_M1_Real_all_5348_points_mode1.txt'
imag_file = '09042025_TowneSPOD_St0_1_M1_Imag_all_5348_points_mode1.txt'
t, signal = load_complex_signal(real_file, imag_file, index=64)


# Load your data
# re = np.loadtxt("09042025_TowneSPOD_St0_1_M1_Real_all_5348_points_mode1.txt")
# im = np.loadtxt("09042025_TowneSPOD_St0_1_M1_Imag_all_5348_points_mode1.txt")

# # Compute the Hilbert Transform of the Real part
# hilbert_of_re = hilbert(re).imag 

# # Compare
# correlation = np.corrcoef(im, hilbert_of_re)[0, 1]
# print(f"Correlation between Imag file and Hilbert(Real): {correlation:.4f}")

# data = np.loadtxt("combined_reim_results.txt", skiprows=1)
# data = data[3000:]

# predicted_df = pd.read_csv("fno_forecast_results.csv")
# predicted_df = pd.read_csv("fno_reim_forecast_results.csv")
# predicted_df = pd.read_csv("fno_complex_forecast_results.csv")
predicted_df = pd.read_csv("/Users/icarus/Downloads/fno_reim_forecast_results_8.csv")

re_pred = predicted_df['re_predicted'].values
im_pred = predicted_df['im_predicted'].values

# def revert_symlog(pred_log):
#     return np.sign(pred_log) * (np.exp(np.abs(pred_log)) - 1)

# re_pred = revert_symlog(re_pred)
# im_pred = revert_symlog(im_pred)

# preds = np.loadtxt(file)
# print(preds)
# preds = np.loadtxt(file,skiprows=1)


# these are the columns name
# Time(s) Pred_R Pred_I Truth_R Truth_I
# 0, 1, 2, 3, 4
# t = preds[:, 0]

# original_signal = signal
fno_reconstructed = re_pred + 1j * im_pred
original_signal = signal
# original_signal = data[:,1] + 1j * data[:,2]
# t = data[:,0]

# this is for spectrum
# Load the signal

# t is just the time in arranged across the rows

# Welch's parameters
Fs = 1 / np.mean(np.diff(t))  # Sampling frequency from the time vector
nFFT = 128
window = hann2(nFFT)

# Compute PSD
Pxx, F = pwelch2(original_signal, nFFT, Fs, window)

Dj = 0.0169;
Uj = 302;
Strouhal = F*Dj/Uj;

# Plot PSD
plt.figure()
plt.semilogx(Strouhal, 10 * np.log10(Pxx), 'k-', linewidth=2, label='original')
plt.title('Power Spectral Density')
plt.xlabel('St')

plt.grid(True)

# plt.show()


# Compute PSD
Pxx1, F = pwelch2(fno_reconstructed, nFFT, Fs, window)

Dj = 0.0169;
Uj = 302;
Strouhal = F*Dj/Uj;

# Plot PSD
# plt.figure()
plt.semilogx(Strouhal, 10 * np.log10(Pxx1), 'k-', linewidth=1, label='fno')
plt.title('Power Spectral Density')
plt.xlabel('St')
plt.legend()

plt.grid(True)

plt.show()


# # for phase
analyze_signal_combined(original_signal, fno_reconstructed, t,
                        labels=('Original', 'FNO Reconstructed'))
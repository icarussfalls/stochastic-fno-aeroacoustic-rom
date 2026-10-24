import numpy as np
import matplotlib.pyplot as plt

# Hann window generator
def hann2(window_length):
    if window_length == 0:
        window_length = 100
    N = window_length - 1
    num = np.linspace(0, N, window_length)
    return 0.5 * (1 - np.cos(2 * np.pi * num / N))

# Welch's PSD calculation
def pwelch2(X, nFFT, Fs, window):
    dt = 1 / Fs
    df = Fs / nFFT
    w = window.flatten()
    X = X.flatten()
    hop = nFFT // 2
    k = (len(X) - nFFT) // hop + 1
    
    KMU = k * np.sum(w**2)
    P = np.zeros(nFFT)
    
    for i in range(k):
        start_index = i * hop
        xw = w * X[start_index:start_index + nFFT]
        P += np.abs(np.fft.fft(xw, n=nFFT))**2
        
    Pxx = dt * P[:nFFT // 2 + 1] / KMU
    F = np.arange(nFFT // 2 + 1) * df

    return Pxx, F

# Load the real and imaginary parts of the signal from .txt files
def load_complex_signal(real_file, imag_file):
    real_data = np.loadtxt(real_file)
    imag_data = np.loadtxt(imag_file)
    t = real_data[:, 0]  # Assuming the time vector is in the first column
    signal = real_data[:, 1] + 1j * imag_data[:, 1]  # Combine real and imaginary parts
    return t, signal

# # ============================================= St = 0.1 ====================================== #
# real_file = '09042025_TowneSPOD_St0_1_M1_Real_all_5348_points_mode1.txt'
# imag_file = '09042025_TowneSPOD_St0_1_M1_Imag_all_5348_points_mode1.txt'
# # ============================================================================================ #

# # Load the signal
# t, signal = load_complex_signal(real_file, imag_file)

# # t is just the time in arranged across the rows

# # Welch's parameters
# Fs = 1 / np.mean(np.diff(t))  # Sampling frequency from the time vector
# nFFT = 128
# window = hann2(nFFT)

# # Compute PSD
# Pxx, F = pwelch2(signal, nFFT, Fs, window)

# Dj = 0.0169;
# Uj = 302;
# Strouhal = F*Dj/Uj;

# # Plot PSD
# plt.figure()
# plt.semilogx(Strouhal, 10 * np.log10(Pxx), 'k-', linewidth=2)
# plt.title('Power Spectral Density')
# plt.xlabel('St')

# plt.grid(True)

# plt.show()

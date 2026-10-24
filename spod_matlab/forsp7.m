%% SPOD Analysis of Mach 0.9 Turbulent Jet (Full Continuous Tracking)
% Large-eddy simulation data from the database described in [1].
% Physical interpretation of SPOD results in [2].
% Comprehensive derivation of SPOD in [3].
%
% References:
% [1] G. A. Brès et al., Importance of the nozzle-exit boundary-layer
%     state in subsonic turbulent jets, J. Fluid Mech. 851, 83-124, 2018
% [2] O. T. Schmidt, A. Towne, G. Rigas, T. Colonius, G. A. Bres,
%     Spectral analysis of jet turbulence, J. Fluid Mech. 855, 953-982, 2018
% [3] A. Towne, O. T. Schmidt, T. Colonius, Spectral proper orthogonal
%     decomposition and its relationship to DMD and resolvent analysis,
%     J. Fluid Mech. 847, 821-867, 2018

%% SPOD Analysis of Mach 0.9 Turbulent Jet (Full Continuous Tracking)
% Large-eddy simulation data from the database described in [1].
% Physical interpretation of SPOD results in [2].
% Comprehensive derivation of SPOD in [3].
%
% References:
% [1] G. A. Brès et al., Importance of the nozzle-exit boundary-layer
%     state in subsonic turbulent jets, J. Fluid Mech. 851, 83-124, 2018
% [2] O. T. Schmidt, A. Towne, G. Rigas, T. Colonius, G. A. Bres,
%     Spectral analysis of jet turbulence, J. Fluid Mech. 855, 953-982, 2018
% [3] A. Towne, O. T. Schmidt, T. Colonius, Spectral proper orthogonal
%     decomposition and its relationship to DMD and resolvent analysis,
%     J. Fluid Mech. 847, 821-867, 2018

%% Environment Setup & Constants
clc; clear variables; close all;
addpath('utils');

Uj = 302;       % Jet velocity (m/s)
Dj = 0.0169;    % Jet diameter (m)

%% Data Loading & Preprocessing
disp('Loading the structured jet database (sp7.mat)...')
load(fullfile('jet_data', 'sp7.mat'), 'Pa_n0', 't_inp', 'x_ij', 'y_ij');

% Reduce time samples to fit blocks for nDFT 128
p = squeeze(Pa_n0(1, :, :, 1:5348)); 

% Reorder to [nt × nr × nx] -> [5348 × 51 × 100]
p = permute(p, [3, 2, 1]); 

% Normalize coordinates by Jet Diameter
x = x_ij' ./ Dj; % [nx × 1] (normalized x-coordinate)
r = y_ij' ./ Dj; % [nr × 1] (normalized r-coordinate)

% Time step calculation
dt = mean(diff(t_inp)); 

% Subtract mean to get fluctuations
p_mean = mean(p, 1);
p_fluct = p - p_mean;

[nt, nr, nx] = size(p);

disp('=== Data Loaded ===');
disp(['p:  ', mat2str(size(p))]);
disp(['x:  ', mat2str(size(x))]);
disp(['r:  ', mat2str(size(r))]);
disp(['dt: ', num2str(dt)]);

%% Fluctuation Metrics Verification
% Find the maximum absolute instantaneous fluctuation across ALL snapshots and grid points
max_instantaneous_p_fluct = max(abs(p_fluct(:)));

fprintf('\n=== FLUCTUATION VERIFICATION ===\n');
fprintf('Maximum Instantaneous Fluctuation Peak: %6.2f Pa\n', max_instantaneous_p_fluct);

% Spatial map [nr × nx]
p_rms_field = squeeze(sqrt(mean(real(p_fluct).^2, 1))); 

% Find the absolute peak fluctuation amplitude in the entire near-field domain
max_p_rms = max(p_rms_field(:));

fprintf('\n=== AERODYNAMIC VERIFICATION METRICS ===\n');
fprintf('Peak Fluctuation Amplitude (p_rms): %6.2f Pa\n\n', max_p_rms);

%% SPOD Parameters Setup
mean_dt = mean(diff(t_inp));
t_uniform = t_inp(1) + (0:nt-1) * dt;
n_snapshots = length(t_uniform);

% SPOD Configuration
window_length = 128;
overlap = floor(window_length / 2); % 50% block overlap
win = hann(window_length);
weight = trapzWeightsPolar(r(:, 1), x(1, :)); % Weight vector for integration

%% Run SPOD and retain blockwise coefficients
[L, P, f, Lc, A_blocks] = spod( ...
    p_fluct, win, weight, overlap, mean_dt);

St_vector = f * Dj / Uj;

%% continuous coefficients for modelling
n_modes_used = 1;
A_continuous = tcoeffs( ...
    p_fluct, P, win, weight, n_modes_used);

A_coeffs_continuous_real = single(real(A_continuous));
A_coeffs_continuous_imag = single(imag(A_continuous));

%% Save a separate file for reconstruction verification
% This preserves your existing database.
save('spod_reconstruction_check.mat', ...
    'P', 'L', 'f', 'St_vector', ...
    'A_blocks', ...
    'A_coeffs_continuous_real', ...
    'A_coeffs_continuous_imag', ...
    'win', 'overlap', 'mean_dt', ...
    'p_mean', 'x_ij', 'y_ij', 'weight', ...
    'Uj', 'Dj', ...
    '-v7.3');
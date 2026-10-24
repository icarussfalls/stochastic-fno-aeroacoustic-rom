%% Reconstruct the retained SPOD pressure contribution using ALL coefficients
% Matches the supplied tcoeffs.m: N-point sliding windows, hop = 1.
% Uses the window-weighted overlap convention of invspod, evaluated without
% intermediate division by w. No coefficient decimation and no Hilbert transform.
% This is synthesis of the selected mode/frequency, not a broadband full field.
% Arbitrary model coefficients need not be consistent STFT analysis outputs;
% overlap synthesis can change their time-domain statistics.
%
% The stock invspod.m infers block count from P's mode dimension and is not
% called here. The local function implements the same overlap formula with
% explicit frame count and the padding/time origin used by tcoeffs.

clearvars;
spod_path = 'spod_reconstruction_check.mat';
npy_path = '/Users/icarus/Desktop/SPODcode/fno_ensemble_coefficients.npy';
output_dir = 'fno_invspod_all_samples_output';
target_st = 0.1;
mode_index = 1;
ensemble_start_index = 1;  % MATLAB index in original tcoeffs record
save_all_pressure = true;

assert(isfile(spod_path), 'Cannot find SPOD file: %s',spod_path);
assert(isfile(npy_path), 'Cannot find coefficient file: %s',npy_path);
% A fresh directory prevents old and new realization files being mixed.
if isfolder(output_dir)
    assert(isempty(dir(fullfile(output_dir,'*.mat'))), ...
        'Output directory contains MAT files. Choose a fresh output_dir.');
end
S = load(spod_path);
required = {'P','win','St_vector','A_coeffs_continuous_real', ...
    'A_coeffs_continuous_imag','mean_dt'};
for j=1:numel(required)
    assert(isfield(S,required{j}), 'Missing SPOD variable: %s',required{j});
end

%% Read the complex NumPy ensemble
bridge_file = [tempname '.mat'];
try
    pyrun(["import numpy as np", ...
        "from scipy.io import savemat", ...
        "z = np.load(npy_path, allow_pickle=False)", ...
        "assert z.ndim == 2 and np.iscomplexobj(z), 'Expected complex [ensemble,time] array'", ...
        "savemat(mat_path, {'ensemble_coeffs': z})"], ...
        npy_path=string(npy_path), mat_path=string(bridge_file));
    D = load(bridge_file,'ensemble_coeffs');
    delete(bridge_file);
catch ME
    if isfile(bridge_file), delete(bridge_file); end
    error('NPY loading failed: %s',ME.message);
end
Z = double(D.ensemble_coeffs);
clear D;
assert(ismatrix(Z) && all(isfinite(Z(:))), 'Invalid ensemble array.');
[n_ens,n_time] = size(Z);

%% Original spatial basis and matched LES coefficient record
w = double(S.win(:));
N = numel(w);
assert(mod(N,2)==0 && N>=4, 'Expected an even window length.');
assert(all(isfinite(w)) && all(w>=0) && sum(w)>0, 'Invalid window.');
assert(n_time>2*N, 'Record is too short for the interior comparison.');
[~,fi] = min(abs(S.St_vector(:)-target_st));
st_actual = S.St_vector(fi);
assert(ndims(S.P)==4, 'Expected P: [frequency, radial, streamwise, mode].');
assert(size(S.P,1)==N/2+1 && fi>1 && fi<N/2+1, ...
    'This script handles an interior frequency bin of real-input SPOD.');
assert(mode_index>=1 && mode_index<=size(S.P,4), 'Invalid mode index.');
nr = size(S.P,2); nx = size(S.P,3);
psi = double(reshape(S.P(fi,:,:,mode_index),nr,nx));
% Do not independently renormalize psi or the coefficients.
z_original = double(reshape(S.A_coeffs_continuous_real(fi,mode_index,:),[],1)) ...
    + 1i*double(reshape(S.A_coeffs_continuous_imag(fi,mode_index,:),[],1));
assert(all(isfinite(z_original)), 'Non-finite LES coefficients.');
assert(ensemble_start_index>=1 && ensemble_start_index==floor(ensemble_start_index));
original_indices = ensemble_start_index+(0:n_time-1);
assert(original_indices(end)<=numel(z_original), 'LES does not cover the ensemble interval.');
z_reference = z_original(original_indices);
z_reference = z_reference(:);

%% Reverse the boundary correction encoded by tcoeffs
% Both arrays represent coefficients in the original tcoeffs convention.
% Use ORIGINAL record indices/length, not the shorter predicted record length.
% For the supplied full interior windows this correction equals 1.
correction = tcoeffs_correction(original_indices,numel(z_original),w);

%% Optional check against directly exported block coefficients
% This is verification only. It does NOT select or discard prediction samples.
mapping_error = NaN;
if isfield(S,'A_blocks') && isfield(S,'overlap')
    original_hop = N-S.overlap;
    nblocks = size(S.A_blocks,3);
    original_centres = (0:nblocks-1)*original_hop+N/2+1;
    if original_centres(end)<=numel(z_original)
        b = double(reshape(S.A_blocks(fi,mode_index,:),[],1));
        mapping_error = norm(z_original(original_centres)-b)/max(norm(b),eps);
        fprintf('[CHECK] Original LES block mapping: %.3g\n',mapping_error);
        assert(mapping_error<1e-5, 'Original coefficient conventions do not match.');
    end
end

%% One coefficient per sliding frame; reconstruct original sample positions
hop = 1;
centres = 1:n_time;
t_relative = (0:n_time-1)'*S.mean_dt;
valid_time = false(n_time,1);
valid_time(N/2+1:n_time-N/2) = true;
% Output includes all n_time samples. Statistics exclude outer half-windows
% where the available coefficient-frame coverage is incomplete.
synthesis_method = 'tcoeffs hop-one window-weighted inverse synthesis';
if ~isfolder(output_dir), mkdir(output_dir); end
fprintf('[INFO] St %.6f, mode %d, %d realizations\n',st_actual,mode_index,n_ens);
fprintf('[INFO] Using ALL %d coefficient frames per realization; synthesis hop = 1\n',n_time);
fprintf('[INFO] Pressure output: %d samples; RMS interval: %d samples\n',n_time,nnz(valid_time));

%% Matched LES reconstruction
c_reference = synthesize_all_frames(z_reference,correction,w,fi);
p_reference = real(reshape(c_reference,[],1,1).*reshape(psi,1,nr,nx));
rms_reference = reshape(sqrt(mean(p_reference(valid_time,:,:).^2,1)),nr,nx);
save(fullfile(output_dir,'les_rank1_reference.mat'), ...
    'p_reference','rms_reference','valid_time','t_relative', ...
    'c_reference','synthesis_method','ensemble_start_index','-v7.3');
clear p_reference;

%% Full ensemble: one pressure realization in memory at a time
rms_ensemble = zeros(n_ens,nr,nx);
for e=1:n_ens
    % Every Z(e,i) contributes: there is no Z(e,centres(1:64:end)).
    c_predicted = synthesize_all_frames(Z(e,:).',correction,w,fi);
    p_predicted = real(reshape(c_predicted,[],1,1).*reshape(psi,1,nr,nx));
    assert(all(isfinite(p_predicted(:))), 'Non-finite reconstructed pressure.');
    rms_ensemble(e,:,:) = sqrt(mean(p_predicted(valid_time,:,:).^2,1));
    if save_all_pressure || e==1
        realization_index=e;
        save(fullfile(output_dir,sprintf('pressure_realization_%02d.mat',e)), ...
            'p_predicted','c_predicted','valid_time','t_relative', ...
            'realization_index','synthesis_method','ensemble_start_index','-v7.3');
    end
    fprintf('[DONE] Realization %d/%d: all %d coefficients used\n',e,n_ens,n_time);
end
rms_mean = reshape(mean(rms_ensemble,1),nr,nx);
rms_pooled = reshape(sqrt(mean(rms_ensemble.^2,1)),nr,nx);
reference_mean_square = sum(rms_reference(:).^2);
model_mean_square = sum(rms_pooled(:).^2);
assert(reference_mean_square>0 && model_mean_square>0);
energy_ratio = model_mean_square/reference_mean_square;
pressure_mean_square_bias_pct = 100*(energy_ratio-1);
pressure_mean_square_error_pct = abs(pressure_mean_square_bias_pct);
level_offset_dB = 10*log10(energy_ratio);
rms_relative_error_pct = 100*norm(rms_pooled(:)-rms_reference(:))/norm(rms_reference(:));
% All continuous coefficient samples: a separate diagnostic, not pressure energy.
coefficient_mean_square_les = mean(abs(z_reference).^2);
coefficient_mean_square_model = mean(abs(Z(:)).^2);
coefficient_bias_pct = 100*(coefficient_mean_square_model/coefficient_mean_square_les-1);

save(fullfile(output_dir,'ensemble_rms_summary.mat'), ...
    'rms_ensemble','rms_mean','rms_pooled','rms_reference','level_offset_dB', ...
    'reference_mean_square','model_mean_square','energy_ratio', ...
    'pressure_mean_square_bias_pct','pressure_mean_square_error_pct', ...
    'rms_relative_error_pct','coefficient_mean_square_les', ...
    'coefficient_mean_square_model','coefficient_bias_pct', ...
    'st_actual','mode_index','mapping_error','valid_time','t_relative', ...
    'ensemble_start_index','original_indices','centres','w','hop', ...
    'correction','synthesis_method','-v7.3');
fprintf('\n[METRIC] Coefficient mean-square bias: %+.6f%%\n',coefficient_bias_pct);
fprintf('[METRIC] Pressure mean-square bias: %+.6f%%\n',pressure_mean_square_bias_pct);
fprintf('[METRIC] Relative pressure mean-square error: %.6f%%\n',pressure_mean_square_error_pct);
fprintf('[METRIC] RMS-field relative error: %.6f%%\n',rms_relative_error_pct);
fprintf('[METRIC] Spatial RMS-map level offset: %+.6f dB\n',level_offset_dB);
fprintf('[SAVED] %s\n',output_dir);

function corr = tcoeffs_correction(indices,nt_original,w)
    % Exact branch/index convention from the supplied tcoeffs.m.
    N=numel(w); half=N/2; corr=ones(numel(indices),1);
    for j=1:numel(indices)
        i=indices(j);
        if i<half+1
            corr(j)=sum(w)/sum(w(half-i+1:N));
        elseif i>nt_original-half+1
            corr(j)=sum(w)/sum(w(1:nt_original+half-i));
        end
    end
    assert(all(isfinite(corr)) && all(corr>0), 'Invalid boundary correction.');
end

function c = synthesize_all_frames(z,corr,w,fi)
    % Each frame begins at its coefficient index in the padded record.
    % For a selected positive bin and its conjugate, the windowed inverse
    % FFT equals Re{2*sum(w)/N * z/corr * exp(i*2*pi*k*n/N) * psi}.
    % Overlap numerator is the sum of those windowed inverse frames;
    % denominator is the sum of their analysis windows (invspod convention).
    % Rank-one separability lets us do this convolution before multiplying psi.
    z=z(:); N=numel(w); T=numel(z);
    kernel=exp(2i*pi*(fi-1)*(0:N-1)'/N);
    numerator=(2*sum(w)/N)*conv(z./corr,kernel,'full');
    support=conv(ones(T,1),w,'full');
    original_samples=N/2+(1:T);
    assert(all(support(original_samples)>0), 'Missing window support.');
    c=numerator(original_samples)./support(original_samples);
    c=c(:);
end

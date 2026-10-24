# Stochastic FNO Aeroacoustic ROM

Research code for **Stochastic Fourier Neural Operator Closure for Reduced-Order Modeling of Turbulent Jet Pressure**, by Pradip Pokhrel, Matthias Winter, Vasily Gryazev, and Sergey Karabasov.

This model generates long sequences of turbulent jet pressure coefficients from a short initial history. It combines a second-order ordinary differential equation (ODE) with a stochastic Fourier neural operator (FNO). The ODE supplies phase, and the FNO predicts amplitude mean and standard deviation. Pressure fields are reconstructed using one fixed spatial mode from spectral proper orthogonal decomposition (SPOD).

The main workflow is **MATLAB SPOD preparation → Python model training and generation → MATLAB pressure reconstruction → Python evaluation**.

## Method

1. Compute SPOD from mean-subtracted large-eddy simulation (LES) pressure. Select the leading axisymmetric mode at the frequency bin nearest the target Strouhal number, `St = 0.1`.
2. Use the continuous complex coefficient to fit a second-order ODE by Operator Inference. Phillips' acoustic analogy motivates the ODE structure.
3. Train the FNO using the recent coefficient history and the ODE forecast. The main model uses **independent Gaussian noise during training**, with a trajectory Energy Score loss. The ODE coefficients remain trainable alongside the network weights.
4. Start from 64 observed coefficient samples and generate successive 64-sample blocks. Each generated block becomes the next input history. At inference, noise is filtered using the local ODE spectrum, normalized, and used to sample amplitudes. Negative amplitudes are clipped to zero before combining them with the ODE phase.
5. Process each complete generated sequence offline: apply an ODE-derived Fourier amplitude mask, rank-match amplitudes to that realization's raw generated amplitudes, and apply a half-Hann spectral taper to reduce rapid changes at block joins.
6. Reconstruct the pressure contribution of the selected frequency and spatial mode using all generated coefficient samples.

The final spectral processing uses the complete sequence. The processed results therefore describe autonomous statistical generation with offline processing. The model retains one spatial mode at one frequency; the reconstructed pressure is that mode's contribution to the LES pressure field.

## Main files

| File | Purpose |
| --- | --- |
| [`spod_matlab/forsp7.m`](spod_matlab/forsp7.m) | Prepare pressure fluctuations, compute SPOD, and export continuous coefficients and reconstruction data. |
| [`dissertation_code.ipynb`](dissertation_code.ipynb) | Fit the ODE; define and train the FNO; generate, process, export, and evaluate coefficient ensembles. |
| [`spod_matlab/reconstruct_fno_all_samples.m`](spod_matlab/reconstruct_fno_all_samples.m) | Reconstruct the selected pressure contribution for the LES reference and every ensemble realization. |
| [`find_spectrum.py`](find_spectrum.py) | Hann-window and Welch spectrum functions used by the notebook and plotting scripts. |
| [`generate_metrics.py`](generate_metrics.py) | Generate ensemble diagnostic plots from the exported summary and distribution files. |
| [`spod_matlab/`](spod_matlab/) | SPOD routines, examples, and project-specific MATLAB scripts. |

The repository also contains checkpoints, figures, text exports, and exploratory scripts. The workflow below follows the main notebook and the two MATLAB scripts.

## Requirements

- Python with JupyterLab, NumPy, SciPy, pandas, h5py, Matplotlib, seaborn, scikit-image, PyTorch, and torchdiffeq. The notebook records Python 3.12.11; dependency versions are not pinned.
- MATLAB with `hann` available, for example through Signal Processing Toolbox.
- MATLAB's Python integration, including `pyrun`, configured with a compatible Python environment containing NumPy and SciPy. The reconstruction script uses these to read the complex NumPy array.

From a terminal:

```bash
git clone https://github.com/icarussfalls/stochastic-fno-aeroacoustic-rom.git
cd stochastic-fno-aeroacoustic-rom
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy scipy pandas h5py matplotlib seaborn scikit-image torch torchdiffeq
jupyter lab dissertation_code.ipynb
```

The notebook selects CUDA when available and otherwise uses the CPU. Configure MATLAB's Python environment separately for the reconstruction step.

## Data

The study uses axisymmetric pressure from the NASA Small Hot Jet Acoustic Rig (SHJAR), Set Point 7 (SP7), LES case discussed by Gryazev et al. [1]. The underlying LES dataset is **not included** in this repository. Data supporting the study are available from the corresponding author upon reasonable request.

Place the input file at `spod_matlab/jet_data/sp7.mat`. The preparation script expects:

| Variable | Contents |
| --- | --- |
| `Pa_n0` | Axisymmetric pressure array; the script selects `Pa_n0(1, :, :, 1:5348)` and rearranges it to time × radius × streamwise position. |
| `t_inp` | Snapshot times in seconds. |
| `x_ij`, `y_ij` | Streamwise and radial coordinate arrays in metres. |

The supplied configuration uses 5348 snapshots on a 51 × 100 radial–streamwise grid, jet diameter `Dj = 0.0169 m`, and reference velocity `Uj = 302 m/s`.

## Run the workflow

### 1. Compute SPOD and save the coefficients

In MATLAB, set the working directory to `spod_matlab` and run:

```matlab
forsp7

% Include the time array required by the Python notebook.
save('spod_reconstruction_check.mat', 't_uniform', '-append');
```

`forsp7.m` computes the temporal mean, subtracts it from the pressure, and uses a 128-sample Hann window with 64-sample overlap and cylindrical integration weights. It calls `tcoeffs` to obtain continuous coefficients for the first mode at each frequency, with a one-sample frame hop.

The output `spod_reconstruction_check.mat` contains the spatial modes, frequency grid, block coefficients, continuous real and imaginary coefficients, window, sampling interval, and coordinates. The additional `save` command above is needed because the script currently creates `t_uniform` without including it in its save list. Alternatively, add `'t_uniform'` to that list before running the script.

The script constructs a uniform time array using the mean interval in `t_inp`; it does not interpolate the pressure samples. Input pressure must already be sampled consistently with that interval.

### 2. Fit the ODE and train the FNO

Open `dissertation_code.ipynb` from the repository root. In its first cell, set:

```python
data_path = 'spod_matlab/spod_reconstruction_check.mat'
```

Update later data-loading cells to use the same file. Several cells currently contain absolute `/Users/icarus/...` paths and the earlier filename `spod_complete_continuous_database.mat`; replace these with paths on your machine. Use the same SPOD basis and coefficient export throughout training and reconstruction.

Run the ODE fitting, model definition, normalization, and data-loader cells, followed by the **uncolored training** cell. The alternative colored-noise training block is commented out and is not needed for the main configuration.

| Setting | Main configuration |
| --- | --- |
| Selected mode | First mode; nearest frequency bin to `St = 0.1` |
| Actual selected bin for the study | `St ≈ 0.0854`, approximately 1525.84 Hz |
| Training / validation samples | First 2700 / next 300; remaining samples form the test interval |
| History / prediction length | 64 / 64 samples |
| Batch size | 8 |
| FNO width / retained Fourier bins | 64 / 8 |
| Optimizer | AdamW |
| Initial learning rate / weight decay | `1e-4` / `1e-2` |
| Maximum epochs / early-stopping patience | 100 / 20 epochs |
| Monte Carlo samples per Energy Score evaluation | 100 |

The training loop saves the best validation weights to `best_model.pt`. Before generation, enable the checkpoint-loading cell to restore those weights; the training loop does not automatically restore the best checkpoint at its end. If using an existing checkpoint, first construct the model and its required normalization values. Keep each checkpoint with the configuration and exports from its own run.

### 3. Generate and export the ensemble

Run the inference section, the offline spectral-processing cell, and the export cells. The default ensemble contains 30 realizations. Each uses 64 observed initialization samples followed by 82 generated blocks: **5312 samples in total, including 5248 generated samples**. The last incomplete block is omitted.

The active export cells write:

| Output | Contents |
| --- | --- |
| `fno_sindy_ensemble_summary1.txt` | Time, LES real and imaginary coefficients, ensemble means and standard deviations, and the deterministic ODE reference. |
| `fno_sindy_ensemble_distributions1.txt` | Flattened real and imaginary samples from all processed realizations. |
| `fno_ensemble_coefficients.npy` | Complex array of shape `(30, 5312)` for pressure reconstruction. |

Run the complex-array export after offline processing so that it uses the final `matrix_real` and `matrix_imag` arrays. Some plotting cells and `generate_metrics.py` read filenames without the `1` suffix. Update those inputs to the files from the same run before evaluating them.

### 4. Reconstruct pressure in MATLAB

In `spod_matlab/reconstruct_fno_all_samples.m`, set:

```matlab
spod_path = 'spod_reconstruction_check.mat';
npy_path = fullfile('..', 'fno_ensemble_coefficients.npy');
output_dir = 'fno_invspod_all_samples_output';
```

Run `reconstruct_fno_all_samples` from the `spod_matlab` directory. Use an output directory without existing `.mat` files; the script checks this to prevent mixing results from different runs.

The reconstruction uses the first spatial mode at the selected frequency, its conjugate contribution, and window-overlap synthesis with a one-sample hop. Every coefficient sample contributes. Both the LES reference and model ensemble use the same spatial mode and reconstruction convention.

The script writes:

- `les_rank1_reference.mat`: reconstructed LES reference and its root-mean-square (RMS) pressure field.
- `pressure_realization_01.mat`, etc.: reconstructed pressure for each realization when `save_all_pressure = true`.
- `ensemble_rms_summary.mat`: ensemble RMS fields, reference fields, comparison metrics, and time-selection metadata.

Pressure arrays retain all 5312 samples. The saved `valid_time` mask excludes 64 samples at each end when computing statistics, leaving 5184 interior samples.

### 5. Evaluate the results

Return to the notebook's evaluation cells and point them to the matching coefficient exports and MATLAB reconstruction directory. These compare amplitude distributions, power spectral densities (PSDs), phase, temporal correlations, RMS pressure fields, and spatial wavenumber spectra. `generate_metrics.py` provides additional coefficient-ensemble plots after its input filenames are updated.

The main study reports close agreement in spectral shape and temporal coherence, with a **+28.58% coefficient mean-square bias**. Coefficient mean square, `mean(abs(z)**2)`, and reconstructed pressure mean square are different quantities and are evaluated separately.

Full-sequence ensemble comparisons include the initial observed samples and time intervals used in training and validation. The SPOD basis is computed from the available LES database, and overlapping coefficient-extraction windows can cross the later train/validation/test boundaries. These comparisons assess generation within the retained basis and should be distinguished from test-interval-only coefficient diagnostics.

## SPOD attribution and license

The MATLAB SPOD implementation originates from [SpectralPOD/spod_matlab](https://github.com/SpectralPOD/spod_matlab), with modifications and additional scripts for this project. Its [README](spod_matlab/README.md) provides the original documentation and references. The SPOD formulation and frequency–time coefficient method are described in [2] and [3].

The bundled SPOD code retains its [Caltech license](spod_matlab/LICENSE.txt), including its academic/non-commercial use conditions and redistribution requirements. See that file for the complete terms. A separate license for the original ROM code has not yet been specified.

## References

1. V. Gryazev et al., “Predictive modelling of the dominant axisymmetric pressure mode of a high-speed turbulent jet flow using spectral proper orthogonal decomposition and machine learning,” *Physics of Fluids* **38**, 095126 (2026). [DOI: 10.1063/5.0341107](https://doi.org/10.1063/5.0341107).
2. A. Towne, O. T. Schmidt, and T. Colonius, “Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis,” *Journal of Fluid Mechanics* **847**, 821–867 (2018). [Preprint](https://arxiv.org/abs/1708.04393).
3. A. Nekkanti and O. T. Schmidt, “Frequency–time analysis, low-rank reconstruction and denoising of turbulent flows using SPOD,” *Journal of Fluid Mechanics* **926**, A26 (2021). [DOI: 10.1017/jfm.2021.681](https://doi.org/10.1017/jfm.2021.681).

## Contact

Pradip Pokhrel — [pokhrel.pradip163@gmail.com](mailto:pokhrel.pradip163@gmail.com).

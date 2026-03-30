# CASCADE

**CWT-Adam Spectral Curve-fitting And Decomposition Engine**

CASCADE is a Python/PyTorch library for decomposing Raman (and BCARS) spectra into pseudo-Voigt peak components. It uses continuous wavelet transforms (CWT) for robust peak detection and GPU-accelerated Adam optimisation for fitting.

---

## Overview

The core workflow is:

1. **Detect peaks** — compute a multi-scale CWT response and identify peak candidates via cross-scale derivative voting
2. **Build initial guesses** — extract peak positions, widths, and amplitudes from the wavelet response
3. **Fit** — optimise pseudo-Voigt parameters (amplitude, centre, σ, γ) with bounded Adam
4. **Evaluate** — compute precision/recall/F1 and shape RMSE against ground truth

CASCADE was originally implemented in JAX; `tidytorch_utils.py` is a drop-in PyTorch rewrite that supports `torch.compile` and runs on CUDA, MPS, or CPU.

---

## Repository Structure

```
CASCADE/
├── BCARSFitting.ipynb      # Main notebook: load data, fit a single pixel, batch-fit an ROI
├── dataset_utils.py        # Data I/O, synthetic Raman data generation, wavelet transforms
├── tidytorch_utils.py      # PyTorch fitting engine (peak detection, optimisation, evaluation)
├── plot_utils.py           # Visualisation utilities (fit breakdown, sweep plots, RMSE analysis)
├── LICENSE                 # BSD 3-Clause
└── README.md
```

---

## Dependencies

### Required

```
numpy
torch          # >= 2.0 recommended for torch.compile
scipy          # scipy.special.wofz (Voigt profiles), scipy.optimize.linear_sum_assignment
h5py
lazy5          # HDF5 convenience wrapper used by load_h5_file / save_h5_file
matplotlib
pandas
```

Install with:

```bash
pip install numpy torch scipy h5py lazy5 matplotlib pandas
```
---

## Quick Start

### Fit a single synthetic spectrum

```python
import numpy as np
import torch
from dataset_utils import RamanDataset, SampleWrapper
from tidytorch_utils import init_sweep_context, _fit_one, _cpu
import plot_utils

# Spectral axis and wavelet width priors
x      = np.linspace(300, 1800, 512, dtype=np.float32)
widths = np.linspace(3, 60, 20, dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmas = torch.as_tensor(widths, dtype=torch.float32)
gammas = torch.tensor([5.0], dtype=torch.float32)

# Pre-cache GPU tensors (do this once per session)
init_sweep_context(x, sigmas, gammas, device, widths)
plot_utils.init_plot_context(x, widths)

# Generate a synthetic sample and fit it
ds     = RamanDataset(x=x, widths=widths, n_samples=1, n_peaks=(3, 6), seed=0)
sample = SampleWrapper(ds[0])

stats, params, spec_denoised = _fit_one(sample)
print(f"F1={stats['f1']:.3f}  Precision={stats['precision']:.3f}  Recall={stats['recall']:.3f}")
```

### Fit real BCARS data

See `BCARSFitting.ipynb` for a full walkthrough:

1. Load an HDF5 file with `load_h5_file`
2. Crop to the fingerprint region and denoise with `denoise_spectrum`
3. Fit a single pixel interactively with `process_conv_deriv_fit`
4. Batch-fit an ROI using `init_sweep_context` + the GPU kernels `_denoise_batch`, `_lor4_transform_batch`, `_fit_batch_adam`
5. Save results with `save_h5_file`

---

## Module Reference

### `dataset_utils.py`

**Data I/O**
- `load_h5_file(filename, ...)` — load BCARS hyperspectral data from HDF5; auto-detects data/NRB/dark frame paths
- `save_h5_file(filename, SAVE_FOLDER, ...)` — write fitted results back to HDF5 via `lazy5`

**Synthetic data generation**
- `voigt_peak(x, center, amp, sigma, gamma, Real=False)` — absorptive or dispersive Voigt profile; requires `scipy.special.wofz`
- `generate_multipeak_Raman(x, centers, amps, sigmas, gammas, ...)` — sum of Voigt peaks with optional noise
- `RamanDataset` — PyTorch `Dataset` that generates synthetic Raman spectra on the fly with ground-truth peak parameters; supports curriculum learning, optional prior maps, and optional neural-network pipeline estimates
- `SampleWrapper` — thin tuple subclass providing named attribute access (`.spectrum`, `.centers`, `.amplitudes`, etc.) to `RamanDataset` samples
- `dataset_from_curriculum_stage(...)` — helper that builds a `RamanDataset` from a curriculum config dict
- `raman_collate_fn(batch)` — `DataLoader` collation function that zero-pads variable-length peak lists

**Wavelet transforms**
- `lorentz4_wavelet(a, dx, support_factor)` — single Lorentz4 wavelet kernel (NumPy)
- `mexican_hat_wavelet(a, dx, support_factor)` — single Mexican Hat wavelet kernel (NumPy)
- `dispersive_lorentzian_wavelet(a, dx, support_factor)` — dispersive Lorentzian wavelet kernel (NumPy)
- `multiscale_lorentz4_transform(x, signal, widths)` — CWT with Lorentz4 wavelets (NumPy, used inside `RamanDataset`)
- `multiscale_mexhat_transform(x, signal, widths)` — CWT with Mexican Hat wavelets (NumPy)
- `cwt_dispersive_lorentzian(signal, x, scales, ...)` — CWT with dispersive Lorentzian wavelets, with optional cone-of-influence masking
- `multiscale_anisotropic_target(x, scales, x0, s0, ...)` — generate a 2-D (scale × position) training target map for a single peak

### `tidytorch_utils.py`

**Optimiser & loss**
- `pseudo_voigt(x, sigma, gamma)` — Thompson-Cox-Hastings pseudo-Voigt; JIT-compiled
- `compute_model(p, x)` — vectorised sum of pseudo-Voigt peaks; JIT-compiled
- `fit_with_bounded_adam(y, x, p0_stack, ...)` — Adam with warmup + cosine LR decay and per-step bounds projection (single spectrum)

**Wavelet bank construction**
- `precompute_wavelets(sigmas, gammas, x)` — build a `(n_sigmas, n_gammas, n_pts)` pseudo-Voigt wavelet bank
- `precompute_lorentz4_wavelets(widths, x)` — build a `(n_widths, n_pts)` Lorentz4 wavelet bank
- `voigt_multiscale_transform(signal, ...)` — FFT cross-correlation against the Voigt wavelet bank
- `lorentz4_multiscale_transform(signal, ...)` — FFT cross-correlation against the Lor4 wavelet bank

**Peak detection**
- `find_peaks_derivative_mask(x, y, ...)` — sign-change + concavity peak detection (single signal)
- `find_peaks_derivative_mask_batch(x, y_batch, ...)` — vectorised version for `(n_scales, n_pts)`
- `build_initial_guesses_from_derivative_mask(...)` — convert wavelet response + peak mask into a flat `(max_peaks * 4,)` initial-guess vector with optional min-spacing enforcement and fine-scale bias

**Per-spectrum fitting**
- `process_pixel_fit(spectrum, x, ...)` — full detect → initialise → fit pipeline for one spectrum
- `process_conv_deriv_fit(spectrum, x, ...)` — like `process_pixel_fit` but uses cross-scale voting on the wavelet responses rather than the raw spectrum for peak detection

**Batch / cube fitting**
- `process_in_batches_adam(y_batch, x, ...)` — sequential GPU fitting of a `(height, width, n_wn)` hyperspectral cube
- `init_sweep_context(x, sigmas, gammas, device, widths, ...)` — pre-cache GPU tensors (wavelet bank, denoise kernel, bounds) for repeated fitting; **must be called before** `_fit_one`, `_run_sweep`, or `_plot_sweep`
- `_denoise_batch(spectra)` — batched FFT-based Gaussian/Voigt smoothing
- `_lor4_transform_batch(spectra)` — batched Lor4 CWT
- `_fit_batch_adam(spectra, x, p0_batch, grad_mask, ...)` — batched Adam optimisation for a whole block of spectra simultaneously
- `_run_sweep(param_name, param_values, ...)` — fully vectorised parameter sweep (noise or separability) with metrics
- `_fit_one(sample_wrapper, ...)` — fit a single `SampleWrapper` sample using the cached sweep context; returns `(stats, params, denoised_spectrum)`

**Evaluation**
- `_match_peaks(gt_centers, ..., rec_params_flat, ...)` — Hungarian-algorithm peak matching with precision/recall/F1 and shape RMSE
- `denoise_spectrum(signal, x, sigma, gamma)` — standalone denoising for use before `process_conv_deriv_fit`
- `prune_peaks(params, amp_threshold)` — zero out sub-threshold peaks
- `deduplicate_peaks(params, min_spacing)` — remove peaks that are closer than `min_spacing` to a stronger neighbour

### `plot_utils.py`

- `init_plot_context(x, widths)` — cache `x` and `widths`; **must be called before** `_plot_sweep`
- `plot_voigt_fit_res(x, y_true, params, ...)` — decomposition plot with residual panel
- `plot_voigt_fit_compare_detailed(x, y_true, gt_params, rec_params, ...)` — GT vs. recovered comparison with greedy peak matching, colour-coded matched/missing/extra peaks
- `plot_shape_rmse(x, y_true, gt_params, rec_params, ...)` — per-pair shape RMSE visualisation (spectrum overview + sorted bar chart)
- `_plot_sweep(param_name, param_values, all_results, ...)` — journal-ready 3×2 figure summarising a parameter sweep; requires `init_plot_context` and `init_sweep_context`

---

## Key Concepts

### Pseudo-Voigt line shape

Peaks are modelled as Thompson-Cox-Hastings pseudo-Voigt profiles — a weighted blend of Gaussian (width σ) and Lorentzian (width γ). The mixing ratio η is determined analytically from σ and γ:

```
f(x) = η · L(x; γ) + (1-η) · G(x; σ)
```

Parameters per peak: `[amplitude, centre, sigma, gamma]` → flat vector of length `n_peaks × 4`.

### Multi-scale wavelet transform

The CWT cross-correlates the spectrum against a bank of wavelets at different scales. Three wavelet families are available:

- **Lorentz4** — 4th-order Lorentzian derivative; default for fitting
- **Mexican Hat** — 2nd derivative of Gaussian
- **Dispersive Lorentzian** — imaginary part of Voigt; sensitive to dispersive (real part of CARS) signal

### Cross-scale voting peak detection

Instead of differentiating the noisy raw spectrum, `process_conv_deriv_fit` applies derivative peak detection independently to each wavelet scale's response. A candidate position is retained only when at least `min_scale_votes` scales flag it as a local maximum. This dramatically reduces false detections from noise.

### Bounded Adam

`fit_with_bounded_adam` uses the Adam optimiser with:
- 100-step linear learning-rate warmup
- Cosine decay from `peak_lr=1e-2` to `end_lr=1e-5`
- Per-step gradient clipping (L2 norm ≤ 1)
- Hard bounds projection after every step: amplitude ∈ [0, 1], centre ∈ [0, 4000], σ/γ ∈ [0.05, 50]

---

## Notes

- `RamanSample` (a `NamedTuple` defined in `dataset_utils.py`) documents the field names but is not used at runtime. `RamanDataset.__getitem__` returns plain tuples; use `SampleWrapper` for named access.
- `init_sweep_context` must be called once before `_fit_one`, `_run_sweep`, or `_plot_sweep`. It caches the wavelet bank and denoise kernel on the target device. Similarly, `init_plot_context` must be called before `_plot_sweep`.
- GPU memory: for large cubes, process in chunks (see `BCARSFitting.ipynb` for chunked batch fitting).
- The notebook uses `importlib.reload` on all three modules, which is useful during development but can be removed for production use.

---

## License

BSD 3-Clause — Copyright 2026, Abigail Bonde. See `LICENSE`.

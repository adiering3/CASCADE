## Setup variables for general use
File_Path="/storage/home/hcoda1/2/ebaker63/r-mcicerone3-0/Abby data/preprocessed_medfilter_replaced_myo3_polyme_35mWSC45mWProber_202602090655_PROCESS_202632_11_57_2_850702.h5"
file = "replaced_myo3_p0lyme"

## wavenumber axis cropping parameters — adjust these to match the spectral range of your data (can be set after image loading to extract from dataset attributes)
n_pix = 2304 # total number of pixels in spectral axis (before cropping)
crop_start = 550 # index to start cropping spectral axis (inclusive)
crop_end = 1250 # index to end cropping spectral axis (exclusive)
BATCH_SIZE = 10000 # number of spectra to process in each batch, depends on memory constraints of your GPU — adjust as needed for larger/smaller cubes or different hardware
## spatial axis cropping parameters
x_start = 175 # index to start cropping spatial axis 0 (inclusive)
x_end = 375 # index to end cropping spatial axis 0 (exclusive)
y_start = 0 # index to start cropping spatial axis 1 (inclusive)
y_end = 330 # index to end cropping spatial axis 1 (exclusive)

# Fit controls
MAX_ITER_BATCH = 20000 # maximum iterations for Adam optimizer in each batch fit
TOL_BATCH = 1 # tolerance for convergence in each batch fit (lower = more similar to input data but longer runtime)
AMP_THR_BATCH = 1e-4 # amplitude threshold for peak detection in final pruning step (in same units as input spectra)
MIN_SPACING_BATCH = 5.0 # minimum spacing between peaks in wavenumber units (used in both initial guess pruning and final deduplication)
MAX_PEAKS_BATCH = 200 # maximum number of peaks to fit per spectrum (after initial guess pruning)
RESPONSE_THRESHOLD = 1e-5 # threshold for peak response in the initial voting step
MIN_SCALE_VOTES = 5 # minimum number of scales that must vote for a peak to be considered in the initial guess step (lower = more peaks but longer runtime)

# Progress checkpoint cadence (in spectra/pixels)
CHECKPOINT_EVERY = 5000 # cadence for progress updates during initial guess construction (lower = more frequent updates but slightly longer runtime due to overhead)

# Progress checkpoint cadence for optimizer iterations
FIT_CHECKPOINT_EVERY = 250 # cadence for progress updates during Adam fitting (lower = more frequent updates but slightly longer runtime due to overhead)

# Aggressive-start controls
AGGR_STEPS = 80 # number of initial steps to use aggressive optimization parameters (higher = more aggressive fitting at start, which can help escape local minima but may cause instability if too high)
AGGR_LR_MULT = 4.0 # learning rate multiplier during aggressive steps (higher = more aggressive updates, which can help escape local minima but may cause instability if too high)
AGGR_CLIP = 3.0 # gradient clipping norm during aggressive steps (lower = more conservative updates, which can improve stability but may slow down convergence if too low)
AGGR_BETA1 = 0.85 # beta1 parameter for Adam during aggressive steps (lower = more momentum, which can help escape local minima but may cause instability if too low)
# Core
import importlib
import time
t_start=time.perf_counter()
import tidytorch_utils as ttu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import torch
import h5py
import lazy5
from lazy5.inspect import get_datasets, get_attrs_dset
from datetime import datetime



# Common imports
from dataset_utils import voigt_peak

from plot_utils import (
    plot_voigt_fit_res,
    init_plot_context
)

from tidytorch_utils import (    
    pseudo_voigt,
    compute_wavelet_peak,
    precompute_lorentz4_wavelets,
    # sweep functions — call init_sweep_context() + init_plot_context() before using
)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

import os
if torch.cuda.is_available():
    device_name=torch.cuda.get_device_name(1)
else:
    device_name="cpu"
print("device:", device_name)

def load_h5_file(filename, galvo=True, data_path=None, nrb_path=None, dark_path=None):
    """Load BCARS hyperspectral data from an HDF5 file.

    Parameters
    ----------
    filename  : path to the .h5 file
    galvo     : unused flag kept for API compatibility
    data_path : explicit dataset path for data. If None, tries known paths then
                auto-detects the first 3-D dataset in the file.
    nrb_path  : explicit dataset path for NRB. If None, tries known paths then
                falls back to a vector of ones.
    dark_path : explicit dataset path for dark frame. If None, tries known paths
                then returns None if not found.

    Returns
    -------
    data, nrb, dark, attrs
      dark is None if no dark frame was found.
    """
    _DATA_CANDIDATES = [
        '/raw_data/hyperspectral_image_0000',
        '/preprocessed_images/medfilter_raw',
        '/preprocessed_images/medfilter_ratio_SVD_KK_PhaseErrorCorrectALS_ScaleErrorCorrectSG',
        '/preprocessed_images/ratio_SVD_KK_PhaseErrorCorrectALS',
        '/preprocessed_images/medfilter_ratio',
        '/preprocessed_images/ratio',
    ]
    _NRB_CANDIDATES = [
        '/preprocessed_images/medfilter_nrb_for_ratio',
        '/preprocessed_images/nrb',
        '/nrb',
    ]
    _DARK_CANDIDATES = [
        '/raw_data/dark_image_pre',
        '/raw_data/dark_image_post',
        '/raw_data/dark',
        '/dark',
    ]

    data, nrb, dark, attrs = None, None, None, None
    found_data_path = None

    with h5py.File(filename, "r") as f:
        all_datasets = get_datasets(f)
        print("Available datasets:", all_datasets)

        # ── Data ──────────────────────────────────────────────────────────────
        if data_path is not None:
            if data_path not in f:
                raise KeyError(
                    f"data_path '{data_path}' not found.\n"
                    f"Available datasets: {all_datasets}"
                )
            found_data_path = data_path
        else:
            for path in _DATA_CANDIDATES:
                if path in f:
                    found_data_path = path
                    break

            if found_data_path is None:
                # Auto-detect: pick the first dataset with 3+ dimensions
                for path in all_datasets:
                    if path in f and len(f[path].shape) >= 3:
                        found_data_path = path
                        print(f"Auto-detected data at: {path} — pass data_path='{path}' to suppress this message")
                        break

            if found_data_path is None:
                raise KeyError(
                    "Could not find a suitable data dataset.\n"
                    f"Available datasets: {all_datasets}\n"
                    "Pass data_path= to specify one explicitly."
                )

        data = np.array(f[found_data_path])
        print(f"Data   loaded from: {found_data_path}  shape: {data.shape}")

        # ── NRB ───────────────────────────────────────────────────────────────
        found_nrb_path = nrb_path
        if found_nrb_path is not None:
            if found_nrb_path not in f:
                raise KeyError(
                    f"nrb_path '{found_nrb_path}' not found.\n"
                    f"Available datasets: {all_datasets}"
                )
        else:
            for path in _NRB_CANDIDATES:
                if path in f:
                    found_nrb_path = path
                    break

        if found_nrb_path is not None:
            nrb = np.array(f[found_nrb_path])
            print(f"NRB    loaded from: {found_nrb_path}  shape: {nrb.shape}")
        else:
            nrb = np.ones(data.shape[-1])
            print(f"NRB not found — using ones  shape: {nrb.shape}")

        # ── Dark ──────────────────────────────────────────────────────────────
        found_dark_path = dark_path
        if found_dark_path is not None:
            if found_dark_path not in f:
                raise KeyError(
                    f"dark_path '{found_dark_path}' not found.\n"
                    f"Available datasets: {all_datasets}"
                )
        else:
            for path in _DARK_CANDIDATES:
                if path in f:
                    found_dark_path = path
                    break

        if found_dark_path is not None:
            dark = np.array(f[found_dark_path])
            print(f"Dark   loaded from: {found_dark_path}  shape: {dark.shape}")
        else:
            print("Dark not found — dark=None")

        # ── Attributes ────────────────────────────────────────────────────────
        attrs = get_attrs_dset(filename, found_data_path)
        print(f"Attrs  loaded from: {found_data_path}")

    return data, nrb, dark, attrs


def save_h5_file(filename, data, nrb, dark, attrs, model, peak_params, SAVE_FOLDER):
    # save original data
    lazy5.create.save(file=f'fitted_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/raw',         data=np.array(data,       dtype=np.uint16), mode='w')
    lazy5.create.save(file=f'fitted_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/nrb',         data=np.array(nrb,        dtype=np.uint16), mode='a')
    if dark is not None:
        lazy5.create.save(file=f'fitted_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/dark',    data=np.array(dark,       dtype=np.uint16), mode='a')
    # save fitted model and peak params
    lazy5.create.save(file=f'fitted_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/model',       data=np.array(model,      dtype=np.uint16), mode='a')
    lazy5.create.save(file=f'fitted_{filename}', pth=SAVE_FOLDER, dset='preprocessed_images/peak_params', data=np.array(peak_params, dtype=np.uint16), mode='a')

    fid = os.path.join(SAVE_FOLDER, f'fitted_{filename}')
    for dset in ['preprocessed_images/raw', 'preprocessed_images/nrb', 'preprocessed_images/model', 'preprocessed_images/peak_params']:
        lazy5.alter.write_attr_dict(dset=dset, attr_dict=attrs, fid=fid)
    if dark is not None:
        lazy5.alter.write_attr_dict(dset='preprocessed_images/dark', attr_dict=attrs, fid=fid)
now = datetime.now().strftime("%m%d%Y_%H:%M:%S")
RAW_PATH = File_Path
raw_data, nrb, dark, attrs = load_h5_file(RAW_PATH)
# data, nrb, dark, attrs = load_h5_file("/home/adiering3/projects/Hvetch_ScaleMap/preprocessed_medfilter_hvetch_nodule_3a_05152025_02_PROCESS_20251210_13_21_4_384624.h5")
pre_t0=time.perf_counter()
filename = os.path.join(f"{file}_{now}")
coeffs = attrs['Calib.a_vec']
attrs['Calib.n_pix']= n_pix

ctr = attrs['Calib.ctr_wl0']
probe = attrs['Calib.probe']
probe *= 1e-7
converted_nm = np.polyval(coeffs, np.arange(n_pix))
converted_nm *= 0.0000001
wn = 1/converted_nm - 1/probe
crop_wn= wn[crop_start:crop_end] ## Crop the wavenumber axis to match the data
wavenumbers= crop_wn

data = raw_data[:, :, crop_start:crop_end]



x = wavenumbers.copy()
x = x.astype(np.float32)

widths = np.linspace(1, 10, 100)

sigmas = torch.as_tensor(widths, dtype=torch.float32)
gammas = torch.tensor([5.0], dtype=torch.float32)
x_gpu  = torch.as_tensor(x, dtype=torch.float32)

x_c  = x_gpu - x_gpu.mean()

sigs = sigmas.view(-1, 1, 1)
gams = gammas.view( 1,-1, 1)
x_bc = x_c.view(   1, 1,-1)

profiles = pseudo_voigt(x_bc, sigs, gams)
max_vals = profiles.amax(dim=-1, keepdim=True)
wavelet_peaks_cached = profiles / (max_vals + 1e-12) 
wavelet_peaks_cached = wavelet_peaks_cached.to(device)

# ── Precomputed constants (shared across all sweep calls) ─────────────────────
_x_dev     = x_gpu.to(device)
_lor4_bank = precompute_lorentz4_wavelets(sigmas.to(device), _x_dev)  # (n_widths, n_pts)

_sigma_t = torch.tensor(3.0,  dtype=torch.float32, device=device)
_gamma_t = torch.tensor(1e-6, dtype=torch.float32, device=device)
_kernel  = compute_wavelet_peak(_sigma_t, _gamma_t, _x_dev)
_kernel  = _kernel / _kernel.sum()
_ker_fft = torch.fft.fft(torch.fft.ifftshift(_kernel))  # (n_pts,)

# Param bounds tensors (reused each step)
_LO = torch.tensor([0.0,    0.0, 0.05, 0.05], device=device)
_HI = torch.tensor([1.0, 4000.0, 50.0, 50.0], device=device)


ttu.init_sweep_context(x_gpu, sigmas, gammas, device, widths)
init_plot_context(x, widths)

# Select a region from your already-loaded cube
# You can change these bounds to benchmark/fit a different area
# ── Batch fit directly on loaded BCARS cube (no RamanDataset) ───────────────────
with torch.no_grad():
    torch.cuda.empty_cache()

# Make sure sweep context exists for tidytorch_utils internals
ttu.init_sweep_context(x_gpu, sigmas, gammas, device, widths)
init_plot_context(x, widths)

# Select a region from your already-loaded cube
# You can change these bounds to benchmark/fit a different area

data_crop=data[y_start:y_end, x_start:x_end, :]
pixel_count=data_crop.shape[0]*data_crop.shape[1]

print(time.perf_counter()-t_start)
with open(f"Full_ROI_Fit_params_{file}.txt", "a") as f:
    t_batch=[]
    tbp=[]
    t_fit=[]
    tfp=[]
    p_fit=[]
    processed_gpu=torch.as_tensor(np.zeros_like(data_crop), dtype=torch.float32)
    X0=0
    Y0=0
    X1=np.sqrt(BATCH_SIZE).astype(int)
    Y1=np.sqrt(BATCH_SIZE).astype(int)
    process_count=0
    t_init=time.perf_counter()
    while process_count<=pixel_count:


        # X0, X1 = 160, 165
        # Y0, Y1 = 295, 300

        # Build (B, n_pts) batch from existing data variable
        cube_roi = np.asarray(data_crop[X0:X1, Y0:Y1, :])
        if np.iscomplexobj(cube_roi):
            cube_roi = np.imag(cube_roi)
        spectra_np = cube_roi.reshape(-1, cube_roi.shape[-1]).astype(np.float32)
        B = spectra_np.shape[0]
        print(f"ROI shape: {cube_roi.shape}  ->  batch spectra: {B} × {spectra_np.shape[1]}\n")

        spectra_gpu = torch.as_tensor(spectra_np, dtype=torch.float32, device=ttu._ctx_device)



        def _sync_dev(dev):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elif dev.type == "mps":
                torch.mps.synchronize()

        # Warmup (optional, untimed)
        with torch.no_grad():
            _ = ttu._denoise_batch(spectra_gpu[:1])

        _sync_dev(ttu._ctx_device)
        t0 = time.perf_counter()

        # 1) Denoise + transform + vote
        spec_d = ttu._denoise_batch(spectra_gpu)
        resp = ttu._lor4_transform_batch(spec_d)
        _, n_widths, n_pts = resp.shape

        flat_masks = ttu.find_peaks_derivative_mask_batch(
            ttu._x_dev, resp.reshape(B * n_widths, n_pts), min_height=0.0
        )
        masks = flat_masks.reshape(B, n_widths, n_pts)
        vote_count = masks.long().sum(dim=1)
        max_resp = resp.max(dim=1).values
        peak_masks = (vote_count >= MIN_SCALE_VOTES) & (max_resp > RESPONSE_THRESHOLD)

        # 2) Build padded p0 + gradient mask
        npq_max = MAX_PEAKS_BATCH * 4
        p0_flat_list = []
        n_real_peaks = []

        t_init0 = time.perf_counter()
        for i in range(B):
            p0 = ttu.build_initial_guesses_from_derivative_mask(
                resp[i].unsqueeze(1),
                ttu._ctx_sigmas.to(ttu._ctx_device),
                ttu._ctx_gammas[:1].to(ttu._ctx_device),
                ttu._x_dev,
                spec_d[i],
                peak_masks[i],
                max_peaks=MAX_PEAKS_BATCH,
                min_spacing=MIN_SPACING_BATCH,
                scale_preference_fraction=0.008,
            )
            p0_flat = p0[:npq_max]
            p0_flat_list.append(p0_flat)
            n_peaks_i = int((p0_flat[0::4] > 0).sum().item())
            n_real_peaks.append(min(n_peaks_i, MAX_PEAKS_BATCH))

            done = i + 1
            if (done % CHECKPOINT_EVERY == 0) or (done == B):
                elapsed = time.perf_counter() - t_init0
                px_per_sec = done / max(elapsed, 1e-9)
                eta = (B - done) / max(px_per_sec, 1e-9)
                print(f"[checkpoint] initialized {done}/{B} spectra\n")
                print(f"({100.0 * done / B:.1f}%) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s\n"
                )

        max_real = max(1, max(n_real_peaks) if n_real_peaks else 1)
        npq = max_real * 4
        p0_batch = torch.zeros(B, npq, device=ttu._ctx_device)
        grad_mask = torch.zeros(B, npq, dtype=torch.float32, device=ttu._ctx_device)

        for i, (p0_flat, n_peaks_i) in enumerate(zip(p0_flat_list, n_real_peaks)):
            n_take = min(p0_flat.numel(), npq)
            p0_batch[i, :n_take] = p0_flat[:n_take]
            grad_mask[i, :n_peaks_i * 4] = 1.0

        # 3) Batched Adam fit
        _sync_dev(ttu._ctx_device)
        t_fit0 = time.perf_counter()
        params_batch = ttu._fit_batch_adam(
            spec_d, ttu._x_dev, p0_batch, grad_mask,
            max_iter=MAX_ITER_BATCH,
            tol=TOL_BATCH,
            aggressive_start_steps=AGGR_STEPS,
            aggressive_lr_mult=AGGR_LR_MULT,
            aggressive_clip_norm=AGGR_CLIP,
            aggressive_beta1=AGGR_BETA1,
            progress_every=FIT_CHECKPOINT_EVERY,
            progress_prefix="fit",
        )
        _sync_dev(ttu._ctx_device)
        fit_only_sec = time.perf_counter() - t_fit0

        # 4) Post-process fitted peaks
        params_batch = ttu._prune_peaks_batch(params_batch, amp_threshold=AMP_THR_BATCH)
        params_batch = ttu._deduplicate_peaks_batch(params_batch, MIN_SPACING_BATCH)

        _sync_dev(ttu._ctx_device)
        total_sec = time.perf_counter() - t0

        n_fit_peaks = (params_batch.reshape(B, -1, 4)[:, :, 0] > AMP_THR_BATCH).sum(dim=1)
        print(f"\nBatch fit summary (loaded BCARS data):\n")
        print(f"  total time: {total_sec:.3f}s  -> {total_sec/B:.4f}s per pixel\n")
        if BATCH_SIZE == cube_roi.shape[0]*cube_roi.shape[1]:
            t_batch.append(total_sec)
        tbp.append(total_sec/B)
        print(f"  fit-only : {fit_only_sec:.3f}s  -> {fit_only_sec/B:.4f}s per pixel\n")
        if BATCH_SIZE == cube_roi.shape[0]*cube_roi.shape[1]:
            t_fit.append(fit_only_sec)
        tfp.append(fit_only_sec/B)
        print(f"  fitted peaks/pixel: mean={n_fit_peaks.float().mean().item():.1f}  median={n_fit_peaks.float().median().item():.1f}\n")
        p_fit.append(n_fit_peaks.float().mean().item())

        # ── Reshape batched params back to ROI-aligned cube and save ─────────────────────
        params_np = params_batch.detach().cpu().numpy().astype(np.float32)   # (B, n_params)
        roi_h = int(X1 - X0)
        roi_w = int(Y1 - Y0)

        if params_np.shape[0] != roi_h * roi_w:
            raise ValueError(
                f"Batch size mismatch: got {params_np.shape[0]} spectra, expected {roi_h*roi_w} from ROI shape."
            )

        # Parameter cube aligned to ROI coordinates: [x, y, param]
        peak_params_roi = params_np.reshape(roi_h, roi_w, -1)
        print(f"peak_params_roi shape: {peak_params_roi.shape}  (x={roi_h}, y={roi_w}, params={peak_params_roi.shape[-1]})\n")

        # Optional fitted model cube for ROI (same spatial shape as ROI, spectral axis = len(x))
        with torch.no_grad():
            model_batch = ttu._compute_model_batch(params_batch, ttu._x_dev).detach().cpu().numpy().astype(np.float32)
        model_roi = model_batch.reshape(roi_h, roi_w, -1)
        print(f"model_roi shape: {model_roi.shape}\n")
        processed_gpu[X0:X1, Y0:Y1, :] = torch.from_numpy(model_roi)
        process_count+=(X1-X0)*(Y1-Y0)
        print(f"Processed pixels: {process_count}\n")
        if X1+np.sqrt(BATCH_SIZE).astype(int) >= data_crop.shape[0]:
            if X1<data_crop.shape[0]:
                X0=X1
                X1=data_crop.shape[0]
            else:
                X0=0
                X1=np.sqrt(BATCH_SIZE).astype(int)
                if Y1+np.sqrt(BATCH_SIZE).astype(int) <= data_crop.shape[1]:
                    if Y1==data_crop.shape[1]:
                        Y0=Y1
                        Y1=data_crop.shape[1]
                    else:
                        Y0=Y1
                        Y1+=np.sqrt(BATCH_SIZE).astype(int)
        else:
            X0=X1
            X1=X1+np.sqrt(BATCH_SIZE).astype(int)
        
        del model_batch
        del params_batch
        del p0_batch
        del flat_masks
        del masks
        del resp
        del spectra_gpu
        del spec_d
        del grad_mask
        del peak_masks
        del vote_count
        del spectra_np
        del cube_roi
        del p0_flat_list
        del n_real_peaks
        del params_np
        del peak_params_roi
        del model_roi
    t_final = time.perf_counter()
    f.write(f"\nTotal processing time for entire cube of size {data_crop.shape[0]}x{data_crop.shape[1]}x{data_crop.shape[2]} with error tolerance {TOL_BATCH}: {t_final - t_init:.3f}s\n")
    f.write(f"Average time per batch of {BATCH_SIZE} spectra: {np.mean(t_batch):.3f}s\n")
    f.write(f"Standard deviation of time per batch: {np.std(t_batch):.3f}s\n")
    f.write(f"Average time for ADAM fitting per batch: {np.mean(t_fit):.3f}s\n")
    f.write(f"Standard deviation of ADAM fitting time per batch: {np.std(t_fit):.3f}s\n")
    f.write(f"Average time per pixel: {np.mean(tbp):.4f}s\n")
    f.write(f"Standard deviation of time per pixel: {np.std(tbp):.4f}s\n")
    f.write(f"Average time for ADAM fitting per pixel: {np.mean(tfp):.4f}s\n")
    f.write(f"Standard deviation of ADAM fitting time per pixel: {np.std(tfp):.4f}s\n")
    f.write(f"Average number of fitted peaks per pixel: {np.mean(p_fit):.1f}\n")
    f.write(f"Standard deviation of fitted peaks per pixel: {np.std(p_fit):.1f}\n")
    np.savez_compressed(
        f"bcars_batched_fit_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{file}_tolerance={TOL_BATCH}.npz",
        processed_spectrum=processed_gpu.cpu().numpy(),
        unprocessed_spectrum=data_crop,
        x_axis=np.asarray(x, dtype=np.float32),
        wn_axis=np.asarray(wavenumbers, dtype=np.float32)
    )
    f.close()
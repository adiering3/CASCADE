from math import gamma

import numpy as np
import torch
import h5py
import lazy5
from lazy5.inspect import get_datasets, get_attrs_dset
from torch.utils.data import Dataset


try:
    from scipy.special import wofz
except Exception as e:
    wofz = None


import os


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


def save_h5_file(
    filename,
    SAVE_FOLDER,
    attrs=None,
    data=None,
    original=None,
    nrb=None,
    dark=None,
    model=None,
    peak_params=None,
    x_axis=None,
):
    """
    Save BCARS fit results to an HDF5 file using lazy5.

    Parameters
    ----------
    filename : str
        Base filename for the output HDF5 file.
    SAVE_FOLDER : str
        Folder where the file will be saved.
    attrs : dict, optional
        Attribute dictionary to write to saved datasets.
    data : array-like, optional
        Raw/original data cube. If None, `original` will be used.
    original : array-like, optional
        Alternative name for raw/original data, matching np.savez usage.
    nrb : array-like, optional
        NRB dataset.
    dark : array-like, optional
        Dark dataset.
    model : array-like, optional
        Fitted model data.
    peak_params : array-like, optional
        Peak parameter array.
    x_axis : array-like, optional
        Spectral axis / x-axis values.
    """

    if attrs is None:
        attrs = {}

    # Let `original` behave like the raw data input from np.savez_compressed
    raw_data = data if data is not None else original

    out_file = f"fitted_{filename}"
    fid = os.path.join(SAVE_FOLDER, out_file)

    first_write = True

    def _save_dataset(dset_name, arr, dtype=None):
        nonlocal first_write
        if arr is None:
            return

        arr_np = np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

        lazy5.create.save(
            file=out_file,
            pth=SAVE_FOLDER,
            dset=dset_name,
            data=arr_np,
            mode='w' if first_write else 'a'
        )
        first_write = False

        if attrs:
            lazy5.alter.write_attr_dict(dset=dset_name, attr_dict=attrs, fid=fid)

    # Save datasets
    _save_dataset('preprocessed_images/raw', raw_data, dtype=np.uint16)
    _save_dataset('preprocessed_images/nrb', nrb, dtype=np.uint16)
    _save_dataset('preprocessed_images/dark', dark, dtype=np.uint16)
    _save_dataset('preprocessed_images/model', model, dtype=np.float32)
    _save_dataset('preprocessed_images/peak_params', peak_params)   # keep native dtype unless you want otherwise
    _save_dataset('preprocessed_images/x_axis', x_axis, dtype=np.float32)

def voigt_peak(x, center, amp, sigma, gamma=5.0, Real=False):
    if wofz is None:
        raise ImportError("scipy.special.wofz is required for voigt_peak")

    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    w = wofz(z)

    if Real:
        # Dispersive (Hilbert-transform-like) profile.
        # Normalize so `amp` corresponds to the peak absolute amplitude (|y| max).
        profile = np.imag(w) / (sigma * np.sqrt(2 * np.pi))
        peak_ref = float(np.max(np.abs(profile)))
        return amp * profile / (peak_ref + 1e-12)

    # Absorptive Voigt profile.
    # Normalize so `amp` corresponds to the peak height (y max).
    profile = np.real(w) / (sigma * np.sqrt(2 * np.pi))
    peak_ref = float(np.max(profile))
    return amp * profile / (peak_ref + 1e-12)


def generate_multipeak_Raman(x, centers, amps, sigmas, gammas, noise_std=0.0, Real=False):
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers)
    amps = np.asarray(amps)
    sigmas = np.asarray(sigmas)
    gammas = np.asarray(gammas) if np.ndim(gammas) > 0 else np.full_like(centers, float(gammas))

    assert len(centers) == len(amps) == len(sigmas)

    y = np.zeros_like(x)
    for c, a, s, g in zip(centers, amps, sigmas, gammas):
        y += voigt_peak(x, c, a, s, g, Real)

    if noise_std > 0:
        y += noise_std * np.random.randn(len(x))

    return y.astype(np.float32)


def lorentz4_wavelet(a, dx=1.0, support_factor=8):
    half_len = int(np.ceil(support_factor * a / dx))
    size = 2 * half_len + 1
    x = np.arange(-half_len, half_len + 1) * dx / a
    psi = (1 - 6*x**2 + x**4) / (1 + x**2)**4
    psi /= np.sqrt(np.sum(psi**2) * dx + 1e-12)
    return psi


def mexican_hat_wavelet(a, dx=1.0, support_factor=8):
    half_len = int(np.ceil(support_factor * a / dx))
    x = np.arange(-half_len, half_len + 1) * dx / max(a, 1e-12)
    psi = (1.0 - x**2) * np.exp(-0.5 * x**2)
    psi = psi - np.mean(psi)
    psi /= np.sqrt(np.sum(psi**2) * dx + 1e-12)
    return psi


def multiscale_lorentz4_transform(x, signal, widths, support_factor=8):
    dx = x[1] - x[0]
    N = len(signal)
    W = np.zeros((len(widths), N), dtype=np.float32)
    for i, a in enumerate(widths):
        psi = lorentz4_wavelet(a, dx=dx, support_factor=support_factor)
        conv = np.convolve(signal, psi, mode="same")
        if conv.shape[0] != N:
            raise RuntimeError(f"Wavelet convolution length mismatch: {conv.shape[0]} vs {N}")
        W[i] = conv
    return W


def multiscale_mexhat_transform(x, signal, widths, support_factor=8):
    dx = x[1] - x[0]
    N = len(signal)
    W = np.zeros((len(widths), N), dtype=np.float32)
    for i, a in enumerate(widths):
        psi = mexican_hat_wavelet(a, dx=dx, support_factor=support_factor)
        conv = np.convolve(signal, psi, mode="same")
        if conv.shape[0] != N:
            raise RuntimeError(f"Wavelet convolution length mismatch: {conv.shape[0]} vs {N}")
        W[i] = conv
    return W


def dispersive_lorentzian_wavelet(a, dx=1.0, support_factor=8):
    """
    Finite-support dispersive Lorentzian wavelet at scale a.
    """
    half_len = int(np.ceil(support_factor * a / max(dx, 1e-12)))
    t = np.arange(-half_len, half_len + 1, dtype=np.float32) * dx / max(a, 1e-12)
    psi = t / (1 + t**2)
    # Enforce near-zero mean on finite support before L2 normalization.
    psi = psi - np.mean(psi)
    psi /= np.sqrt(np.sum(psi**2) * dx + 1e-12)
    return psi.astype(np.float32)


def cwt_dispersive_lorentzian(
    signal,
    x,
    scales,
    support_factor=8,
    pad_mode="reflect",
    mask_coi=True,
):
    """
    signal : 1D array
    x      : spectral axis (uniform spacing)
    scales : iterable of scales in same units as x

    Returns:
        W : (n_scales, len(signal)) array
    """
    signal = np.asarray(signal, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)

    dx = x[1] - x[0]
    n = len(signal)
    W = np.zeros((len(scales), n), dtype=np.float32)
    max_half_len = int(
        np.ceil(support_factor * float(np.max(scales)) / max(float(dx), 1e-12))
    )
    pad = max(0, max_half_len)
    signal_pad = np.pad(signal, (pad, pad), mode=pad_mode) if pad > 0 else signal

    for i, a in enumerate(scales):
        half_len = int(np.ceil(support_factor * float(a) / max(float(dx), 1e-12)))
        psi = dispersive_lorentzian_wavelet(a, dx=dx, support_factor=support_factor)
        conv_pad = np.convolve(signal_pad, psi, mode="same")
        if conv_pad.shape[0] != signal_pad.shape[0]:
            raise RuntimeError(
                f"Wavelet convolution length mismatch: {conv_pad.shape[0]} vs {signal_pad.shape[0]}"
            )
        conv = conv_pad[pad : pad + n] if pad > 0 else conv_pad
        # Mask the cone-of-influence region where boundary effects dominate.
        if mask_coi and half_len > 0:
            k = min(half_len, n // 2)
            conv[:k] = 0.0
            conv[n - k :] = 0.0
        W[i] = conv

    return W


def multiscale_anisotropic_target(x, scales, x0, s0, gamma_x=6.0, sigma_s=0.7, scale_spread=0):
    X, S = np.meshgrid(x, scales)
    lor_x = 1.0 / (1.0 + ((X - x0) / gamma_x) ** 2)
    s_idx = np.argmin(np.abs(scales - s0))
    T = np.zeros_like(X, dtype=np.float32)
    for ds in range(-scale_spread, scale_spread + 1):
        j = s_idx + ds
        if j < 0 or j >= len(scales):
            continue
        w = np.exp(-0.5 * (ds / sigma_s) ** 2)
        T[j] += w * lor_x[j]
    T /= T.sum() + 1e-12
    return T

from typing import NamedTuple, Optional
import torch


class RamanSample(NamedTuple):
    # --- Core outputs (original order preserved) ---
    wavelet: torch.Tensor
    target: torch.Tensor
    true_xs: torch.Tensor
    centers: torch.Tensor
    amplitudes: torch.Tensor
    gammas: torch.Tensor
    fwhm: torch.Tensor
    separability: torch.Tensor
    sigmas: torch.Tensor
    spectrum: torch.Tensor

    # --- Optional outputs ---
    wavelet_lor: Optional[torch.Tensor] = None
    wavelet_disp: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None

class RamanDataset(Dataset):
    
    def __init__(
        self,
        x,
        widths,
        n_samples=5000,
        n_peaks=1,
        amp_range=(0.01, 1.0),
        fwhm_range=(7.0, 30.0),
        separability_range=(1.0, 4.0),
        gamma=5.0,
        noise_std=0.0,
        margin=50.0,
        beta_amp=0.35,
        sep_extra=0.0,
        min_peaks_after_trunc=1,
        max_sigma=None,
        LogAmp=False,
        wavelet_repr="linear",
        log_eps=1e-6,
        target_gamma_x=None,
        target_sigma_s=None,
        target_scale_spread=None,
        wavelet="Lor4",
        MexHat=None,
        disp_lor_support_factor=8,
        disp_lor_pad_mode="reflect",
        disp_lor_mask_coi=True,
        return_both_wavelets=False,
        return_priors=False,
        ridge_prior_cfg=None,
        signedpair_prior_cfg=None,
        prior_line_sigma_px=2.0,
        return_pipeline_estimates=False,
        pipeline_checkpoint_path=None,
        pipeline_device=None,
        pipeline_prob_mode="softmax",
        pipeline_max_peaks=50,
        pipeline_min_distance=5,
        pipeline_min_prominence_rel=0.01,
        pipeline_edge_exclude=25,
        pipeline_post_cfg=None,
        pipeline_use_nondispersive_y=True,
        rng=None,
        seed=42,
        Real=False,
    ):
        self.x = np.asarray(x, dtype=np.float32)

        widths = np.asarray(widths, dtype=np.float32)
        if max_sigma is not None:
            widths = widths[widths <= max_sigma]
            if len(widths) == 0:
                raise ValueError("max_sigma removed all wavelet scales")

        self.widths = widths
        self.n = int(n_samples)

        if isinstance(n_peaks, tuple):
            self.n_peaks_min, self.n_peaks_max = map(int, n_peaks)
        else:
            self.n_peaks_min = self.n_peaks_max = int(n_peaks)

        self.amp_range = tuple(amp_range)
        self.fwhm_range = tuple(fwhm_range)
        self.S_range = tuple(separability_range)

        self.LogAmp = bool(LogAmp)
        self.wavelet_repr = str(wavelet_repr)
        self.log_eps = float(log_eps)

        self.target_gamma_x = float(target_gamma_x) if target_gamma_x is not None else 6.0
        self.target_sigma_s = float(target_sigma_s) if target_sigma_s is not None else 0.7
        self.target_scale_spread = int(target_scale_spread) if target_scale_spread is not None else 0
        # Backward compatibility: legacy MexHat bool still maps to wavelet choice.
        if MexHat is not None:
            self.wavelet = "MexHat" if bool(MexHat) else "Lor4"
        else:
            self.wavelet = str(wavelet)
        if self.wavelet not in ("Lor4", "MexHat", "DispLor"):
            raise ValueError(
                f"wavelet must be one of ('Lor4', 'MexHat', 'DispLor'), got {self.wavelet}"
            )
        self.disp_lor_support_factor = float(disp_lor_support_factor)
        self.disp_lor_pad_mode = str(disp_lor_pad_mode)
        self.disp_lor_mask_coi = bool(disp_lor_mask_coi)
        self.return_both_wavelets = bool(return_both_wavelets)
        self.return_priors = bool(return_priors)
        self.ridge_prior_cfg = ridge_prior_cfg
        self.signedpair_prior_cfg = signedpair_prior_cfg
        self.prior_line_sigma_px = float(prior_line_sigma_px)
        self.Real = bool(Real)
        self.return_pipeline_estimates = bool(return_pipeline_estimates)
        self.pipeline_checkpoint_path = pipeline_checkpoint_path
        self.pipeline_device = pipeline_device
        self.pipeline_prob_mode = str(pipeline_prob_mode)
        self.pipeline_max_peaks = int(pipeline_max_peaks)
        self.pipeline_min_distance = int(pipeline_min_distance)
        self.pipeline_min_prominence_rel = float(pipeline_min_prominence_rel)
        self.pipeline_edge_exclude = int(pipeline_edge_exclude)
        self.pipeline_post_cfg = pipeline_post_cfg
        self.pipeline_use_nondispersive_y = bool(pipeline_use_nondispersive_y)

        self._pipeline_model = None
        self._pipeline_model_is_dual = False
        self._pipeline_post_cfg_obj = None
        if self.return_pipeline_estimates:
            if not self.pipeline_checkpoint_path:
                raise ValueError("pipeline_checkpoint_path is required when return_pipeline_estimates=True")
            # Pipeline requires companion wavelets for dual/single UNet input compatibility.
            self.return_both_wavelets = True

        # LogAmp governs non-dispersive (Lor/absorptive) amplitude sampling only.
        # Dispersive spectrum amplitudes are always sampled linearly.

        if self.wavelet_repr not in ("linear", "log", "abs"):
            raise ValueError(
                f"wavelet_repr must be one of ('linear', 'log', 'abs'), got {wavelet_repr}"
            )
        # wavelet_repr controls Lor4 representation; DispLor remains signed-linear
        # when wavelet_repr='log' to preserve signed structure.

        if isinstance(gamma, tuple):
            self.gamma_range = tuple(gamma)
            self.gamma_constant = None
        else:
            self.gamma_constant = float(gamma)
            self.gamma_range = None        
        self.noise_std = float(noise_std)
        self.margin = float(margin)
        self.beta_amp = float(beta_amp)
        self.sep_extra = float(sep_extra)
        self.min_peaks_after_trunc = int(min_peaks_after_trunc)

        # ---------------------------------------
        # Deterministic dataset seed
        # ---------------------------------------
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        self.seed = int(seed)
        self.rng = None  # no persistent RNG


        self.seed = seed

        self.xmin = float(self.x.min()) + self.margin
        self.xmax = float(self.x.max()) - self.margin
        self.dx = float(np.mean(np.diff(self.x)))

        self.meta = dict(
            n_peaks=(self.n_peaks_min, self.n_peaks_max),
            amp_range=self.amp_range,
            amp_definition="peak_height" if not self.Real else "peak_abs_height",
            amp_sampling=("log" if self.LogAmp else "linear") if not self.Real else "linear",
            amp_sampling_lor="log" if self.LogAmp else "linear",
            amp_sampling_disp="linear",
            fwhm_range=self.fwhm_range,
            separability_range=self.S_range,
            noise_std=self.noise_std,
            dx=self.dx,
            beta_amp=self.beta_amp,
            width_units="FWHM (cm^-1)",
            separability_def="minimum enforced adjacent-pair separability",
            max_sigma=max_sigma,
            n_scales=len(self.widths),
            wavelet_repr=self.wavelet_repr,
            log_eps=self.log_eps if self.wavelet_repr == "log" else None,
            target_gamma_x=self.target_gamma_x,
            target_sigma_s=self.target_sigma_s,
            target_scale_spread=self.target_scale_spread,
            wavelet=self.wavelet,
            disp_lor_support_factor=self.disp_lor_support_factor,
            disp_lor_pad_mode=self.disp_lor_pad_mode,
            disp_lor_mask_coi=self.disp_lor_mask_coi,
            return_both_wavelets=self.return_both_wavelets,
            return_priors=self.return_priors,
            prior_line_sigma_px=self.prior_line_sigma_px,
            Real=self.Real,
            seed=self.seed,
            return_pipeline_estimates=self.return_pipeline_estimates,
            pipeline_checkpoint_path=self.pipeline_checkpoint_path,
            pipeline_prob_mode=self.pipeline_prob_mode,
            pipeline_max_peaks=self.pipeline_max_peaks,
            pipeline_min_distance=self.pipeline_min_distance,
            pipeline_min_prominence_rel=self.pipeline_min_prominence_rel,
            pipeline_edge_exclude=self.pipeline_edge_exclude,
            pipeline_use_nondispersive_y=self.pipeline_use_nondispersive_y,
        )

    def __len__(self):
        return self.n
    
    def _rng_for_idx(self, idx):
        """
        Create deterministic RNG tied to sample index.
        Ensures reproducibility independent of shuffling/workers.
        """
        return np.random.default_rng(self.seed + int(idx))

    def _apply_wavelet_representation(self, W):
        if self.wavelet_repr == "linear":
            return W
        if self.wavelet_repr == "abs":
            return np.abs(W).astype(np.float32)
        return np.log(np.abs(W) + self.log_eps).astype(np.float32)

    def _apply_repr_for_wavelet(self, W, wavelet_name):
        # Keep signed DispLor companion available even when primary repr uses log.
        if wavelet_name == "DispLor" and self.wavelet_repr == "log":
            return W.astype(np.float32)
        return self._apply_wavelet_representation(W)

    def _sample_n_peaks(self, rng):
        if self.n_peaks_min == self.n_peaks_max:
            return self.n_peaks_min
        return int(rng.integers(self.n_peaks_min, self.n_peaks_max + 1))


    def _required_sep(self, w1, a1, w2, a2, S):
        w_eff = max(w1, w2)
        r = max(a1, a2) / max(1e-12, min(a1, a2))
        return S * w_eff * (1.0 + self.beta_amp * np.log10(r)) + self.sep_extra

    def _sample_amplitudes(self, n_peaks):
        return self._sample_amplitudes_mode(n_peaks, log_mode=self.LogAmp)

    def _sample_amplitudes_mode(self, n_peaks, rng=None, log_mode=False):
        if rng is None:
            rng = self.rng if self.rng is not None else np.random.default_rng()
        a_min, a_max = self.amp_range
        if not bool(log_mode):
            return rng.uniform(a_min, a_max, size=n_peaks).astype(np.float32)
        if a_min <= 0:
            raise ValueError("LogAmp=True requires amp_range > 0")
        log_min = np.log10(a_min)
        log_max = np.log10(a_max)
        return (10.0 ** rng.uniform(log_min, log_max, size=n_peaks)).astype(np.float32)

    def _line_map_from_xs(self, x_idx_list, n_scales, n_x):
        if len(x_idx_list) == 0:
            return np.zeros((n_scales, n_x), dtype=np.float32)
        xx = np.arange(n_x, dtype=np.float32)[None, :]
        sigma = max(float(self.prior_line_sigma_px), 1e-6)
        m = np.zeros((n_scales, n_x), dtype=np.float32)
        for x0 in x_idx_list:
            g = np.exp(-0.5 * ((xx - float(x0)) / sigma) ** 2).astype(np.float32)
            m += g
        m /= (np.max(m) + 1e-12)
        return m

    def _compute_prior_maps(self, W_disp_linear):
        # Lazy import avoids heavy dependency when priors are unused.
        from mswt_peak_priors import (
            RidgePriorsConfig,
            SignedPairPriorsConfig,
            ridge_peak_priors_from_W,
            signed_pair_peak_priors_from_W,
        )

        ridge_cfg = self.ridge_prior_cfg if self.ridge_prior_cfg is not None else RidgePriorsConfig()
        sp_cfg = self.signedpair_prior_cfg if self.signedpair_prior_cfg is not None else SignedPairPriorsConfig()

        ridge_xs, _ = ridge_peak_priors_from_W(
            W_disp_linear,
            ridge_cfg,
            max_peaks=100,
            use_logabs=True,
        )
        center_mode = sp_cfg.center_mode
        if str(center_mode) == "auto":
            if self.Real:
                center_mode = "center_lobe"
            else:
                center_mode = "pair"
        sp_xs, _ = signed_pair_peak_priors_from_W(
            W_disp_linear,
            sp_cfg,
            max_peaks=100,
            widths=self.widths,
            x_axis=self.x,
            center_mode=center_mode,
        )

        n_scales, n_x = W_disp_linear.shape
        ridge_map = self._line_map_from_xs(ridge_xs, n_scales, n_x)
        sp_map = self._line_map_from_xs(sp_xs, n_scales, n_x)
        priors = np.stack([ridge_map, sp_map], axis=0).astype(np.float32)
        return priors

    def _pipeline_init_once(self):
        if self._pipeline_model is not None:
            return
        from model_defs import load_model_for_training, DualStemSkipUNetHeatmap
        from post_UNet_utils import ScaleAmpConfig

        if self.pipeline_device is not None:
            dev = torch.device(self.pipeline_device)
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        self._pipeline_device_resolved = dev
        m = load_model_for_training(self.pipeline_checkpoint_path, dev)
        m.eval()
        self._pipeline_model = m
        self._pipeline_model_is_dual = isinstance(m, DualStemSkipUNetHeatmap)
        if self.pipeline_post_cfg is None:
            self._pipeline_post_cfg_obj = ScaleAmpConfig()
        elif isinstance(self.pipeline_post_cfg, ScaleAmpConfig):
            self._pipeline_post_cfg_obj = self.pipeline_post_cfg
        elif isinstance(self.pipeline_post_cfg, dict):
            self._pipeline_post_cfg_obj = ScaleAmpConfig(**self.pipeline_post_cfg)
        else:
            raise TypeError("pipeline_post_cfg must be None, dict, or post_UNet_utils.ScaleAmpConfig")

    def _build_model_input_from_W(self, W_2d, device):
        from utils_wavelet_geometry import wavelet_shape_maps

        Wt = torch.from_numpy(np.asarray(W_2d, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        maps = wavelet_shape_maps(Wt)
        X = torch.cat([Wt, maps["coh"], maps["lap"], maps["gradmag"]], dim=1)
        return X

    def _probability_map_from_logits(self, logits):
        if logits.ndim == 4:
            logits = logits[:, 0]
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        mode = self.pipeline_prob_mode
        if mode == "sigmoid_norm":
            P = torch.sigmoid(logits)
            P = P / (P.sum(dim=(1, 2), keepdim=True) + 1e-8)
            return P
        # default softmax
        B, H, W = logits.shape
        return torch.softmax(logits.reshape(B, -1), dim=1).reshape(B, H, W)

    def _extract_x_peaks_1d(self, probs_hw):
        Px = torch.as_tensor(probs_hw).detach().float().cpu().numpy()
        if Px.ndim == 2:
            Px = Px.sum(axis=0)
        Px = Px.reshape(-1)

        k = int(max(0, self.pipeline_edge_exclude))
        if k > 0 and 2 * k < Px.size:
            Px[:k] = 0.0
            Px[-k:] = 0.0

        thr = float(self.pipeline_min_prominence_rel) * float(np.max(Px) if Px.size else 0.0)
        peaks = []
        for i in range(1, len(Px) - 1):
            if Px[i] > Px[i - 1] and Px[i] > Px[i + 1] and Px[i] > thr:
                peaks.append(i)
        if len(Px) >= 2:
            if Px[0] > Px[1] and Px[0] > thr:
                peaks.append(0)
            if Px[-1] > Px[-2] and Px[-1] > thr:
                peaks.append(len(Px) - 1)

        if not peaks and len(Px):
            j = int(np.argmax(Px))
            if Px[j] > thr:
                peaks = [j]

        peaks = sorted(peaks, key=lambda j: Px[j], reverse=True)
        kept = []
        dmin = int(max(0, self.pipeline_min_distance))
        for p in peaks:
            if all(abs(p - q) >= dmin for q in kept):
                kept.append(p)
            if len(kept) >= int(self.pipeline_max_peaks):
                break
        return np.asarray(kept, dtype=np.int64)

    def _run_pipeline_estimator(self, y_for_post, W_lor_out, W_disp_out, priors_map):
        from post_UNet_utils import estimate_scale_amplitude_noniterative

        self._pipeline_init_once()
        dev = self._pipeline_device_resolved

        with torch.no_grad():
            if self._pipeline_model_is_dual:
                X_lor = self._build_model_input_from_W(W_lor_out, dev)
                X_disp = self._build_model_input_from_W(W_disp_out, dev)
                pri = None
                if priors_map is not None:
                    pri = torch.from_numpy(priors_map.astype(np.float32)).unsqueeze(0).to(dev)
                logits = self._pipeline_model(X_lor, X_disp, priors=pri)
            else:
                # Use primary wavelet for single-stem model path.
                X = self._build_model_input_from_W(W_lor_out, dev)
                logits = self._pipeline_model(X)
            P = self._probability_map_from_logits(logits)[0]
            pred_x = self._extract_x_peaks_1d(P)

        est = estimate_scale_amplitude_noniterative(
            pred_x_idx=pred_x,
            y=np.asarray(y_for_post, dtype=np.float32),
            x_axis=self.x,
            widths=self.widths,
            W_lor4=np.asarray(W_lor_out, dtype=np.float32),
            W_disp=np.asarray(W_disp_out, dtype=np.float32),
            cfg=self._pipeline_post_cfg_obj,
            return_debug=False,
        )
        return (
            np.asarray(est["x_idx"], dtype=np.int64),
            np.asarray(est["amp"], dtype=np.float32),
            np.asarray(est["fwhm"], dtype=np.float32),
            np.asarray(est["sigma"], dtype=np.float32),
            np.asarray(est["gamma"], dtype=np.float32),
        )

    def __getitem__(self, idx):
        rng = self._rng_for_idx(idx)

        n_peaks = self._sample_n_peaks(rng)
        S_global = rng.uniform(*self.S_range)

        amps_lor = self._sample_amplitudes_mode(n_peaks, rng, log_mode=self.LogAmp)
        amps_disp = self._sample_amplitudes_mode(n_peaks, rng, log_mode=False)
        fwhm = rng.uniform(*self.fwhm_range, size=n_peaks).astype(np.float32)
        sigmas = (fwhm / 2.354820045).astype(np.float32)
        if self.gamma_range is not None:
            gammas = rng.uniform(*self.gamma_range, size=n_peaks).astype(np.float32)
        else:
            gammas = np.full(n_peaks, self.gamma_constant, dtype=np.float32)

        seps = np.zeros(n_peaks - 1, dtype=np.float32)
        S_adj = np.zeros(n_peaks - 1, dtype=np.float32)

        for i in range(n_peaks - 1):
            req = self._required_sep(
                fwhm[i], amps_lor[i], fwhm[i + 1], amps_lor[i + 1], S_global
            )
            seps[i] = req

            w_eff = max(fwhm[i], fwhm[i + 1])
            r = max(amps_lor[i], amps_lor[i + 1]) / max(1e-12, min(amps_lor[i], amps_lor[i + 1]))
            S_adj[i] = req / (w_eff * (1.0 + self.beta_amp * np.log10(r)))

        total_span = float(np.sum(seps)) if n_peaks > 1 else 0.0
        cushion = 0.5 * float(np.max(fwhm))

        left_min = self.xmin + cushion
        left_max = self.xmax - cushion - total_span
        c0 = left_min if left_max <= left_min else rng.uniform(left_min, left_max)

        centers = np.empty(n_peaks, dtype=np.float32)
        centers[0] = c0
        for i in range(1, n_peaks):
            centers[i] = centers[i - 1] + seps[i - 1]

        keep = (centers >= self.xmin) & (centers <= self.xmax)
        if np.count_nonzero(keep) < self.min_peaks_after_trunc:
            keep[:] = False
            keep[: max(1, self.min_peaks_after_trunc)] = True

        centers = centers[keep]
        amps_lor = amps_lor[keep]
        amps_disp = amps_disp[keep]
        sigmas = sigmas[keep]
        gammas = gammas[keep]
        fwhm = fwhm[keep]

        S_peak = np.full(len(centers), np.inf, dtype=np.float32)
        for i in range(len(centers)):
            vals = []
            if i > 0:
                vals.append(S_adj[i - 1])
            if i < len(centers) - 1:
                vals.append(S_adj[i])
            if vals:
                S_peak[i] = min(vals)

        y_lor = generate_multipeak_Raman(self.x, centers, amps_lor, sigmas, gammas, Real=False)
        y_disp = generate_multipeak_Raman(self.x, centers, amps_disp, sigmas, gammas, Real=True)
        y = y_disp if self.Real else y_lor
        if self.noise_std > 0:
            y += self.noise_std * rng.standard_normal(len(y)).astype(np.float32)

        W_lor = multiscale_lorentz4_transform(self.x, y_lor, self.widths)
        W_disp = cwt_dispersive_lorentzian(
            y_disp,
            self.x,
            self.widths,
            support_factor=self.disp_lor_support_factor,
            pad_mode=self.disp_lor_pad_mode,
            mask_coi=self.disp_lor_mask_coi,
        )

        if self.wavelet == "Lor4":
            W = W_lor
        elif self.wavelet == "MexHat":
            W = multiscale_mexhat_transform(self.x, y, self.widths)
        else:
            W = W_disp
        W = self._apply_repr_for_wavelet(W, self.wavelet)
        W /= (np.std(W) + 1e-12)

        W_lor_out = self._apply_repr_for_wavelet(W_lor, "Lor4")
        W_lor_out /= (np.std(W_lor_out) + 1e-12)
        W_disp_out = self._apply_repr_for_wavelet(W_disp, "DispLor")
        W_disp_out /= (np.std(W_disp_out) + 1e-12)

        priors_map = None
        if self.return_priors:
            # Priors are derived from signed DispLor response (linear repr).
            priors_map = self._compute_prior_maps(W_disp.astype(np.float32))

        target = np.zeros_like(W, dtype=np.float32)
        for c, s in zip(centers, sigmas):
            target += multiscale_anisotropic_target(
                self.x,
                self.widths,
                float(c),
                float(s),
                gamma_x=self.target_gamma_x,
                sigma_s=self.target_sigma_s,
                scale_spread=self.target_scale_spread,
            )
        target /= target.sum() + 1e-12

        true_xs = np.array(
            [int(np.argmin(np.abs(self.x - c))) for c in centers],
            dtype=np.int64,
        )

        base = (
            torch.tensor(W, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(true_xs, dtype=torch.long),
            torch.tensor(centers, dtype=torch.float32),
            torch.tensor(amps_disp if self.Real else amps_lor, dtype=torch.float32),
            torch.tensor(fwhm, dtype=torch.float32),
            torch.tensor(S_peak, dtype=torch.float32),
            torch.tensor(sigmas, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(gammas, dtype=torch.float32),

        )

        if self.return_both_wavelets:
            base = base + (
                torch.tensor(W_lor_out, dtype=torch.float32),
                torch.tensor(W_disp_out, dtype=torch.float32),
            )

        if self.return_priors:
            base = base + (torch.tensor(priors_map, dtype=torch.float32),)

        if self.return_pipeline_estimates:
            y_for_post = y_lor if self.pipeline_use_nondispersive_y else y
            p_x, p_amp, p_fwhm, p_sigma, p_gamma = self._run_pipeline_estimator(
                y_for_post=y_for_post,
                W_lor_out=W_lor_out,
                W_disp_out=W_disp_out,
                priors_map=priors_map if self.return_priors else None,
            )
            base = base + (
                torch.tensor(p_x, dtype=torch.long),
                torch.tensor(p_amp, dtype=torch.float32),
                torch.tensor(p_fwhm, dtype=torch.float32),
            )
            if self._pipeline_post_cfg_obj.return_sigma:
                base = base + (torch.tensor(p_sigma, dtype=torch.float32),)
            if self._pipeline_post_cfg_obj.return_gamma:
                base = base + (torch.tensor(p_gamma, dtype=torch.float32),)

        return base


def dataset_from_curriculum_stage(
    *,
    curriculum,
    phase_idx,
    x,
    widths,
    base_defaults=None,
    n_peaks=None,
    separability_range=None,
    noise_std=None,
    dataset_overrides=None,
):
    """
    Build RamanDataset using defaults from a curriculum stage (1-based phase index),
    with optional overrides.

    Parameters
    ----------
    curriculum : list[dict]
        Curriculum list where each stage may contain "dataset_changes".
    phase_idx : int
        1-based index into curriculum.
    x, widths : array-like
        Raman axis and wavelet scales.
    base_defaults : dict | None
        Optional baseline RamanDataset kwargs. Stage changes and explicit overrides
        are applied on top.
    n_peaks, separability_range, noise_std : optional
        Common explicit overrides.
    dataset_overrides : dict | None
        Any additional RamanDataset kwargs to override.

    Returns
    -------
    tuple
        (dataset, stage, dataset_kwargs_used)
    """
    if phase_idx < 1 or phase_idx > len(curriculum):
        raise IndexError(f"phase_idx must be in [1, {len(curriculum)}], got {phase_idx}")

    stage = curriculum[phase_idx - 1]
    stage_changes = dict(stage.get("dataset_changes", {}))

    if n_peaks is not None:
        stage_changes["n_peaks"] = n_peaks
    if separability_range is not None:
        stage_changes["separability_range"] = separability_range
    if noise_std is not None:
        stage_changes["noise_std"] = noise_std

    kwargs = dict(
        n_samples=1000,
        n_peaks=(1, 1),
        fwhm_range=(7.0, 45.0),
        separability_range=(1.0, 3.0),
        amp_range=(0.01, 1.0),
        noise_std=0.0,
        target_gamma_x=6.0,
        target_sigma_s=0.7,
        target_scale_spread=0,
        LogAmp=True,
        wavelet="DispLor",
        Real=True,
        wavelet_repr="linear",
        margin=200.0,
        disp_lor_mask_coi=False,
    )
    if base_defaults:
        kwargs.update(dict(base_defaults))
    kwargs.update(stage_changes)
    if dataset_overrides:
        kwargs.update(dict(dataset_overrides))

    ds = RamanDataset(x=x, widths=widths, **kwargs)
    return ds, stage, kwargs


def raman_collate_fn(batch):
    W_list, target_list, true_xs_list, centers_list, amps_list = zip(
        *[sample[:5] for sample in batch]
    )

    spectrum_list = [sample[8] for sample in batch] if len(batch[0]) > 8 else None

    W_lor_list = None
    W_disp_list = None
    priors_list = None
    pipeline_x_list = None
    pipeline_amp_list = None
    pipeline_fwhm_list = None
    if len(batch[0]) >= 11:
        W_lor_list = [sample[9] for sample in batch]
        W_disp_list = [sample[10] for sample in batch]
        idx = 11
        if len(batch[0]) > idx and batch[0][idx].ndim == 2 and batch[0][idx].shape[0] == 2:
            priors_list = [sample[idx] for sample in batch]
            idx += 1
        if len(batch[0]) >= idx + 3:
            pipeline_x_list = [sample[idx] for sample in batch]
            pipeline_amp_list = [sample[idx + 1] for sample in batch]
            pipeline_fwhm_list = [sample[idx + 2] for sample in batch]
    elif len(batch[0]) == 10:
        # Priors-only extension (without return_both_wavelets).
        priors_list = [sample[9] for sample in batch]

    W = torch.stack(W_list, dim=0)
    target = torch.stack(target_list, dim=0)

    max_peaks = max(x.shape[0] for x in true_xs_list)

    def pad_1d(seq, pad_value=0):
        out = torch.full((len(seq), max_peaks), pad_value, dtype=seq[0].dtype)
        for i, x in enumerate(seq):
            out[i, : x.shape[0]] = x
        return out

    true_xs = pad_1d(true_xs_list, pad_value=-1)
    centers = pad_1d(centers_list, pad_value=0.0)
    amps = pad_1d(amps_list, pad_value=0.0)

    if spectrum_list is not None:
        spectrum = torch.stack(spectrum_list, dim=0)
        out = [W, target, true_xs, centers, amps, spectrum]
        if W_lor_list is not None and W_disp_list is not None:
            out.extend([torch.stack(W_lor_list, dim=0), torch.stack(W_disp_list, dim=0)])
        if priors_list is not None:
            out.append(torch.stack(priors_list, dim=0))
        if pipeline_x_list is not None:
            max_pred = max(x.shape[0] for x in pipeline_x_list)
            def pad_pred(seq, pad_value=0):
                outp = torch.full((len(seq), max_pred), pad_value, dtype=seq[0].dtype)
                for i, x in enumerate(seq):
                    outp[i, : x.shape[0]] = x
                return outp
            out.extend([
                pad_pred(pipeline_x_list, pad_value=-1),
                pad_pred(pipeline_amp_list, pad_value=0.0),
                pad_pred(pipeline_fwhm_list, pad_value=0.0),
            ])
        return tuple(out)

    return W, target, true_xs, centers, amps


class SampleWrapper(tuple):
    """
    Tuple subclass with named attribute access.

    GT fields:
        .wavelet  .target  .true_xs  .centers  .amplitudes
        .fwhm  .separability  .sigmas  .spectrum  .gammas
        .wavelet_lor  .wavelet_disp  .priors

    Pipeline prediction fields:
        .pred_x      integer indices into x (≈ cm⁻¹ for this axis)
        .pred_amp    predicted amplitudes
        .pred_fwhm   predicted FWHM
        .pred_sigma  predicted sigma
        .pred_gamma  predicted gamma
    """
    _FIELDS = [
        'wavelet', 'target', 'true_xs', 'centers', 'amplitudes',
        'fwhm', 'separability', 'sigmas', 'spectrum', 'gammas',
        'wavelet_lor', 'wavelet_disp', 'priors',
        'pred_x', 'pred_amp', 'pred_fwhm', 'pred_sigma', 'pred_gamma',
    ]

    def __getattr__(self, name):
        try:
            idx = self._FIELDS.index(name)
        except ValueError:
            raise AttributeError(f"SampleWrapper has no attribute '{name}'")
        if idx >= len(self):
            raise AttributeError(
                f"'{name}' not present in this sample "
                f"(field index {idx} but len={len(self)})"
            )
        return self[idx]

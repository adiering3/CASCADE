"""
tidytorch_utils.py

PyTorch reimplementation of tidyfit_utils.py.
Drop-in replacement: same function names and return signatures throughout.

Key differences from the JAX version:
  - jax.lax.scan  → Python for loop
  - jax.vmap      → batched tensor broadcasting (compute_model) or loops (precompute_wavelets)
  - optax         → torch.optim.Adam + LambdaLR + clip_grad_norm_
  - jax.jit       → torch.compile (optional, call torch.compile(fn) yourself)
  - process_in_batches: JAX vmap-per-pixel → sequential loop over spectra.
    GPU is used for model evaluation; the fit loop itself cannot be vmapped
    across spectra because it contains a Python for loop.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
from dataset_utils import RamanDataset, voigt_peak

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

FWHM_FROM_SIGMA = 2.35482  # 2 * sqrt(2 * ln 2)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_tensor(a, device: Optional[torch.device] = None) -> torch.Tensor:
    """Accept numpy array, PyTorch tensor, or Python scalar → float32 tensor."""
    if isinstance(a, torch.Tensor):
        t = a.detach().float()
    elif hasattr(a, "__array__"):       # numpy / JAX arrays
        t = torch.from_numpy(np.asarray(a, dtype=np.float32))
    else:
        t = torch.tensor(a, dtype=torch.float32)
    return t.to(device) if device is not None else t


# ---------------------------------------------------------------------------
# Peak shape model
# ---------------------------------------------------------------------------

def single_peak(
    wn: torch.Tensor,
    peak_height,
    center,
    sigma,
    gamma,
) -> torch.Tensor:
    """Pseudo-Voigt peak scaled so its maximum equals peak_height."""
    wn          = _to_tensor(wn)
    peak_height = _to_tensor(peak_height, wn.device)
    center      = _to_tensor(center,      wn.device)
    sigma       = _to_tensor(sigma,       wn.device)
    gamma       = _to_tensor(gamma,       wn.device)

    profile = pseudo_voigt(wn - center, sigma, gamma)
    return (peak_height / (profile.max() + 1e-12)) * profile


def compute_wavelet_peak(sigma, gamma, x: torch.Tensor) -> torch.Tensor:
    """Unit-height pseudo-Voigt wavelet, zero-centred on x."""
    x_c = x - x.mean()
    profile = pseudo_voigt(x_c, sigma, gamma)
    return profile / (profile.max() + 1e-12)


def precompute_wavelets(sigmas, gammas, x):
    """Build a pseudo-Voigt wavelet bank for every (sigma, gamma) pair.

    Uses broadcasting to compute all (n_sigmas × n_gammas) wavelets in a
    single vectorised call — replaces the double jax.vmap in the original.

    Parameters
    ----------
    sigmas : (n_sigmas,) tensor — Gaussian half-widths
    gammas : (n_gammas,) tensor — Lorentzian half-widths
    x      : (n_pts,)   tensor — spectral axis

    Returns
    -------
    bank : (n_sigmas, n_gammas, n_pts) float32 tensor, unit-height wavelets
    """
    x_c = x - x.mean()
    # (n_s, 1, 1) and (1, n_g, 1) broadcast against (1, 1, n_pts)
    s = sigmas.reshape(-1, 1, 1)
    g = gammas.reshape(1, -1, 1)
    xb = x_c.reshape(1, 1, -1)

    profiles = pseudo_voigt(xb, s, g)  # (n_s, n_g, n_pts)
    return profiles / (profiles.max(dim=-1, keepdim=True).values + 1e-12)
# ---------------------------------------------------------------------------
# Lorentz4 wavelet support
# ---------------------------------------------------------------------------

def lorentz4_wavelet_torch(a: float, x: torch.Tensor) -> torch.Tensor:
    """
    Lorentz4 wavelet at scale a, zero-centred on x.
    Mirrors dataset_utils.lorentz4_wavelet but returns a torch tensor.
    """
    x_c = x - x.mean()
    t   = x_c / a
    psi = (1 - 6*t**2 + t**4) / (1 + t**2)**4
    dx  = (x[1] - x[0]).item()
    return psi / (torch.sqrt((psi**2).sum() * dx) + 1e-12)


def precompute_lorentz4_wavelets(widths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Build Lor4 wavelet bank for each width.
    Returns shape (n_widths, n_pts).
    Pass this as wavelet_peaks when calling process_pixel_fit with convolution='Lor4'.
    """
    return torch.stack([lorentz4_wavelet_torch(a.item(), x) for a in widths])


def lorentz4_multiscale_transform(
    signal: torch.Tensor,
    widths,               # kept for API symmetry with voigt version; not used directly
    x,                    # kept for API symmetry; not used directly
    precomputed_wavelets: torch.Tensor,  # (n_widths, n_pts)
) -> torch.Tensor:
    """
    FFT cross-correlation of signal against a precomputed Lor4 wavelet bank.

    Returns shape (n_widths, 1, n_pts) — the trailing 1 keeps the output
    compatible with build_initial_guesses_from_derivative_mask, which
    expects a 3-D (n_sigmas, n_gammas, n_pts) response tensor.
    """
    signal_fft   = torch.fft.fft(signal)                         # (n_pts,)
    wavelets_fft = torch.fft.fft(torch.fft.ifftshift(precomputed_wavelets, dim=-1))
    cross = torch.fft.ifft(signal_fft.unsqueeze(0) * wavelets_fft.conj())
    responses = cross.real  # no fftshift
    energy    = (precomputed_wavelets**2).sum(dim=1, keepdim=True) + 1e-12
    responses = responses / energy                                # (n_widths, n_pts)

    return responses.unsqueeze(1)                                 # (n_widths, 1, n_pts)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def denoise_spectrum(
    signal: torch.Tensor,
    x: torch.Tensor,
    sigma: float,
    gamma: float = 0.0,
) -> torch.Tensor:
    """
    Smooth a spectrum by convolving with a unit-area pseudo-Voigt kernel.

    Use this to suppress high-frequency noise before derivative-based peak
    detection.  A good choice for sigma is roughly half the width of your
    narrowest expected real peak — wide enough to kill noise bumps, narrow
    enough to preserve peak centres and shapes.  Pass the returned tensor
    in place of the raw spectrum to process_conv_deriv_fit or
    process_pixel_fit.

    Parameters
    ----------
    signal : (n_pts,) raw spectrum
    x      : (n_pts,) wavenumber axis (need not be uniformly spaced)
    sigma  : Gaussian half-width of the smoothing kernel (same units as x)
    gamma  : Lorentzian half-width; 0.0 → pure Gaussian (minimal ringing)

    Returns
    -------
    smoothed : (n_pts,) denoised spectrum, same dtype/device as signal
    """
    signal  = _to_tensor(signal)
    x       = _to_tensor(x, signal.device)

    n    = signal.shape[0]
    # Convert sigma from wavenumber units to pixel units using the median
    # local spacing.  This is robust to non-uniform axes (e.g. 1/lambda BCARS).
    dx      = (x[-1] - x[0]).abs() / (n - 1)
    sigma_px = float(sigma) / dx.item()
    gamma_px = max(float(gamma), 1e-6) / dx.item()

    sigma_t = torch.tensor(sigma_px, dtype=torch.float32, device=x.device)
    gamma_t = torch.tensor(gamma_px, dtype=torch.float32, device=x.device)

    # Build the kernel on an INTEGER index axis centred exactly at n//2.
    # ifftshift then moves that centre to index 0 with no residual phase error,
    # regardless of whether the wavenumber axis is uniform or not.
    idx    = torch.arange(n, dtype=torch.float32, device=x.device) - n // 2
    kernel = pseudo_voigt(idx, sigma_t, gamma_t)
    kernel = kernel / kernel.sum()                        # → unit area

    # ifftshift moves the kernel centre from index n//2 to index 0 so the
    # FFT convolution produces a correctly aligned result with no circular
    # phase shift in the output.
    sig_fft = torch.fft.fft(signal)
    ker_fft = torch.fft.fft(torch.fft.ifftshift(kernel))
    return torch.fft.ifft(sig_fft * ker_fft).real


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_peaks_derivative_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    min_height: float = 0.0,
) -> torch.Tensor:
    """
    Boolean mask of peak locations via sign-change of the first derivative.
    Uses explicit finite differences (no torch.gradient version dependency).
    """
    dx = (x[1] - x[0]).item()

    # First derivative  (central differences, first-order at edges)
    dy        = torch.empty_like(y)
    dy[1:-1]  = (y[2:] - y[:-2]) / (2.0 * dx)
    dy[0]     = (y[1]  - y[0])   / dx
    dy[-1]    = (y[-1] - y[-2])  / dx

    # Second derivative
    d2y       = torch.empty_like(y)
    d2y[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dx * dx)
    d2y[0]    = d2y[1]
    d2y[-1]   = d2y[-2]

    sign_change     = (dy[:-1] > 0) & (dy[1:] <= 0)
    concave         = d2y[1:-1] < 0
    peak_mask_inner = sign_change[:-1] & concave

    mask        = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    mask[1:-1]  = peak_mask_inner
    return mask & (y > min_height)

def find_peaks_derivative_mask_batch(x, y_batch, min_height=0.0):
    """
    Vectorized peak detection across multiple signals simultaneously.
    y_batch: (n_scales, n_pts)
    Returns: (n_scales, n_pts) boolean mask
    """
    dx = (x[1] - x[0]).item()

    # First derivative (central differences) — all scales at once
    dy = torch.empty_like(y_batch)
    dy[:, 1:-1] = (y_batch[:, 2:] - y_batch[:, :-2]) / (2.0 * dx)
    dy[:, 0]    = (y_batch[:, 1]  - y_batch[:, 0])    / dx
    dy[:, -1]   = (y_batch[:, -1] - y_batch[:, -2])   / dx

    # Second derivative
    d2y = torch.empty_like(y_batch)
    d2y[:, 1:-1] = (y_batch[:, 2:] - 2.0 * y_batch[:, 1:-1] + y_batch[:, :-2]) / (dx * dx)
    d2y[:, 0]    = d2y[:, 1]
    d2y[:, -1]   = d2y[:, -2]

    sign_change     = (dy[:, :-1] > 0) & (dy[:, 1:] <= 0)
    concave         = d2y[:, 1:-1] < 0
    peak_mask_inner = sign_change[:, :-1] & concave

    mask = torch.zeros_like(y_batch, dtype=torch.bool)
    mask[:, 1:-1] = peak_mask_inner
    return mask & (y_batch > min_height)
# ---------------------------------------------------------------------------
# Initial guess builder
# ---------------------------------------------------------------------------

def build_initial_guesses_from_derivative_mask(
    response_tensor: torch.Tensor,
    sigmas: torch.Tensor,
    gammas: torch.Tensor,
    wavenumbers: torch.Tensor,
    signal: torch.Tensor,
    peak_mask: torch.Tensor,
    max_peaks: int = 200,
    min_spacing: float = 0.0,
    scale_preference_fraction: float = 0.8,
) -> torch.Tensor:
    """
    Build a flat (max_peaks * 4,) initial-guess vector from the wavelet
    response tensor and the peak mask.

    min_spacing : float
        Minimum wavenumber separation between any two initial-guess peaks.
        Candidates are considered in descending amplitude order; a candidate
        is skipped if it falls within ``min_spacing`` of an already-kept peak.
        0.0 (default) disables the check and preserves original behaviour.

    scale_preference_fraction : float
        Bias toward finer (narrower) wavelets.  At each position, any scale
        whose per-scale-normalised response is at least this fraction of the
        best-across-scales response is considered "competitive"; the finest
        (narrowest) competitive scale is chosen.  1.0 → always pick the
        single best scale (no bias); 0.0 → always pick the finest scale
        regardless of fit quality.  Default 0.8.
    """
    n_sigmas, n_gammas, _ = response_tensor.shape
    n_scales = n_sigmas * n_gammas
    flat     = response_tensor.reshape(n_scales, -1)

    # Per-scale normalisation: remove the effect of wide wavelets having
    # larger absolute cross-correlations by dividing each scale by its
    # own maximum response across the spectrum.
    flat_max  = flat.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    flat_norm = flat / flat_max                        # (n_scales, n_pts)

    # For each position find the best normalised response across scales, then
    # mark every scale within `scale_preference_fraction` of that best as
    # "competitive".  Among competitive scales, prefer the finest (lowest
    # index = smallest sigma) to counter the residual bias toward wide scales
    # whose flat response curves stay near their maximum over large regions.
    pos_best = flat_norm.max(dim=0, keepdim=True).values          # (1, n_pts)
    eligible = flat_norm >= scale_preference_fraction * pos_best  # (n_scales, n_pts)

    # Weights: index 0 (finest) → n_scales, coarsest → 1.
    fine_weights = torch.arange(n_scales, 0, -1,
                                device=flat.device, dtype=torch.float32)
    best_idx = torch.argmax(
        eligible.float() * fine_weights.unsqueeze(1), dim=0
    )                                                              # (n_pts,)

    i_coords = best_idx // n_gammas
    j_coords = best_idx  % n_gammas

    all_peaks = torch.stack(
        [signal, wavenumbers, sigmas[i_coords], gammas[j_coords]],
        dim=1,
    )                                              # (n_pts, 4)

    all_peaks = all_peaks * peak_mask.float().unsqueeze(1)
    sort_idx  = torch.argsort(-all_peaks[:, 0])
    sorted_peaks = all_peaks[sort_idx]             # (n_pts, 4), descending amp

    if min_spacing > 0.0:
        # Vectorized non-maximum suppression in center-space:
        # after descending-amplitude sort, suppress any candidate that lies
        # within min_spacing of ANY higher-amplitude candidate.
        alive = sorted_peaks[:, 0] > 0
        cand = sorted_peaks[alive]

        if cand.shape[0] == 0:
            return torch.zeros(max_peaks, 4, device=wavenumbers.device).reshape(-1)

        ctrs = cand[:, 1]
        dist = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).abs()         # (K, K)
        too_close = (dist < min_spacing).triu(diagonal=1)            # only i<j (i higher amp)
        suppress = too_close.any(dim=0)                              # suppress j if any stronger i is close
        kept = cand[~suppress][:max_peaks]

        out = torch.zeros(max_peaks, 4, device=wavenumbers.device)
        out[:kept.shape[0]] = kept
        return out.reshape(-1)
    else:
        return sorted_peaks[:max_peaks].reshape(-1)


# ---------------------------------------------------------------------------
# Wavelet transform
# ---------------------------------------------------------------------------

def voigt_multiscale_transform(
    signal: torch.Tensor,
    sigmas,        # kept for API compatibility with original; not used
    gammas,        # kept for API compatibility with original; not used
    x,             # kept for API compatibility with original; not used
    precomputed_wavelets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FFT cross-correlation of signal against all precomputed wavelets.

    Parameters
    ----------
    precomputed_wavelets : (n_sigmas, n_gammas, n_pts)

    Returns
    -------
    (response_tensor, last_response)  — same shape contract as the JAX version.
    """
    n_sigmas, n_gammas, n_pts = precomputed_wavelets.shape
    signal_fft    = torch.fft.fft(signal)                          # (n_pts,)

    wavelets_flat = precomputed_wavelets.reshape(-1, n_pts)
    wavelets_fft = torch.fft.fft(torch.fft.ifftshift(wavelets_flat, dim=-1))
    cross = torch.fft.ifft(signal_fft.unsqueeze(0) * wavelets_fft.conj())
    responses = cross.real  # no fftshift needed now
    energy    = (wavelets_flat ** 2).sum(dim=1, keepdim=True) + 1e-12
    responses = (responses / energy).reshape(n_sigmas, n_gammas, n_pts)

    return responses, responses[-1, -1, :]


# Alias — keeps old name working if anything imports it directly
voigt_multiscale_transform_jax = voigt_multiscale_transform


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
import math
@torch.compile(fullgraph=True)
def pseudo_voigt(x, sigma, gamma):
    """Thompson-Cox-Hastings approximation — accepts torch tensors."""
    sigma = torch.clamp(sigma, min=1e-6)
    gamma = torch.clamp(gamma, min=1e-6)

    fwhm_g = 2.35482 * sigma
    fwhm_l = 2.0 * gamma

    fwhm = (fwhm_g**5 + 2.69269 * fwhm_g**4 * fwhm_l +
            2.42843 * fwhm_g**3 * fwhm_l**2 +
            4.47163 * fwhm_g**2 * fwhm_l**3 +
            0.07842 * fwhm_g * fwhm_l**4 + fwhm_l**5)**0.2
    fwhm = torch.clamp(fwhm, min=1e-6)

    ratio = fwhm_l / fwhm
    eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
    eta = torch.clamp(eta, 0.0, 1.0)

    z = torch.clamp(x / fwhm, -50.0, 50.0)
    gaussian   = torch.exp(-4.0 * math.log(2.0) * z**2)
    lorentzian = 1.0 / (1.0 + 4.0 * z**2)

    return eta * lorentzian + (1.0 - eta) * gaussian

@torch.compile(fullgraph=True)
def compute_model(p, x):
    """
    Vectorized computation of all peaks using broadcasting.
    Replaces jax.vmap — no loop needed.

    Args:
        p: torch tensor of shape (n_peaks * 4,)
        x: torch tensor of x-values
    """
    p_r  = p.reshape(-1, 4)
    amps = p_r[:, 0].unsqueeze(1)  # (n_peaks, 1)
    ctrs = p_r[:, 1].unsqueeze(1)  # (n_peaks, 1)
    sigs = p_r[:, 2].unsqueeze(1)  # (n_peaks, 1)
    gams = p_r[:, 3].unsqueeze(1)  # (n_peaks, 1)

    x_shifted = x.unsqueeze(0) - ctrs          # (n_peaks, n_points)
    pv = pseudo_voigt(x_shifted, sigs, gams)   # (n_peaks, n_points)
    return (amps * pv).sum(0)                   # (n_points,)

@torch.compile(fullgraph=True)
def residual_projected(p, x, y):
    """Least-squares residual — differentiable via torch autograd."""
    model = compute_model(p, x)
    return torch.sum((model - y) ** 2)


# _LO / _HI are initialised on first use inside project_bounds and
# re-created whenever the tensor device changes.  init_sweep_context also
# sets them explicitly for the batch fitting path.
_LO = None
_HI = None
@torch.compile(fullgraph=True)
def project_bounds(p, x, spectrum, fwhm_exp=None, fwhm_max_scale=None, hyperparams=None):
    global _LO, _HI
    if _LO is None or _LO.device != p.device:
        _LO = torch.tensor([0.0, 0.0, 0.05, 0.05], device=p.device)
        _HI = torch.tensor([1.0, 4000.0, 50.0, 50.0], device=p.device)
    p_r = p.reshape(-1, 4)

    # Build per-peak upper bounds: amplitude cap = spectrum value at peak center + 10%
    hi = _HI.unsqueeze(0).expand_as(p_r).clone()  # (N, 4)
    low = _LO.unsqueeze(0).expand_as(p_r).clone()         # (N, 4)
    centers = p_r[:, 1]  # peak center positions (x-axis)

    # Find nearest x index for each peak center
    # x is assumed to be a 1D tensor of the spectral axis
    indices = torch.argmin(torch.abs(x.unsqueeze(0) - centers.unsqueeze(1)), dim=1)
    amp_caps = spectrum[indices]  

    hi[:, 0] = amp_caps * 1.1 
    hi[:,1] = centers * 1.1
    low[:,1] = centers * 0.9
    result = torch.clamp(p_r, low, hi)
    result = torch.where(torch.isfinite(result), result, p_r)
    return result.reshape(-1)

@torch.compile(fullgraph=True)
def _compiled_forward(params, x_t, y_t):
    model = compute_model(params, x_t)
    return torch.sum((model - y_t) ** 2)

def fit_with_bounded_adam(y, x, p0_stack, max_iter=1000, tol=1e-8):
    """Fit a single spectrum using Adam with warmup/cosine LR and bounds projection.

    LR schedule:
      - Steps 0 → warmup (100):   linear ramp from 0 → peak_lr (1e-2)
      - Steps warmup → max_iter:  cosine decay from peak_lr → end_lr (1e-5)

    After each gradient step, parameters are clamped to valid bounds:
      amplitude ∈ [0, 1],  centre ∈ [0, 4000],  sigma/gamma ∈ [0.05, 50]

    Convergence is checked every 50 steps; early exit when loss < tol.

    Parameters
    ----------
    y         : (n_pts,) target spectrum
    x         : (n_pts,) wavenumber axis
    p0_stack  : (max_peaks * 4,) flat initial parameter vector [amp,ctr,sig,gam, ...]
    max_iter  : maximum optimisation steps
    tol       : loss value below which optimisation is considered converged

    Returns
    -------
    final_params : (max_peaks * 4,) optimised parameters (detached)
    last_loss    : float, final loss value
    converged    : bool
    n_iter       : int, number of steps taken
    """
    if isinstance(p0_stack, torch.Tensor):
        dev = p0_stack.device
    elif isinstance(y, torch.Tensor):
        dev = y.device
    elif isinstance(x, torch.Tensor):
        dev = x.device
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_t = x.clone().detach().float().to(dev) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32, device=dev)
    y_t = y.clone().detach().float().to(dev) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32, device=dev)
    params = p0_stack.clone().detach().float().to(dev) if isinstance(p0_stack, torch.Tensor) else torch.tensor(p0_stack, dtype=torch.float32, device=dev)

    params = project_bounds(params, x_t, y_t)
    params = params.requires_grad_(True)

    peak_lr = 1e-2
    end_lr  = 1e-5
    warmup  = 100

    # fused=True uses a single CUDA kernel for the Adam update — faster on GPU
    optimizer = torch.optim.Adam([params], lr=peak_lr, betas=(0.9, 0.999), fused=True)
    
    last_loss = float('inf')
    n_iter = 0

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = residual_projected(params, x_t, y_t)
        loss.backward()

        # Manual grad clipping — avoids clip_grad_norm_ overhead
        grad = params.grad
        grad_norm = grad.norm()
        if grad_norm > 1.0:
            grad.mul_(1.0 / grad_norm)

        # Manual LR schedule — avoids scheduler overhead
        if i < warmup:
            lr = peak_lr * (i / warmup)
        else:
            progress = (i - warmup) / max(1, max_iter - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = end_lr + (peak_lr - end_lr) * cosine
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()

        with torch.no_grad():
            params.data = project_bounds(params.data, x_t, y_t)

        n_iter += 1
        if i % 50 == 0:
            last_loss = loss.item()
            if last_loss < tol:
                break


    final_params = params.detach()
    converged = last_loss < tol
    return final_params, last_loss, converged, n_iter

# ---------------------------------------------------------------------------
# Post-fit utilities
# ---------------------------------------------------------------------------

def prune_peaks(params: torch.Tensor, amp_threshold: float = 1e-3) -> torch.Tensor:
    """Zero out peaks whose amplitude is below threshold.

    Parameters
    ----------
    params        : (max_peaks * 4,) flat parameter vector
    amp_threshold : peaks with |amplitude| ≤ this value are zeroed

    Returns
    -------
    (max_peaks * 4,) with sub-threshold peaks replaced by zeros
    """
    peaks     = params.reshape(-1, 4)
    keep_mask = peaks[:, 0].abs() > amp_threshold
    return (peaks * keep_mask.unsqueeze(1)).reshape(-1)


def deduplicate_peaks(params: torch.Tensor, min_spacing: float) -> torch.Tensor:
    """Remove peaks that are too close to a stronger neighbour.

    Peaks are sorted by descending amplitude.  A peak is suppressed if any
    higher-amplitude peak lies within *min_spacing* wavenumber units of its
    centre.  Uses a vectorised pairwise distance matrix instead of a Python
    loop — O(N²) memory but fast in practice for typical peak counts.

    Parameters
    ----------
    params      : (max_peaks * 4,) flat parameter vector [amp, ctr, sig, gam, ...]
    min_spacing : minimum allowed centre separation; 0.0 disables the check

    Returns
    -------
    (max_peaks * 4,) with suppressed peaks zeroed out
    """
    if min_spacing <= 0.0:
        return params

    peaks = params.reshape(-1, 4)
    amps = peaks[:, 0]
    ctrs = peaks[:, 1]

    # Sort by descending amplitude
    order = torch.argsort(-amps)
    sorted_ctrs = ctrs[order]
    sorted_amps = amps[order]

    # Pairwise distance matrix between centers
    dist = (sorted_ctrs.unsqueeze(0) - sorted_ctrs.unsqueeze(1)).abs()  # (N, N)

    # For each pair (i, j) where i < j (i has higher amp), if dist < min_spacing, suppress j
    # Build upper-triangular "too close" mask
    too_close = (dist < min_spacing).triu(diagonal=1)  # (N, N), only i < j

    # Also mask out zero-amplitude peaks
    alive = sorted_amps > 0  # (N,)
    too_close = too_close & alive.unsqueeze(0) & alive.unsqueeze(1)

    # A peak j is suppressed if ANY higher-amp peak i is too close
    suppress_sorted = too_close.any(dim=0)  # (N,)

    # Map back to original order
    suppress = torch.zeros_like(amps, dtype=torch.bool)
    suppress[order] = suppress_sorted

    return (peaks * (~suppress).unsqueeze(1)).reshape(-1)


def _prune_peaks_batch(params_batch: torch.Tensor, amp_threshold: float = 1e-3) -> torch.Tensor:
    """Batch version of prune_peaks for (B, n_params)."""
    peaks = params_batch.reshape(params_batch.shape[0], -1, 4)
    keep_mask = peaks[:, :, 0].abs() > amp_threshold
    return (peaks * keep_mask.unsqueeze(-1)).reshape_as(params_batch)


def _deduplicate_peaks_batch(params_batch: torch.Tensor, min_spacing: float) -> torch.Tensor:
    """Batch version of deduplicate_peaks for (B, n_params)."""
    if min_spacing <= 0.0:
        return params_batch

    B = params_batch.shape[0]
    peaks = params_batch.reshape(B, -1, 4)
    amps = peaks[:, :, 0]
    ctrs = peaks[:, :, 1]

    order = torch.argsort(-amps, dim=1)
    order_exp = order.unsqueeze(-1).expand(-1, -1, 4)
    sorted_peaks = torch.gather(peaks, 1, order_exp)

    sorted_ctrs = sorted_peaks[:, :, 1]
    sorted_amps = sorted_peaks[:, :, 0]

    dist = (sorted_ctrs.unsqueeze(2) - sorted_ctrs.unsqueeze(1)).abs()  # (B, N, N)
    too_close = torch.triu(dist < min_spacing, diagonal=1)

    alive = sorted_amps > 0
    too_close = too_close & alive.unsqueeze(2) & alive.unsqueeze(1)

    suppress_sorted = too_close.any(dim=1)  # (B, N)
    kept_sorted = sorted_peaks * (~suppress_sorted).unsqueeze(-1)

    restored = torch.zeros_like(peaks)
    restored.scatter_(1, order_exp, kept_sorted)
    return restored.reshape_as(params_batch)

# ---------------------------------------------------------------------------
# Per-pixel processor
# ---------------------------------------------------------------------------

def process_pixel_fit(
    spectrum: torch.Tensor,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    gammas: torch.Tensor,
    wavelet_peaks: torch.Tensor,
    response_threshold: float = 0.0001,
    max_peaks: int = 200,
    max_iter: int = 5000,
    tol: float = 1e-8,
    convolution: str = 'voigt',
    min_spacing_in: float = 0.0,
    min_spacing_post: float = 0.0,
    scale_preference_fraction: float = 0.8,
) -> Tuple[torch.Tensor, bool, int]:
    """
    Fit one spectrum. Accepts numpy arrays or torch tensors for all inputs.

    convolution : 'voigt' | 'Lor4'
        'voigt' — use voigt_multiscale_transform with a (n_sigmas, n_gammas, n_pts)
                  wavelet bank built by precompute_wavelets().
        'Lor4'  — use lorentz4_multiscale_transform with a (n_widths, n_pts)
                  wavelet bank built by precompute_lorentz4_wavelets().
                  Pass widths as sigmas; gammas should be a 1-element tensor
                  (used as the initial gamma guess for all detected peaks).
    min_spacing : float
        Minimum wavenumber gap enforced between initial-guess peak centers.
        0.0 (default) disables the check.
    scale_preference_fraction : float
        Passed to build_initial_guesses_from_derivative_mask.  Lower values
        bias more aggressively toward narrower wavelet scales.  Default 0.8.
    """
    device   = wavelet_peaks.device
    spectrum = _to_tensor(spectrum, device)
    x        = _to_tensor(x,        device)
    sigmas   = _to_tensor(sigmas,   device)
    gammas   = _to_tensor(gammas,   device)

    is_valid = not torch.all(spectrum == 0).item()
    if convolution == 'voigt':
        response_tensor, _ = voigt_multiscale_transform(
            spectrum, sigmas, gammas, x, wavelet_peaks
        )
    elif convolution == 'Lor4':
        # Build Lor4 wavelet bank from sigmas (treated as widths) on the fly.
        # wavelet_peaks is ignored in this branch — pass your voigt bank or None.
        lor4_bank = precompute_lorentz4_wavelets(sigmas, x)   # (n_widths, n_pts)
        response_tensor = lorentz4_multiscale_transform(
            spectrum, sigmas, x, lor4_bank
        )                                                       # (n_widths, 1, n_pts)
        # build_initial_guesses expects gammas to index the n_gammas=1 dimension
        gammas = gammas[:1]
    else:
        raise ValueError(f"Unknown convolution type: {convolution!r}")
    peak_mask = find_peaks_derivative_mask(x, spectrum, min_height=response_threshold)
    p0        = build_initial_guesses_from_derivative_mask(
        response_tensor, sigmas, gammas, x, spectrum, peak_mask, max_peaks,
        min_spacing=min_spacing_in,
        scale_preference_fraction=scale_preference_fraction,
    )

    params, error, converged, n_iter = fit_with_bounded_adam(
        y=spectrum, x=x, p0_stack=p0, max_iter=max_iter, tol=tol
    )
    params = prune_peaks(params, amp_threshold=1e-3)
    params = deduplicate_peaks(params, min_spacing_post)

    if not is_valid:
        params = torch.zeros_like(params)
        error  = float("nan")

    return params, converged, n_iter, response_tensor, peak_mask, p0


# ---------------------------------------------------------------------------
# Convolution-response derivative peak detection
# ---------------------------------------------------------------------------

def process_conv_deriv_fit(
    spectrum: torch.Tensor,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    gammas: torch.Tensor,
    wavelet_peaks: torch.Tensor,
    response_threshold: float = 0.0001,
    min_scale_votes: int = 2,
    max_peaks: int = 200,
    max_iter: int = 5000,
    tol: float = 1e-8,
    convolution: str = 'Lor4',
    amp_threshold: float = 1e-3,
    min_height: float = 0.0,
    min_spacing_in: float = 0.0,
    min_spacing_post: float = 0.0,
    scale_preference_fraction: float = 0.8,
) -> Tuple[torch.Tensor, bool, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Like process_pixel_fit, but runs derivative-based peak detection on the
    wavelet convolution responses rather than on the raw spectrum.

    For each wavelet scale, ``find_peaks_derivative_mask`` is applied to that
    scale's response curve (sign-change of 1st derivative + concavity check).
    A position is retained as a peak candidate only when at least
    ``min_scale_votes`` scales independently flag it — cross-scale voting makes
    the detection much more robust than differentiating the noisy raw spectrum.

    Parameters
    ----------
    spectrum          : raw spectrum (n_pts,)
    x                 : wavenumber axis (n_pts,)
    sigmas            : wavelet width parameters
    gammas            : wavelet Lorentzian width parameters
    wavelet_peaks     : precomputed wavelet bank (n_sigmas, n_gammas, n_pts)
                        for 'voigt'; for 'Lor4' the bank is built internally
                        and this argument is used only to infer the device.
    response_threshold: minimum peak response value (max across all scales) to
                        keep a candidate — gates out very weak detections.
    min_scale_votes   : minimum number of wavelet scales that must independently
                        detect a local maximum at a position for it to be kept.
    max_peaks         : maximum number of peaks to initialise
    max_iter          : Adam optimiser iterations
    tol               : convergence tolerance
    convolution       : 'voigt' | 'Lor4'
    amp_threshold     : post-fit amplitude threshold for pruning peaks.
    min_height        : minimum response height for per-scale derivative detection.
    min_spacing       : minimum wavenumber gap between initial-guess peak centers.
                        Candidates are selected greedily by descending amplitude;
                        any candidate within ``min_spacing`` of an already-kept
                        peak is skipped. 0.0 disables (default).
    scale_preference_fraction : float
                        Passed to build_initial_guesses_from_derivative_mask.
                        Lower values bias more aggressively toward narrower
                        wavelet scales.  Default 0.8.

    Returns
    -------
    params, converged, n_iter, response_tensor, peak_mask, p0
      — same signature as process_pixel_fit.
    """
    device   = wavelet_peaks.device
    spectrum = _to_tensor(spectrum, device)
    x        = _to_tensor(x,        device)
    sigmas   = _to_tensor(sigmas,   device)
    gammas   = _to_tensor(gammas,   device)

    is_valid = not torch.all(spectrum == 0).item()

    # ---- 1. Multiscale convolution ----------------------------------------
    if convolution == 'voigt':
        response_tensor, _ = voigt_multiscale_transform(
            spectrum, sigmas, gammas, x, wavelet_peaks
        )
    elif convolution == 'Lor4':
        lor4_bank = precompute_lorentz4_wavelets(sigmas, x)
        response_tensor = lorentz4_multiscale_transform(
            spectrum, sigmas, x, lor4_bank
        )                                               # (n_widths, 1, n_pts)
        gammas = gammas[:1]
    else:
        raise ValueError(f"Unknown convolution type: {convolution!r}")

    n_sigmas, n_gammas, n_pts = response_tensor.shape

    # ---- 2. Vote: derivative peak detection on each scale's response ------
    # vote_count = torch.zeros(n_pts, dtype=torch.long, device=device)
    # for i in range(n_sigmas):
    #     for j in range(n_gammas):
    #         scale_response = response_tensor[i, j, :]
    #         scale_mask = find_peaks_derivative_mask(x, scale_response, min_height=min_height)  # (n_pts,)
    #         vote_count = vote_count + scale_mask.long()
    # Flatten (n_sigmas, n_gammas, n_pts) → (n_scales, n_pts)
    flat_responses = response_tensor.reshape(-1, n_pts)
    all_masks = find_peaks_derivative_mask_batch(x, flat_responses, min_height=min_height)
    vote_count = all_masks.long().sum(dim=0)  # (n_pts,)
    # ---- 3. Consensus peak mask ------------------------------------------
    #  (a) enough scales must agree on a peak here
    #  (b) max response across all scales must clear the threshold
    max_response = response_tensor.reshape(n_sigmas * n_gammas, n_pts).max(dim=0).values
    peak_mask = (vote_count >= min_scale_votes) & (max_response > response_threshold)

    # ---- 4. Build initial guesses (same logic as process_pixel_fit) -------
    p0 = build_initial_guesses_from_derivative_mask(
        response_tensor, sigmas, gammas, x, spectrum, peak_mask, max_peaks,
        min_spacing=min_spacing_in,
        scale_preference_fraction=scale_preference_fraction,
    )

    # ---- 5. Fit ----------------------------------------------------------
    params, error, converged, n_iter = fit_with_bounded_adam(
        y=spectrum, x=x, p0_stack=p0, max_iter=max_iter, tol=tol
    )
    params = prune_peaks(params, amp_threshold=amp_threshold)
    params = deduplicate_peaks(params, min_spacing_post)

    if not is_valid:
        params = torch.zeros_like(params)
        error  = float("nan")

    return params, converged, n_iter, response_tensor, peak_mask, p0


# ---------------------------------------------------------------------------
# Batch processor  (replaces process_in_batches_adam)
# ---------------------------------------------------------------------------

def process_in_batches_adam(
    y_batch: np.ndarray,
    x: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    batch_size: int = 50,
    response_threshold: float = 0.0001,
    max_peaks: int = 200,
    min_spacing_in: float = 0.0,   # kept for API compatibility; unused
    min_spacing_post: float = 0.0,          # kept for API compatibility; unused
    max_iter: int = 5000,
    tol: float = 1e-8,
):
    """
    Process a (height, width, n_wn) hyperspectral cube.

    Note: the JAX version used vmap to parallelise per-pixel fits across the
    batch dimension.  Because fit_with_bounded_adam contains a Python for loop
    it cannot be torch.vmap'd, so spectra are processed sequentially here.
    GPU is still used for all tensor operations inside each fit.

    Returns
    -------
    params  : (height, width, n_peaks * 4)
    conv    : (height, width)  bool
    n_iter  : (height, width)  int
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    height, width, n_wn = y_batch.shape
    n_spectra = height * width

    x_t      = _to_tensor(x,      device)
    sigmas_t = _to_tensor(sigmas, device)
    gammas_t = _to_tensor(gammas, device)
    spectra  = _to_tensor(y_batch.reshape(-1, n_wn), device)

    print("Pre-computing wavelets...")
    wavelet_peaks = precompute_wavelets(sigmas_t, gammas_t, x_t)
    print(f"Wavelets shape: {wavelet_peaks.shape}")

    all_params: list = []
    all_conv:   list = []
    all_iter:   list = []
    start = time.time()

    for i in range(0, n_spectra, batch_size):
        batch = spectra[i : i + batch_size]

        for spec in batch:
            params, conv, n_it = process_pixel_fit(
                spec, x_t, sigmas_t, gammas_t, wavelet_peaks,
                response_threshold=response_threshold,
                max_peaks=max_peaks,
                min_spacing_in=min_spacing_in,
                min_psacing_post = min_spacing_post,
                max_iter=max_iter,
                tol=tol,
            )
            all_params.append(params.cpu().numpy())
            all_conv.append(int(conv))
            all_iter.append(n_it)

        elapsed   = time.time() - start
        done      = min(i + batch_size, n_spectra)
        rate      = done / elapsed
        remaining = (n_spectra - done) / rate if rate > 0 else 0
        print(
            f"{done}/{n_spectra} ({done / n_spectra * 100:.1f}%) — "
            f"{rate:.1f} px/s — ETA: {remaining:.0f}s"
        )

        if device.type == "cuda" and i % (batch_size * 20) == 0 and i > 0:
            torch.cuda.empty_cache()

    params_arr = np.stack(all_params, axis=0)
    conv_arr   = np.array(all_conv,   dtype=bool)
    iter_arr   = np.array(all_iter,   dtype=np.int32)

    total = time.time() - start
    print(f"\nTotal: {total:.1f}s  ({total / n_spectra:.3f}s/px, "
          f"{n_spectra / total:.1f} px/s)")
    print(f"Converged: {conv_arr.sum()}/{n_spectra} ({conv_arr.mean() * 100:.1f}%)")
    print(f"Iterations — mean: {iter_arr.mean():.1f}  "
          f"min: {iter_arr.min()}  max: {iter_arr.max()}")

    return (
        params_arr.reshape(height, width, -1),
        conv_arr.reshape(height, width),
        iter_arr.reshape(height, width),
    )


# ---------------------------------------------------------------------------
# Plotting  (unchanged logic; handles tensor or numpy inputs)
# ---------------------------------------------------------------------------

def plot_voigt_fit_res_two(
    x,
    y_true,
    params,
    peaks_dict=None,
    title: str = "Voigt Decomposition",
    amp_threshold: float = 1e-6,
    scale_residual: bool = True,
):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",   "font.size": 11,
        "axes.labelsize": 12,     "axes.titlesize": 13,
        "legend.fontsize": 10,    "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    x_np   = _np(x)
    y_np   = _np(y_true)
    p_np   = _np(params)
    x_t    = torch.as_tensor(x_np, dtype=torch.float32)

    p      = p_np[: (len(p_np) // 4) * 4].reshape(-1, 4)
    y_peaks = []
    for amp, ctr, sigma, gamma in p:
        if abs(amp) > amp_threshold:
            pk = single_peak(x_t, float(amp), float(ctr), float(sigma), float(gamma))
            y_peaks.append(pk.numpy())

    y_peaks   = np.array(y_peaks) if y_peaks else np.empty((0, len(x_np)))
    model_sum = y_peaks.sum(axis=0) if len(y_peaks) > 0 else np.zeros_like(y_np)
    residual  = y_np - model_sum

    rmse = np.sqrt(np.mean(residual ** 2))
    r2   = 1.0 - np.sum(residual ** 2) / np.sum((y_np - y_np.mean()) ** 2)

    fig = plt.figure(figsize=(6.5, 5.5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.2], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(x_np, y_np,      color="black",   lw=1.5, label="Data")
    ax1.plot(x_np, model_sum, color="#8B0000", lw=1.5, label="Voigt fit")
    for yp in y_peaks:
        ax1.plot(x_np, yp, color="0.75", lw=0.8)
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.set_title(f"{title}  (n = {len(y_peaks)})")
    ax1.legend(frameon=False)
    ax1.tick_params(direction="in")
    ax1.grid(False)

    if scale_residual:
        ax2.plot(x_np, residual * 1e5, color="0.3", lw=1.0)
        ax2.set_ylabel("Residual\n($\\times 10^{-5}$)")
    else:
        ax2.plot(x_np, residual, color="0.3", lw=1.0)
        ax2.set_ylabel("Residual")

    ax2.axhline(0, color="black", linestyle="--", lw=0.8)
    ax2.set_xlabel("Raman shift (cm$^{-1}$)")
    ax2.tick_params(direction="in")
    ax2.grid(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()

    print(f"RMSE = {rmse:.6e}")
    print(f"R²   = {r2:.6f}")
    print(f"Number of peaks = {len(y_peaks)}")

    return fig, (ax1, ax2)




# ── Sweep context ─────────────────────────────────────────────────────────────
# Call init_sweep_context() once before using _fit_one, _run_sweep, or
# _plot_sweep (via plot_utils).  These module-level globals are populated by
# that call and are read by the batch GPU kernels (_denoise_batch,
# _lor4_transform_batch, _fit_batch_adam) as well as the peak-detection and
# matching helpers.  They are intentionally global so that multiple calls to
# _fit_one / _run_sweep within a session share the same pre-computed tensors.

_ctx_x      = None   # (n_pts,) numpy float32 wavenumber axis
_ctx_sigmas = None   # (n_widths,) tensor — wavelet width parameters
_ctx_gammas = None   # (n_gammas,) tensor — Lorentzian gamma parameters
_ctx_widths = None   # width priors passed to RamanDataset
_ctx_device = None   # torch.device used by all batch kernels
_x_dev      = None   # _ctx_x moved to device
_lor4_bank  = None   # (n_widths, n_pts) Lorentz4 wavelet bank on device
_lor4_fft   = None   # (n_widths, n_pts) FFT of _lor4_bank (cached to avoid recompute)
_lor4_energy = None  # (n_widths, 1) per-wavelet L2 energy for FFT normalisation
_ker_fft    = None   # (n_pts,) FFT of the denoise convolution kernel
_LO         = None   # (4,) lower parameter bounds [amp, ctr, sig, gam]
_HI         = None   # (4,) upper parameter bounds


def init_sweep_context(x, sigmas, gammas, device, widths,
                       denoise_sigma=3.0, denoise_gamma=1e-6):
    """Precompute and cache GPU tensors used by _run_sweep, _fit_one, etc.

    Call this once after setting up your data:

        init_sweep_context(x_gpu, sigmas, gammas, device, widths)

    Parameters
    ----------
    x       : array-like (n_pts,) — wavenumber axis
    sigmas  : tensor (n_widths,)  — wavelet width parameters
    gammas  : tensor (n_gammas,)  — Lorentzian gamma parameters
    device  : torch.device
    widths  : array-like          — peak-width priors for RamanDataset
    """
    global _ctx_x, _ctx_sigmas, _ctx_gammas, _ctx_widths, _ctx_device
    global _x_dev, _lor4_bank, _lor4_fft, _lor4_energy, _ker_fft, _LO, _HI

    _ctx_device = device
    _ctx_x      = np.asarray(x, dtype=np.float32)
    _ctx_sigmas = sigmas if isinstance(sigmas, torch.Tensor) else torch.as_tensor(sigmas, dtype=torch.float32)
    _ctx_gammas = gammas if isinstance(gammas, torch.Tensor) else torch.as_tensor(gammas, dtype=torch.float32)
    _ctx_widths = widths

    _x_dev     = torch.as_tensor(_ctx_x, dtype=torch.float32).to(device)
    _lor4_bank = precompute_lorentz4_wavelets(_ctx_sigmas.to(device), _x_dev)
    _lor4_fft = torch.fft.fft(_lor4_bank)
    _lor4_energy = (_lor4_bank ** 2).sum(dim=-1, keepdim=True)

    n       = _x_dev.shape[0]
    dx      = (_x_dev[-1] - _x_dev[0]).abs() / (n - 1)
    _sigma_t = torch.tensor(denoise_sigma / dx.item(), dtype=torch.float32, device=device)
    _gamma_t = torch.tensor(max(denoise_gamma, 1e-6) / dx.item(), dtype=torch.float32, device=device)
    idx      = torch.arange(n, dtype=torch.float32, device=device) - n // 2
    kernel   = pseudo_voigt(idx, _sigma_t, _gamma_t)
    kernel   = kernel / kernel.sum()
    _ker_fft = torch.fft.fft(torch.fft.ifftshift(kernel))

    _LO = torch.tensor([0.0,    0.0, 0.05, 0.05], device=device)
    _HI = torch.tensor([1.0, 4000.0, 50.0, 50.0], device=device)

    print(f"Sweep context initialised on {device}")
    print(f"  Wavelet bank : {_lor4_bank.shape}")
    print(f"  Denoise kern : {_ker_fft.shape}")


# ── Shared helpers for parameter sweeps ──────────────────────────────────────

def _cpu(t):
    """Move a tensor/array to a CPU numpy float32 array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _sync_device(device: Optional[torch.device]) -> None:
    """Synchronise accelerator work so wall-clock timings are accurate."""
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _match_peaks(gt_centers, gt_amps, gt_sigmas, gt_gammas,
                 rec_params_flat, tolerance=15.0, amp_threshold=1e-2,
                 x_arr=None):
    """One-to-one GT↔recovered matching (Hungarian if available).

    Pass x_arr (numpy, shape (n_pts,)) to also compute mean_shape_rmse
    for matched pairs; otherwise it is returned as nan.
    """
    p     = _cpu(rec_params_flat).reshape(-1, 4)
    valid = p[:, 0] > amp_threshold
    rc, ra, rs, rg = p[valid, 1], p[valid, 0], p[valid, 2], p[valid, 3]

    gt_centers = _cpu(gt_centers)
    gt_amps    = _cpu(gt_amps)
    gt_sigmas  = _cpu(gt_sigmas)
    gt_gammas  = _cpu(gt_gammas)

    n_gt, n_rec = len(gt_centers), int(valid.sum())
    matched_gt, matched_rec = set(), set()
    amp_errs, ctr_errs, sig_errs = [], [], []
    matched_pairs_idx = []          # (rec_j, gt_best) — needed for shape RMSE

    if n_gt > 0 and n_rec > 0:
        dmat = np.abs(rc[:, None] - gt_centers[None, :])

        if linear_sum_assignment is not None:
            rec_idx, gt_idx = linear_sum_assignment(dmat)
        else:
            # Fallback to previous greedy behaviour when SciPy is unavailable.
            rec_idx, gt_idx = [], []
            used_gt = set()
            for j, c in enumerate(rc):
                d = np.abs(gt_centers - c)
                b = int(np.argmin(d))
                if b not in used_gt:
                    rec_idx.append(j)
                    gt_idx.append(b)
                    used_gt.add(b)
            rec_idx = np.asarray(rec_idx, dtype=int)
            gt_idx = np.asarray(gt_idx, dtype=int)

        for j, g in zip(rec_idx, gt_idx):
            d = dmat[j, g]
            if d < tolerance:
                matched_gt.add(int(g))
                matched_rec.add(int(j))
                amp_errs.append(abs(ra[j] - gt_amps[g])   / (abs(gt_amps[g])   + 1e-9))
                ctr_errs.append(abs(rc[j] - gt_centers[g]))
                sig_errs.append(abs(rs[j] - gt_sigmas[g]) / (abs(gt_sigmas[g]) + 1e-9))
                matched_pairs_idx.append((int(j), int(g)))

    # ── Shape RMSE for matched pairs ─────────────────────────────────────────
    shape_rmses = []
    if x_arr is not None and matched_pairs_idx:
        for rec_j, gt_b in matched_pairs_idx:
            wf_gt  = np.asarray(voigt_peak(x_arr,
                                           gt_centers[gt_b], gt_amps[gt_b],
                                           gt_sigmas [gt_b], gt_gammas[gt_b]))
            wf_fit = np.asarray(voigt_peak(x_arr,
                                           rc[rec_j], ra[rec_j],
                                           rs[rec_j], rg[rec_j]))
            sr = (np.sqrt(np.mean((wf_gt - wf_fit) ** 2))
                  / (np.sqrt(np.mean(wf_gt ** 2)) + 1e-12))
            shape_rmses.append(float(sr))

    tp = len(matched_gt); fp = n_rec - tp; fn = n_gt - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    nan  = float('nan')

    return dict(
        n_gt=n_gt, n_rec=n_rec, tp=tp, fp=fp, fn=fn,
        precision=prec, recall=rec, f1=f1,
        mean_amp_err    = float(np.mean(amp_errs))    if amp_errs    else nan,
        mean_ctr_err    = float(np.mean(ctr_errs))    if ctr_errs    else nan,
        mean_sig_err    = float(np.mean(sig_errs))    if sig_errs    else nan,
        mean_shape_rmse = float(np.mean(shape_rmses)) if shape_rmses else nan,
    )

# ── Vectorised GPU kernels ────────────────────────────────────────────────────

def _denoise_batch(spectra: torch.Tensor) -> torch.Tensor:
    """(B, n_pts) → (B, n_pts)  single FFT call."""
    return torch.fft.ifft(torch.fft.fft(spectra) * _ker_fft).real


def _lor4_transform_batch(spectra: torch.Tensor) -> torch.Tensor:
    """(B, n_pts) → (B, n_widths, n_pts)  single batched FFT."""
    sig_fft   = torch.fft.fft(spectra)                                  # (B, n_pts)
    cross     = torch.fft.ifft(
        sig_fft.unsqueeze(1) * _lor4_fft.unsqueeze(0).conj()             # (B, n_widths, n_pts)
    )
    responses = torch.fft.fftshift(cross.real, dim=-1)
    return responses / (_lor4_energy.unsqueeze(0) + 1e-12)              # (B, n_widths, n_pts)


def _compute_model_batch(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the Voigt sum model for a batch of spectra in one shot.

    params : (B, max_peaks*4)   [amp, ctr, sig, gam] per peak
    x      : (n_pts,)
    returns: (B, n_pts)
    """
    B    = params.shape[0]
    p_r  = params.reshape(B, -1, 4)                    # (B, max_peaks, 4)
    amps = p_r[:, :, 0, None]                          # (B, max_peaks, 1)
    ctrs = p_r[:, :, 1, None]                          # (B, max_peaks, 1)
    sigs = p_r[:, :, 2, None]                          # (B, max_peaks, 1)
    gams = p_r[:, :, 3, None]                          # (B, max_peaks, 1)
    x_s  = x[None, None, :] - ctrs                    # (B, max_peaks, n_pts)
    pv   = pseudo_voigt(x_s, sigs, gams)               # (B, max_peaks, n_pts)
    return (amps * pv).sum(dim=1)                       # (B, n_pts)


def _fit_batch_adam(spectra: torch.Tensor,
                    x: torch.Tensor,
                    p0_batch: torch.Tensor,
                    grad_mask: torch.Tensor,
                    max_iter: int = 2000,
                    tol: float = 1e-5,
                    aggressive_start_steps: int = 60,
                    aggressive_lr_mult: float = 3.0,
                    aggressive_clip_norm: float = 3.0,
                    aggressive_beta1: float = 0.85,
                    progress_every: int = 0,
                    progress_prefix: str = "fit") -> torch.Tensor:
    """
    Adam optimisation for a whole batch of spectra simultaneously.

    Matches fit_with_bounded_adam exactly:
      - 100-step linear warmup → cosine LR decay to 1e-5
      - Per-spectrum gradient clipping (norm ≤ 1.0 per row)
      - NaN-safe bounds projection at every step

    spectra   : (B, n_pts)          — target spectra (denoised)
    p0_batch  : (B, max_peaks*4)    — padded initial guesses
    grad_mask : (B, max_peaks*4)    — 1.0 for real peaks, 0.0 for padding
    returns   : (B, max_peaks*4)    — optimised parameters
    """
    B, npq = p0_batch.shape

    # Initial bounds projection with NaN guard
    p_init  = p0_batch.reshape(B, -1, 4)
    clamped = torch.clamp(p_init, _LO, _HI)
    params  = torch.where(torch.isfinite(clamped), clamped, p_init).reshape(B, npq)
    params  = params.detach().requires_grad_(True)

    peak_lr = 1e-2
    end_lr  = 1e-5
    warmup  = 100
    aggressive_steps = int(max(0, min(aggressive_start_steps, max_iter)))
    aggressive_lr = min(peak_lr * max(1.0, aggressive_lr_mult), 5e-2)

    optimizer = torch.optim.Adam([params], lr=peak_lr, betas=(0.9, 0.999), fused=True)
    t_fit_start = time.perf_counter()

    last_loss = float('inf')
    for i in range(max_iter):
        optimizer.zero_grad()

        model = _compute_model_batch(params, x)         # (B, n_pts)
        loss  = ((model - spectra) ** 2).sum()          # scalar
        loss.backward()

        with torch.no_grad():
            # Zero gradients for zero-padded peaks
            params.grad.mul_(grad_mask)

            # Per-spectrum gradient clipping — matches fit_with_bounded_adam.
            # Compute L2 norm independently for each spectrum's param block,
            # then scale down any row whose norm exceeds 1.0.
            g      = params.grad                            # (B, npq)
            norms  = g.norm(dim=1, keepdim=True)            # (B, 1)
            clip_now = aggressive_clip_norm if i < aggressive_steps else 1.0
            scale  = (clip_now / norms.clamp(min=clip_now)) # clip at clip_now
            params.grad.mul_(scale)

        # Aggressive-start schedule:
        #   1) aggressive burst for first `aggressive_steps`
        #   2) short transition to base peak_lr
        #   3) cosine decay to end_lr
        if i < aggressive_steps:
            lr = aggressive_lr
        elif i < aggressive_steps + warmup:
            t = (i - aggressive_steps) / max(1, warmup)
            lr = aggressive_lr + t * (peak_lr - aggressive_lr)
        else:
            progress = (i - aggressive_steps - warmup) / max(1, max_iter - aggressive_steps - warmup)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr       = end_lr + (peak_lr - end_lr) * cosine
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            if i < aggressive_steps:
                pg['betas'] = (aggressive_beta1, 0.999)
            else:
                pg['betas'] = (0.9, 0.999)

        optimizer.step()

        # Bounds projection with NaN guard
        with torch.no_grad():
            p_r     = params.data.reshape(B, -1, 4)
            clamped = torch.clamp(p_r, _LO, _HI)
            params.data = torch.where(
                torch.isfinite(clamped), clamped, p_r
            ).reshape(B, npq)

        if i % 50 == 0:
            last_loss = loss.item() / B   # mean per-spectrum loss
            if last_loss < tol:
                break

        if progress_every and progress_every > 0:
            step_done = i + 1
            if (step_done % progress_every == 0) or (step_done == max_iter):
                elapsed = time.perf_counter() - t_fit_start
                it_per_sec = step_done / max(elapsed, 1e-9)
                eta = (max_iter - step_done) / max(it_per_sec, 1e-9)
                loss_mean = loss.item() / B
                print(
                    f"[{progress_prefix}] iter {step_done}/{max_iter} "
                    f"({100.0 * step_done / max_iter:.1f}%) | "
                    f"loss/pix={loss_mean:.4e} | lr={lr:.2e} | "
                    f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
                )

    return params.detach()


# ── Main sweep function ───────────────────────────────────────────────────────

def _run_sweep(param_name, param_values, n_samples=30, seed=42,
               noise_fixed=0.01, sep_fixed=(1.0, 1.5),
               max_iter=2000, tol=1e-5, amp_thr=1e-2, ctr_tol=15.0,
               response_threshold=0.02, min_scale_votes=6,
               min_spacing=7.0, max_peaks=200,
               aggressive_start_steps=60, aggressive_lr_mult=3.0,
               aggressive_clip_norm=3.0, aggressive_beta1=0.85,
               profile=False, return_timing=False):
    """
    Fully vectorised sweep. GPU kernel breakdown per level:
      1. Batch denoise       — 1 FFT on (B, n_pts)
      2. Batch Lor4 xform   — 1 FFT on (B, n_widths, n_pts)
      3. Batch peak voting  — find_peaks_derivative_mask_batch on (B*n_widths, n_pts)
      4. p0 build           — per-spectrum loop (variable peak counts, unavoidable)
      5. Pad p0 → (B, max_peaks*4) and run ONE batched Adam optimisation

    profile : bool
        Print per-level timing and aggregate timing summary.
    return_timing : bool
        If True, return a tuple (all_results, timing_dict).
    """
    gammas_lor = _ctx_gammas[:1].to(_ctx_device)
    sigmas_dev = _ctx_sigmas.to(_ctx_device)
    do_timing = profile or return_timing

    all_results = {}
    timing_per_level = {} if do_timing else None

    def _elapsed(start_t: float) -> float:
        _sync_device(_ctx_device)
        return time.perf_counter() - start_t

    level_meta = []
    all_raw = []
    spectra_blocks = []
    dataset_times = {}

    # 0) Build all sweep datasets first, then fit as one global batch
    for pval in param_values:
        if do_timing:
            t0 = time.perf_counter()

        ds_kw = dict(noise_std=noise_fixed, separability_range=sep_fixed)
        if param_name == 'noise_std':
            ds_kw['noise_std'] = pval
        else:
            ds_kw['separability_range'] = pval

        ds = RamanDataset(
            x=_ctx_x,
            n_peaks=(40, 50),
            widths=_ctx_widths,
            n_samples=n_samples,
            seed=seed,
            **ds_kw,
        )
        raw = [ds[i] for i in range(n_samples)]
        key = pval if param_name == 'noise_std' else str(pval)
        level_meta.append((pval, key, len(raw)))
        all_raw.extend(raw)
        spectra_blocks.append(np.asarray([np.asarray(r[8], dtype=np.float32) for r in raw], dtype=np.float32))

        if do_timing:
            dataset_times[key] = _elapsed(t0)

    if len(all_raw) == 0:
        timing_dict = None
        if do_timing:
            timing_dict = {
                'per_level_sec': {},
                'mean_stage_sec': {
                    'dataset': 0.0,
                    'stack_to_device': 0.0,
                    'denoise': 0.0,
                    'lor4_transform': 0.0,
                    'peak_vote': 0.0,
                    'build_p0': 0.0,
                    'fit_adam': 0.0,
                    'prune_match': 0.0,
                },
                'mean_total_sec': 0.0,
                'mean_per_pixel_sec': 0.0,
                'n_levels': int(len(param_values)),
                'n_samples': int(n_samples),
            }
        if return_timing:
            return all_results, timing_dict
        return all_results

    # ── 1. Stack ALL spectra → GPU once ───────────────────────────────────────
    if do_timing:
        t0 = time.perf_counter()
    spectra_np = np.concatenate(spectra_blocks, axis=0)
    spectra_gpu = torch.as_tensor(spectra_np, dtype=torch.float32, device=_ctx_device)
    B_total = spectra_gpu.shape[0]
    if do_timing:
        stack_to_device_sec = _elapsed(t0)
    else:
        stack_to_device_sec = 0.0

    # ── 2. Batch denoise over full sweep block ─────────────────────────────────
    if do_timing:
        t0 = time.perf_counter()
    spec_d = _denoise_batch(spectra_gpu)
    if do_timing:
        denoise_sec = _elapsed(t0)
    else:
        denoise_sec = 0.0

    # ── 3. Batch Lor4 transform over full sweep block ─────────────────────────
    if do_timing:
        t0 = time.perf_counter()
    resp = _lor4_transform_batch(spec_d)
    _, n_widths, n_pts = resp.shape
    if do_timing:
        lor4_transform_sec = _elapsed(t0)
    else:
        lor4_transform_sec = 0.0

    # ── 4. Batch peak-detection voting over full sweep block ──────────────────
    if do_timing:
        t0 = time.perf_counter()
    flat_masks = find_peaks_derivative_mask_batch(
        _x_dev, resp.reshape(B_total * n_widths, n_pts), min_height=0.0
    )
    masks = flat_masks.reshape(B_total, n_widths, n_pts)
    vote_count = masks.long().sum(dim=1)
    max_resp = resp.max(dim=1).values
    peak_masks = (vote_count >= min_scale_votes) & (max_resp > response_threshold)
    if do_timing:
        peak_vote_sec = _elapsed(t0)
    else:
        peak_vote_sec = 0.0

    # ── 5a. Build initial guesses + pad for full sweep block ─────────────────
    if do_timing:
        t0 = time.perf_counter()
    npq_max = max_peaks * 4
    p0_flat_list = []
    n_real_peaks = []

    for i in range(B_total):
        p0 = build_initial_guesses_from_derivative_mask(
            resp[i].unsqueeze(1),
            sigmas_dev,
            gammas_lor,
            _x_dev,
            spec_d[i],
            peak_masks[i],
            max_peaks=max_peaks,
            min_spacing=min_spacing,
            scale_preference_fraction=0.8,
        )
        p0_flat = p0[:npq_max]
        p0_flat_list.append(p0_flat)

        n_peaks_i = int((p0_flat[0::4] > 0).sum().item())
        n_peaks_i = min(n_peaks_i, max_peaks)
        n_real_peaks.append(n_peaks_i)

    max_real = max(1, max(n_real_peaks) if n_real_peaks else 1)
    npq = max_real * 4
    p0_batch = torch.zeros(B_total, npq, device=_ctx_device)
    grad_mask = torch.zeros(B_total, npq, dtype=torch.float32, device=_ctx_device)

    for i, (p0_flat, n_peaks_i) in enumerate(zip(p0_flat_list, n_real_peaks)):
        n_take = min(p0_flat.numel(), npq)
        p0_batch[i, :n_take] = p0_flat[:n_take]
        grad_mask[i, :n_peaks_i * 4] = 1.0
    if do_timing:
        build_p0_sec = _elapsed(t0)
    else:
        build_p0_sec = 0.0

    # ── 5c. ONE batched Adam call for all spectra in sweep ───────────────────
    if do_timing:
        t0 = time.perf_counter()
    params_batch = _fit_batch_adam(
        spec_d,
        _x_dev,
        p0_batch,
        grad_mask,
        max_iter=max_iter,
        tol=tol,
        aggressive_start_steps=aggressive_start_steps,
        aggressive_lr_mult=aggressive_lr_mult,
        aggressive_clip_norm=aggressive_clip_norm,
        aggressive_beta1=aggressive_beta1,
    )
    if do_timing:
        fit_adam_sec = _elapsed(t0)
    else:
        fit_adam_sec = 0.0

    # ── 6. Prune once, then match grouped by sweep level ─────────────────────
    params_batch = _prune_peaks_batch(params_batch, amp_threshold=amp_thr)
    params_batch = _deduplicate_peaks_batch(params_batch, min_spacing)

    stage_totals_shared = {
        'stack_to_device': stack_to_device_sec,
        'denoise': denoise_sec,
        'lor4_transform': lor4_transform_sec,
        'peak_vote': peak_vote_sec,
        'build_p0': build_p0_sec,
        'fit_adam': fit_adam_sec,
    }

    cursor = 0
    for pval, key, n_level in level_meta:
        if do_timing:
            t0 = time.perf_counter()

        level_stats = []
        for j in range(n_level):
            idx = cursor + j
            r = all_raw[idx]
            stats = _match_peaks(
                r[3], r[4], r[7], r[9],
                params_batch[idx],
                tolerance=ctr_tol,
                amp_threshold=amp_thr,
                x_arr=_ctx_x,
            )
            level_stats.append(stats)

        cursor += n_level
        all_results[key] = level_stats

        f1 = np.mean([s['f1'] for s in level_stats])
        pre = np.mean([s['precision'] for s in level_stats])
        rec = np.mean([s['recall'] for s in level_stats])
        srmse = np.nanmean([s['mean_shape_rmse'] for s in level_stats])
        print(f"  {str(pval):22s}  F1={f1:.3f}  Prec={pre:.3f}  Recall={rec:.3f}  ShapeRMSE={srmse:.3f}")

        if do_timing:
            prune_match_sec = _elapsed(t0)
            share = float(n_level) / float(max(1, B_total))
            level_timing = {
                'dataset': dataset_times.get(key, 0.0),
                'stack_to_device': stage_totals_shared['stack_to_device'] * share,
                'denoise': stage_totals_shared['denoise'] * share,
                'lor4_transform': stage_totals_shared['lor4_transform'] * share,
                'peak_vote': stage_totals_shared['peak_vote'] * share,
                'build_p0': stage_totals_shared['build_p0'] * share,
                'fit_adam': stage_totals_shared['fit_adam'] * share,
                'prune_match': prune_match_sec,
            }
            timing_per_level[key] = level_timing

            if profile:
                total_lvl = sum(level_timing.values())
                per_pixel = total_lvl / max(1, n_level)
                print(
                    "    timing[s] "
                    f"dataset={level_timing['dataset']:.3f}  "
                    f"stack={level_timing['stack_to_device']:.3f}  "
                    f"denoise={level_timing['denoise']:.3f}  "
                    f"xform={level_timing['lor4_transform']:.3f}  "
                    f"vote={level_timing['peak_vote']:.3f}  "
                    f"p0={level_timing['build_p0']:.3f}  "
                    f"fit={level_timing['fit_adam']:.3f}  "
                    f"match={level_timing['prune_match']:.3f}  "
                    f"total={total_lvl:.3f}  "
                    f"per_pixel={per_pixel:.4f}"
                )

    if do_timing:
        stage_names = [
            'dataset', 'stack_to_device', 'denoise', 'lor4_transform',
            'peak_vote', 'build_p0', 'fit_adam', 'prune_match'
        ]
        mean_stage = {
            s: float(np.mean([timing_per_level[k][s] for k in timing_per_level]))
            for s in stage_names
        }
        timing_dict = {
            'per_level_sec': timing_per_level,
            'mean_stage_sec': mean_stage,
            'mean_total_sec': float(sum(mean_stage.values())),
            'mean_per_pixel_sec': float(sum(mean_stage.values()) / max(1, n_samples)),
            'n_levels': int(len(timing_per_level)),
            'n_samples': int(n_samples),
        }

        if profile:
            print("\nSweep timing summary (mean seconds per level):")
            print(
                "  "
                f"dataset={mean_stage['dataset']:.3f}  "
                f"stack={mean_stage['stack_to_device']:.3f}  "
                f"denoise={mean_stage['denoise']:.3f}  "
                f"xform={mean_stage['lor4_transform']:.3f}  "
                f"vote={mean_stage['peak_vote']:.3f}  "
                f"p0={mean_stage['build_p0']:.3f}  "
                f"fit={mean_stage['fit_adam']:.3f}  "
                f"match={mean_stage['prune_match']:.3f}  "
                f"total={timing_dict['mean_total_sec']:.3f}  "
                f"per_pixel={timing_dict['mean_per_pixel_sec']:.4f}"
            )

        if return_timing:
            return all_results, timing_dict

    return all_results


# ── _fit_one: single-spectrum version used by _plot_sweep example panels ──────
def _fit_one(sample_wrapper, max_iter=2000, tol=1e-5, amp_threshold=1e-2,
             ctr_tol=15.0, min_spacing=7.0, response_threshold=0.02,
             min_scale_votes=6, max_peaks=200,
             aggressive_start_steps=60, aggressive_lr_mult=3.0,
             aggressive_clip_norm=3.0, aggressive_beta1=0.85):
    """Fit one SampleWrapper spectrum using the cached sweep context.

    Requires init_sweep_context() to have been called beforehand.

    Runs the full pipeline on a single spectrum:
      denoise → Lor4 CWT → cross-scale vote → initial guesses → batched Adam

    Parameters
    ----------
    sample_wrapper        : SampleWrapper from RamanDataset
    max_iter              : Adam iteration budget
    tol                   : convergence tolerance
    amp_threshold         : minimum fitted amplitude to retain a peak
    ctr_tol               : centre tolerance for GT/fitted peak matching (cm⁻¹)
    min_spacing           : minimum allowed spacing between initial-guess peaks
    response_threshold    : minimum CWT response to retain a candidate position
    min_scale_votes       : minimum number of wavelet scales that must vote for
                            a position before it is considered a peak candidate
    max_peaks             : maximum number of peaks to initialise
    aggressive_start_steps: number of steps to use the boosted LR at the start
    aggressive_lr_mult    : LR multiplier during the aggressive phase
    aggressive_clip_norm  : gradient clip norm during the aggressive phase
    aggressive_beta1      : Adam β₁ during the aggressive phase

    Returns
    -------
    stats    : dict from _match_peaks (precision, recall, f1, errors, ...)
    params   : (max_peaks * 4,) numpy float32 fitted parameters
    spec_d   : (n_pts,) numpy float32 denoised spectrum
    """
    s      = sample_wrapper
    spec_t = torch.as_tensor(_cpu(s.spectrum), dtype=torch.float32).to(_ctx_device)
    spec_d = _denoise_batch(spec_t.unsqueeze(0)).squeeze(0)

    resp      = _lor4_transform_batch(spec_d.unsqueeze(0)).squeeze(0)   # (n_widths, n_pts)
    n_widths, n_pts = resp.shape
    masks     = find_peaks_derivative_mask_batch(_x_dev, resp, min_height=0.0)
    vote      = masks.long().sum(dim=0)
    pmask     = (vote >= min_scale_votes) & (resp.max(dim=0).values > response_threshold)

    p0 = build_initial_guesses_from_derivative_mask(
        resp.unsqueeze(1), _ctx_sigmas.to(_ctx_device), _ctx_gammas[:1].to(_ctx_device),
        _x_dev, spec_d, pmask, max_peaks=max_peaks,
        min_spacing=min_spacing, scale_preference_fraction=0.8,
    )

    # Pad to max_peaks and use batched fitter (batch of 1 for consistency)
    npq      = max_peaks * 4
    n        = min(int((p0[:npq:4] > 0).sum().item()), max_peaks)
    p0_batch = torch.zeros(1, npq, device=_ctx_device)
    gmask    = torch.zeros(1, npq, device=_ctx_device)
    p0_batch[0, :min(p0.numel(), npq)] = p0[:npq]
    gmask[0, :n*4]    = 1.0

    params = _fit_batch_adam(spec_d.unsqueeze(0), _x_dev, p0_batch, gmask,
                             max_iter=max_iter, tol=tol,
                             aggressive_start_steps=aggressive_start_steps,
                             aggressive_lr_mult=aggressive_lr_mult,
                             aggressive_clip_norm=aggressive_clip_norm,
                             aggressive_beta1=aggressive_beta1).squeeze(0)
    params = prune_peaks(params, amp_threshold=amp_threshold)
    params = deduplicate_peaks(params, min_spacing)

    params_np = _cpu(params)
    stats = _match_peaks(s.centers, s.amplitudes, s.sigmas, s.gammas,
                         params_np, tolerance=ctr_tol, amp_threshold=amp_threshold,
                         x_arr=_ctx_x)
    return stats, params_np, _cpu(spec_d)


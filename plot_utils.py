"""
plot_utils.py
=============
Visualisation utilities for CASCADE spectral fits.

Functions
---------
init_plot_context               Cache x and widths; must be called before _plot_sweep.
plot_voigt_fit_res              Single-spectrum decomposition plot with residual panel.
plot_voigt_fit_compare_detailed GT vs. recovered comparison with peak matching.
plot_shape_rmse                 Per-pair shape-RMSE spectrum overview and bar chart.
_plot_sweep                     Journal-ready 3×2 parameter-sweep figure.

Helper classes / functions (internal)
--------------------------------------
HandlerRainbowLine  Custom legend handler that draws a gradient-coloured line.
_thin_ticks         Reduce tick density to at most max_ticks labels.
_style_ax           Apply consistent axis styling.
_add_violin         Single violin + jitter scatter.
_dual_violin        Two interleaved violin sets on shared axes.
_dual_y_violin      Two violin sets on separate y-axes (twin axes).

Usage
-----
Call init_sweep_context (tidytorch_utils) and init_plot_context (this module)
once before using _plot_sweep:

    from tidytorch_utils import init_sweep_context
    from plot_utils import init_plot_context, _plot_sweep

    init_sweep_context(x, sigmas, gammas, device, widths)
    init_plot_context(x, widths)

    results = _run_sweep("noise_std", [0.0, 0.01, 0.05])
    _plot_sweep("noise_std", [0.0, 0.01, 0.05], results)
"""

# Core
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import torch

from dataset_utils import voigt_peak, RamanDataset, SampleWrapper
from tidytorch_utils import _fit_one, _cpu

# ── Plot context — call init_plot_context() once before using _plot_sweep ─────
_plt_x      = None  # (n_pts,) wavenumber axis
_plt_widths = None  # peak-width priors for RamanDataset


def init_plot_context(x, widths):
    """Cache x and widths so _plot_sweep can use them without notebook globals.

    Call after init_sweep_context:
        init_plot_context(x, widths)
    """
    global _plt_x, _plt_widths
    import numpy as np
    _plt_x      = np.asarray(x, dtype=np.float32)
    _plt_widths = widths


class HandlerRainbowLine(HandlerBase):
    """Matplotlib legend handler that renders a legend entry as a rainbow gradient line.

    Pass an instance of this class in the ``handler_map`` argument of
    ``ax.legend()`` to render a proxy handle with a colour gradient:

        proxy = mlines.Line2D([], [], label="Component peaks")
        ax.legend(handles=[proxy],
                  handler_map={proxy: HandlerRainbowLine()})
    """
    def __init__(self, cmap="rainbow", n_segments=20, **kwargs):
        super().__init__(**kwargs)
        self.cmap = plt.get_cmap(cmap)
        self.n_segments = n_segments

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):

        lines = []
        for i in range(self.n_segments):
            xi0 = x0 + width * i / self.n_segments
            xi1 = x0 + width * (i + 1) / self.n_segments

            line = mlines.Line2D(
                [xi0, xi1],
                [y0 + height / 2, y0 + height / 2],
                color=self.cmap(i / self.n_segments),
                linewidth=2,
                transform=trans,
            )
            lines.append(line)

        return lines

        
def plot_voigt_fit_res(x, y_true, params, peaks_dict=None, title=None):
    """Plot a Voigt peak decomposition with a residual panel.

    Draws:
      - Upper panel: raw spectrum, model sum (dashed red), and individual
        component peaks (rainbow-coloured via HandlerRainbowLine).
      - Lower panel: residual = y_true - model_sum, filled grey.

    Also prints RMSE and R² to stdout.

    Parameters
    ----------
    x         : (n_pts,) wavenumber axis
    y_true    : (n_pts,) measured spectrum
    params    : (n_peaks * 4,) flat parameter vector [amp, ctr, sig, gam, ...]
    peaks_dict: unused; kept for API compatibility
    title     : figure title string
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    def _np(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    x      = _np(x)
    y_true = _np(y_true)
    params = _np(params)

    n_peaks = len(params) // 4
    n_pos_peaks = sum(1 for i in range(n_peaks) if params[i*4 ] > 0)
    p = params[:n_peaks * 4].reshape(-1, 4)  # reshape (N, 4): amp, ctr, sigma, gam

    # Generate individual peaks
    y_peaks = []
    for amp, ctr, sigma, gamma in p:
        if amp > 0:
            y_peaks.append(voigt_peak(x, ctr, amp, sigma, gamma))

    y_peaks = np.array(y_peaks)
    print("y_peaks shape:", y_peaks.shape)
    model_sum = y_peaks.sum(axis=0)
    residual = y_true - model_sum

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(x, y_true, 'black', lw=2, alpha=0.6, label='Original')
    ax1.plot(x, model_sum, color='red', lw=2, linestyle= '--', label='Sum model')
    for i, yp in enumerate(y_peaks):
        ax1.plot(x, yp, alpha=1, lw=0.5)
    
    # Calculate fit quality
    rmse = np.sqrt(np.mean(residual**2))
    r2 = 1 - np.sum(residual**2) / np.sum((y_true - np.mean(y_true))**2)
    
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.set_title(f"{title} (n={n_pos_peaks})", fontsize=14)
    
    # Residual plot
    ax2.plot(x, residual, 'gray', lw=1)
    ax2.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax2.fill_between(x, residual, alpha=0.3, color='gray')
    ax2.set_xlabel("Raman shift (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=10)

    # Create dummy handles
    data_line = mlines.Line2D([], [], color="black", lw=1.5, label="Data")
    fit_line = mlines.Line2D([], [], color="red", lw=1.5, linestyle= "--", label="Voigt fit")
    rainbow_proxy = mlines.Line2D([], [], label="Component Peaks")

    ax1.legend(
        handles=[data_line, fit_line, rainbow_proxy],
        handler_map={rainbow_proxy: HandlerRainbowLine()},
        frameon=False,
    )

    print(f"RMSE: {rmse:.4f}   R²: {r2:.4f}")
    ax1.legend()
    plt.tight_layout()
    plt.show()
    # return fig

def plot_voigt_fit_compare_detailed(x, y_true, gt_params, rec_params, title='Voigt Fit Breakdown', 
                            amp_threshold=1e-6, center_tolerance=10.0):
    """
    Plot Voigt/pseudo-Voigt fit comparing ground truth vs reconstruction with peak matching.
    
    Parameters:
    -----------
    x : array
        Wavenumber/frequency axis
    y_true : array
        True/measured spectrum
    gt_params : array
        Ground truth parameters [amp0, ctr0, sigma0, gamma0, amp1, ctr1, ...]
    rec_params : array
        Reconstructed parameters
    title : str
        Plot title
    amp_threshold : float
        Minimum amplitude to plot a peak
    center_tolerance : float
        Max distance between centers to consider peaks as matched
    """
    def _np(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    x         = _np(x)
    y_true    = _np(y_true)
    gt_params = _np(gt_params)
    rec_params = _np(rec_params)
    
    n_peaks_gt = len(gt_params) // 4
    n_peaks_rec = len(rec_params) // 4
    
    # Reshape to (n_peaks, 4): [amp, center, sigma, gamma]
    p_gt = gt_params[:n_peaks_gt * 4].reshape(-1, 4)
    p_rec = rec_params[:n_peaks_rec * 4].reshape(-1, 4)
    
    # Filter out peaks with zero or very small amplitudes
    valid_peaks_gt = p_gt[np.abs(p_gt[:, 0]) > amp_threshold]
    valid_peaks_rec = p_rec[np.abs(p_rec[:, 0]) > amp_threshold]
    
    n_valid_gt = len(valid_peaks_gt)
    n_valid_rec = len(valid_peaks_rec)
    
    print(f"Ground Truth: {n_valid_gt} valid peaks (from {n_peaks_gt} total)")
    print(f"Reconstructed: {n_valid_rec} valid peaks (from {n_peaks_rec} total)")
    
    # Match peaks between GT and reconstruction
    matched_pairs = []
    unmatched_gt = []
    unmatched_rec = list(range(n_valid_rec))
    
    for i, (amp_gt, ctr_gt, sig_gt, gam_gt) in enumerate(valid_peaks_gt):
        # Find closest reconstructed peak
        distances = np.abs(valid_peaks_rec[:, 1] - ctr_gt)
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]
        
        if closest_dist < center_tolerance and closest_idx in unmatched_rec:
            # Match found
            matched_pairs.append((i, closest_idx, closest_dist))
            unmatched_rec.remove(closest_idx)
        else:
            # No match - missing peak
            unmatched_gt.append(i)
    
    n_matched = len(matched_pairs)
    n_missing = len(unmatched_gt)
    n_extra = len(unmatched_rec)
    
    print(f"\nPeak Matching (tolerance={center_tolerance}):")
    print(f"  Matched: {n_matched}")
    print(f"  Missing (in GT, not in rec): {n_missing}")
    print(f"  Extra (in rec, not in GT): {n_extra}")
    
    # Generate individual peaks
    y_peaks_gt = []
    y_peaks_rec = []
    
    for i, (amp, ctr, sigma, gamma) in enumerate(valid_peaks_gt):
        y_peak = _np(voigt_peak(x, ctr, amp, sigma, gamma))
        y_peaks_gt.append(y_peak)
    
    for i, (amp, ctr, sigma, gamma) in enumerate(valid_peaks_rec):
        if amp > amp_threshold:
            y_peak = _np(voigt_peak(x, ctr, amp, sigma, gamma))
            y_peaks_rec.append(y_peak)
    
    if len(y_peaks_gt) == 0:
        y_peaks_gt = [np.zeros_like(x)]
    if len(y_peaks_rec) == 0:
        y_peaks_rec = [np.zeros_like(x)]
    
    y_peaks_gt = np.array(y_peaks_gt)
    y_peaks_rec = np.array(y_peaks_rec)
    
    model_sum_gt = y_peaks_gt.sum(axis=0)
    model_sum_rec = y_peaks_rec.sum(axis=0)
    
    # Calculate fit quality
    residual = y_true - model_sum_rec
    rmse = np.sqrt(np.mean(residual**2))
    r2 = 1 - np.sum(residual**2) / np.sum((y_true - np.mean(y_true))**2)
    
    # Calculate peak-to-peak errors for matched pairs
    peak_errors = []
    for gt_idx, rec_idx, dist in matched_pairs:
        gt_peak = valid_peaks_gt[gt_idx]
        rec_peak = valid_peaks_rec[rec_idx]
        
        amp_error = np.abs(gt_peak[0] - rec_peak[0])
        ctr_error = np.abs(gt_peak[1] - rec_peak[1])
        sig_error = np.abs(gt_peak[2] - rec_peak[2])
        gam_error = np.abs(gt_peak[3] - rec_peak[3])
        
        # Relative errors
        amp_rel_error = amp_error / (np.abs(gt_peak[0]) + 1e-10) * 100
        
        peak_errors.append({
            'gt_idx': gt_idx,
            'rec_idx': rec_idx,
            'center_dist': dist,
            'amp_error': amp_error,
            'amp_rel_error': amp_rel_error,
            'ctr_error': ctr_error,
            'sig_error': sig_error,
            'gam_error': gam_error
        })
    
    # Plot
    fig = plt.figure(figsize=(14, 12), dpi = 300)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])  # Main plot
    ax2 = fig.add_subplot(gs[1])  # Residual
    # ax3 = fig.add_subplot(gs[2])  # Peak matching stats
    # ax4 = fig.add_subplot(gs[3])  # Peak-to-peak errors
    
    # Main plot
    ax1.plot(x, y_true, 'k', lw=2, alpha=0.7, label='Data', zorder=10)
    ax1.plot(x, model_sum_gt, color='blue', lw=2, linestyle='-', 
             label=f'GT Sum ({n_valid_gt} peaks)', alpha=0.7, zorder=9)
    ax1.plot(x, model_sum_rec, color='red', lw=2, linestyle='--', 
             label=f'Rec Sum ({n_valid_rec} peaks)', alpha=0.7, zorder=8)
    
    # Plot matched peaks (green)
    gt_label_added = False
    rec_label_added = False

    for gt_idx, rec_idx, _ in matched_pairs:
        ax1.plot(x, y_peaks_gt[gt_idx], alpha=0.3, lw=1, color='green', label = "Ground Truth" if not gt_label_added else None)
        ax1.plot(x, y_peaks_rec[rec_idx], alpha=0.3, lw=1, color='green', linestyle='--', label = "Recovered" if not rec_label_added else None)
        gt_label_added = True
        rec_label_added = True
    
    # Plot missing peaks (red/orange - in GT but not reconstructed)
    for gt_idx in unmatched_gt:
        gt_peak = valid_peaks_gt[gt_idx]
        ax1.plot(x, y_peaks_gt[gt_idx], alpha=0.5, lw=2, color='orange', 
                label=f'Missing: {gt_peak[1]:.1f}')
        ax1.axvline(gt_peak[1], color='orange', linestyle=':', alpha=0.3)
    
    # Plot extra peaks (purple - in rec but not in GT)
    for rec_idx in unmatched_rec:
        rec_peak = valid_peaks_rec[rec_idx]
        ax1.plot(x, y_peaks_rec[rec_idx], alpha=0.5, lw=2, color='purple',
                label=f'Extra: {rec_peak[1]:.1f}')
        ax1.axvline(rec_peak[1], color='purple', linestyle=':', alpha=0.3)
    
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.set_title(f"{title}\nRMSE: {rmse:.4f}, R²: {r2:.4f}\n"
                  f"Matched: {n_matched}, Missing: {n_missing}, Extra: {n_extra}", 
                  fontsize=14)
    
    ax1.legend(fontsize=9, ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2.plot(x, residual, 'gray', lw=1, label='Residual')
    ax2.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax2.fill_between(x, residual, alpha=0.3, color='gray')
    
    # Mark regions with high residuals (potential missing peaks)
    high_residual_threshold = 3 * np.std(residual)
    high_res_indices = np.abs(residual) > high_residual_threshold
    if np.any(high_res_indices):
        ax2.scatter(x[high_res_indices], residual[high_res_indices], 
                   color='red', s=30, alpha=0.5, label='High Residual', zorder=10)
    
    ax2.set_ylabel("Residual", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    return fig, (ax1, ax2), {
        'matched': matched_pairs,
        'missing_gt': unmatched_gt,
        'extra_rec': unmatched_rec,
        'peak_errors': peak_errors,
        'rmse': rmse,
        'r2': r2
    }


# ---------------------------------------------------------------------------
# Sweep plot — journal-ready, symmetric 3×2 layout
#
# Layout:
#   Row 0:  Precision+Recall (shared axes)     |  F1
#   Row 1:  Amp + Center error (dual y-axis)   |  Shape RMSE
#   Row 2:  Example spectrum (low noise)       |  Example spectrum (high noise)
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_JOURNAL_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "lines.linewidth":   1.4,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}


_AMP_COLOR         = "#2ca02c"
_PREC_COLOR        = "#4878CF"
_REC_COLOR         = "#E07B39"
_F1_COLOR          = "#F03D07"
_CTR_COLOR         = "#B472DA"
_RMSE_COLOR        = "#07E8F8"
_VIOLIN_FACE_ALPHA = 0.25
_DOT_ALPHA         = 0.35
_DOT_SIZE          = 10
_GT_COLOR          = "#078039"
_FIT_COLOR         = "#ee8625"
_SPEC_COLOR        = "black"
_DENOISED_COLOR    = "#888888"

# == Helpers ===================================================================

def _thin_ticks(ax, positions, labels, max_ticks=8):
    n = len(labels)
    if n <= max_ticks:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    else:
        idx = sorted(set(np.linspace(0, n - 1, max_ticks, dtype=int)))
        ax.set_xticks([positions[i] for i in idx])
        ax.set_xticklabels([labels[i] for i in idx])


def _style_ax(ax, ylabel, xlabel, ylim, title, labels, xs, max_ticks):
    _thin_ticks(ax, xs, labels, max_ticks=max_ticks)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=6)
    ax.grid(True, alpha=0.20, axis="y", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_violin(ax, data_by_level, labels, ylabel, xlabel="",
                ylim=None, color="#4878CF", max_ticks=8, title=""):
    xs    = list(range(len(data_by_level)))
    clean = [[v for v in d if not np.isnan(v)] for d in data_by_level]
    vdata = [c if len(c) >= 2 else (c * 2 if c else [0.0, 0.0]) for c in clean]

    parts = ax.violinplot(vdata, positions=xs, showmedians=False,
                          showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_edgecolor(color)
        pc.set_alpha(_VIOLIN_FACE_ALPHA); pc.set_linewidth(0.6)

    for i, vals in enumerate(clean):
        if not vals:
            continue
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax.vlines(i, q25, q75, color=color, lw=2.2, zorder=3)
        ax.scatter([i], [med], color="white", s=22, zorder=4,
                   edgecolors=color, linewidths=1.2)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(clean):
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, s=_DOT_SIZE, alpha=_DOT_ALPHA,
                   zorder=2, linewidths=0, rasterized=True)

    _style_ax(ax, ylabel, xlabel, ylim, title, labels, xs, max_ticks)


def _dual_violin(ax, data_a, data_b, labels, color_a, color_b,
                 label_a, label_b, ylabel, xlabel="", ylim=None,
                 max_ticks=8, title=""):
    """Two interleaved violin sets on shared axes, offset left/right."""
    xs     = np.arange(len(data_a))
    offset = 0.18
    width  = 0.55

    for data, col, sign in [
        (data_a, color_a, -1),
        (data_b, color_b, +1),
    ]:
        clean = [[v for v in d if not np.isnan(v)] for d in data]
        vdata = [c if len(c) >= 2 else (c * 2 if c else [0.0, 0.0])
                 for c in clean]
        pos   = xs + sign * offset

        parts = ax.violinplot(vdata, positions=pos, showmedians=False,
                              showextrema=False, widths=width)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_edgecolor(col)
            pc.set_alpha(_VIOLIN_FACE_ALPHA); pc.set_linewidth(0.6)

        rng = np.random.default_rng(0 if sign == -1 else 42)
        for i, vals in enumerate(clean):
            if not vals:
                continue
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(pos[i], q25, q75, color=col, lw=2.0, zorder=3)
            ax.scatter([pos[i]], [med], color="white", s=20, zorder=4,
                       edgecolors=col, linewidths=1.1)
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), pos[i]) + jitter, vals,
                       color=col, s=_DOT_SIZE, alpha=_DOT_ALPHA,
                       zorder=2, linewidths=0, rasterized=True)

    _style_ax(ax, ylabel, xlabel, ylim, title, labels, list(xs), max_ticks)

    ax.legend(
        handles=[
            Line2D([], [], color=color_a, lw=3, alpha=0.6, label=label_a),
            Line2D([], [], color=color_b, lw=3, alpha=0.6, label=label_b),
        ],
        loc="lower left", frameon=False, fontsize=8, handlelength=1.2,
    )


def _dual_y_violin(ax, data_left, data_right, labels,
                   color_left, color_right,
                   ylabel_left, ylabel_right,
                   xlabel="", title="", max_ticks=8):
    """
    Two violin distributions on the SAME x-axis but DIFFERENT y-axes.
    Left data uses the primary (left) y-axis; right data uses a twinned
    (right) y-axis. Violins are offset left/right to avoid overlap.
    """
    xs     = np.arange(len(data_left))
    offset = 0.18
    width  = 0.50

    # -- Left y-axis -----------------------------------------------------------
    clean_l = [[v for v in d if not np.isnan(v)] for d in data_left]
    vdata_l = [c if len(c) >= 2 else (c * 2 if c else [0.0, 0.0])
               for c in clean_l]
    pos_l   = xs - offset

    parts_l = ax.violinplot(vdata_l, positions=pos_l, showmedians=False,
                            showextrema=False, widths=width)
    for pc in parts_l["bodies"]:
        pc.set_facecolor(color_left); pc.set_edgecolor(color_left)
        pc.set_alpha(_VIOLIN_FACE_ALPHA); pc.set_linewidth(0.6)

    rng_l = np.random.default_rng(0)
    for i, vals in enumerate(clean_l):
        if not vals:
            continue
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax.vlines(pos_l[i], q25, q75, color=color_left, lw=2.0, zorder=3)
        ax.scatter([pos_l[i]], [med], color="white", s=20, zorder=4,
                   edgecolors=color_left, linewidths=1.1)
        jitter = rng_l.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), pos_l[i]) + jitter, vals,
                   color=color_left, s=_DOT_SIZE, alpha=_DOT_ALPHA,
                   zorder=2, linewidths=0, rasterized=True)

    ax.set_ylabel(ylabel_left, color=color_left)
    ax.tick_params(axis="y", colors=color_left)
    ax.spines["left"].set_color(color_left)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, axis="y", linewidth=0.5)

    # -- Right y-axis (twinned) ------------------------------------------------
    ax2 = ax.twinx()

    clean_r = [[v for v in d if not np.isnan(v)] for d in data_right]
    vdata_r = [c if len(c) >= 2 else (c * 2 if c else [0.0, 0.0])
               for c in clean_r]
    pos_r   = xs + offset

    parts_r = ax2.violinplot(vdata_r, positions=pos_r, showmedians=False,
                             showextrema=False, widths=width)
    for pc in parts_r["bodies"]:
        pc.set_facecolor(color_right); pc.set_edgecolor(color_right)
        pc.set_alpha(_VIOLIN_FACE_ALPHA); pc.set_linewidth(0.6)

    rng_r = np.random.default_rng(42)
    for i, vals in enumerate(clean_r):
        if not vals:
            continue
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax2.vlines(pos_r[i], q25, q75, color=color_right, lw=2.0, zorder=3)
        ax2.scatter([pos_r[i]], [med], color="white", s=20, zorder=4,
                    edgecolors=color_right, linewidths=1.1)
        jitter = rng_r.uniform(-0.08, 0.08, len(vals))
        ax2.scatter(np.full(len(vals), pos_r[i]) + jitter, vals,
                    color=color_right, s=_DOT_SIZE, alpha=_DOT_ALPHA,
                    zorder=2, linewidths=0, rasterized=True)

    ax2.set_ylabel(ylabel_right, color=color_right)
    ax2.tick_params(axis="y", colors=color_right)
    ax2.spines["right"].set_color(color_right)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # -- Shared x-axis formatting ----------------------------------------------
    _thin_ticks(ax, list(xs), labels, max_ticks=max_ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=6)

    # -- Legend ----------------------------------------------------------------
    ax.legend(
        handles=[
            Line2D([], [], color=color_left,  lw=3, alpha=0.6, label=ylabel_left),
            Line2D([], [], color=color_right, lw=3, alpha=0.6, label=ylabel_right),
        ],
        loc="upper left", frameon=False, fontsize=8, handlelength=1.2,
    )

    return ax2


def _plot_sweep(param_name, param_values, all_results,
                noise_fixed=0.01, sep_fixed=(1.0, 1.5),
                example_idx=(0, -1), example_seed=123,
                max_iter=2000, tol=1e-5, amp_thr=1e-2,
                save_path=None):
    """Generate a journal-ready 3×2 figure summarising a parameter sweep.

    Requires both init_sweep_context() (tidytorch_utils) and
    init_plot_context() (this module) to have been called beforehand.

    Layout
    ------
    Row 0: Precision & Recall (dual violin)  |  F1 score (single violin)
    Row 1: Amp + Centre error (dual y-axis)  |  Shape RMSE (single violin)
    Row 2: Example spectrum at param_values[example_idx[0]]  |  … [example_idx[1]]

    Parameters
    ----------
    param_name    : "noise_std" or "separability" (used for x-axis labelling
                    and dataset construction inside example panels)
    param_values  : list of parameter values that were swept; must match the
                    keys used in all_results
    all_results   : dict mapping each param value (or str(value)) to a list of
                    metric dicts as returned by _match_peaks / _run_sweep
    noise_fixed   : noise_std held constant when sweeping separability
    sep_fixed     : separability_range held constant when sweeping noise
    example_idx   : indices into param_values for the two example spectrum panels;
                    negative indices count from the end
    example_seed  : RNG seed for the example RamanDataset samples
    max_iter      : Adam iteration budget for example-panel fitting
    tol           : convergence tolerance for example-panel fitting
    amp_thr       : minimum fitted amplitude to show a peak in example panels
    save_path     : if not None, save the figure to this path before showing
    """

    with mpl.rc_context(_JOURNAL_RC):

        keys   = [pv if param_name == "noise_std" else str(pv)
                  for pv in param_values]
        labels = [str(pv) for pv in param_values]
        raw    = {m: [[s[m] for s in all_results[k]] for k in keys]
                  for m in ("precision", "recall", "f1",
                            "mean_amp_err", "mean_ctr_err",
                            "mean_shape_rmse")}

        xlabel_map = {"noise_std": "Noise std", "separability": "Separability"}
        xlabel     = xlabel_map.get(param_name,
                                    param_name.replace("_", " ").capitalize())

        ex_idx = [len(param_values) + i if i < 0 else i for i in example_idx]
        n_ex   = len(ex_idx)

        # == Symmetric 2x3 via subfigures ======================================
        fig = plt.figure(figsize=(15, 9.0), dpi = 300)
        subfigs = fig.subfigures(
            3, 1,
            height_ratios=[1, 1, 1.15],
            hspace=0.08,
        )

        # =====================================================================
        # Row 0:  Precision+Recall  |  F1
        # =====================================================================
        axes_r0 = subfigs[0].subplots(1, 2, gridspec_kw={"wspace": 0.35})

        _dual_violin(
            axes_r0[0], raw["precision"], raw["recall"], labels,
            color_a=_PREC_COLOR, color_b=_REC_COLOR,
            label_a="Precision", label_b="Recall",
            ylabel="Score", xlabel=xlabel,
            ylim=(0, 1.08), title="Precision & Recall",
        )

        _add_violin(
            axes_r0[1], raw["f1"], labels,
            ylabel="Score", xlabel=xlabel,
            ylim=(0, 1.08), color=_F1_COLOR, title="F1 score",
        )

        # =====================================================================
        # Row 1:  Amp + Center (dual y)  |  Shape RMSE
        # =====================================================================
        axes_r1 = subfigs[1].subplots(1, 2, gridspec_kw={"wspace": 0.45})

        _dual_y_violin(
            axes_r1[0], raw["mean_amp_err"], raw["mean_ctr_err"], labels,
            color_left=_AMP_COLOR, color_right=_CTR_COLOR,
            ylabel_left="Rel. amplitude error",
            ylabel_right=u"Center error (cm\u207b\u00b9)",
            xlabel=xlabel,
            title="Peak parameter errors",
        )

        _add_violin(
            axes_r1[1], raw["mean_shape_rmse"], labels,
            ylabel="Shape RMSE (norm.)", xlabel=xlabel,
            color=_RMSE_COLOR, title="Shape RMSE",
        )

        # =====================================================================
        # Row 2:  2 example spectra (full width)
        # =====================================================================
        axes_r2 = subfigs[2].subplots(1, n_ex, gridspec_kw={"wspace": 0.30})
        if n_ex == 1:
            axes_r2 = [axes_r2]

        for i, idx in enumerate(ex_idx):
            pval  = param_values[idx]
            ds_kw = dict(noise_std=noise_fixed,
                         separability_range=sep_fixed)
            if param_name == "noise_std":
                ds_kw["noise_std"] = pval
            else:
                ds_kw["separability_range"] = pval

            ds_ex = RamanDataset(x=_plt_x, n_peaks=(8, 14), widths=_plt_widths,
                                 n_samples=1, seed=example_seed, **ds_kw)
            s_ex  = SampleWrapper(ds_ex[0])
            stats, params_ex, spec_d = _fit_one(
                s_ex, max_iter=max_iter, tol=tol, amp_threshold=amp_thr)

            ax = axes_r2[i]

            ax.plot(_plt_x, _cpu(s_ex.spectrum), color=_SPEC_COLOR,
                    lw=1.6, alpha=0.75)
            ax.plot(_plt_x, spec_d, color=_DENOISED_COLOR,
                    lw=0.9, alpha=0.5, linestyle="--")

            for amp, ctr, sig, gam in zip(
                _cpu(s_ex.amplitudes), _cpu(s_ex.centers),
                _cpu(s_ex.sigmas),     _cpu(s_ex.gammas),
            ):
                if amp > 0:
                    ax.plot(_plt_x, voigt_peak(_plt_x, ctr, amp, sig, gam),
                            lw=1.2, color=_GT_COLOR, alpha=0.85)

            for amp, ctr, sig, gam in params_ex.reshape(-1, 4):
                if amp > amp_thr:
                    ax.plot(_plt_x, voigt_peak(_plt_x, ctr, amp, sig, gam),
                            lw=1.2, color=_FIT_COLOR, alpha=0.75,
                            linestyle="--")

            ctrs = _cpu(s_ex.centers)
            span = max(250, (ctrs.max() - ctrs.min()) * 0.6)
            ax.set_xlim(ctrs.mean() - span, ctrs.mean() + span)

            if param_name == "noise_std":
                lbl = str(pval)
            else:
                lbl = f"{pval[0]:.1f}-{pval[1]:.1f}"

            ax.set_title(
                f"{param_name} = {lbl}   |   "
                f"F1 = {stats['f1']:.2f}   |   "
                f"Shape RMSE = {stats['mean_shape_rmse']:.3f}\n"
                f"TP = {stats['tp']}    FP = {stats['fp']}    FN = {stats['fn']}",
                fontsize=8, linespacing=1.35, pad=8,
            )
            ax.set_xlabel(u"Raman shift (cm\u207b\u00b9)")
            ax.set_ylabel("Intensity" if i == 0 else "")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # -- Spectra legend ----------------------------------------------------
        legend_handles = [
            Line2D([], [], color=_GT_COLOR,      lw=1.5,                  label="GT peaks"),
            Line2D([], [], color=_FIT_COLOR,      lw=1.5, linestyle="--", label="Fitted peaks"),
            Line2D([], [], color=_SPEC_COLOR,     lw=1.5,                  label="Spectrum"),
            Line2D([], [], color=_DENOISED_COLOR, lw=1.0, linestyle="--", label="Denoised"),
        ]
        subfigs[2].legend(
            handles=legend_handles, loc="lower center",
            ncol=4, bbox_to_anchor=(0.5, -0.04),
            frameon=False, columnspacing=1.5,
        )

        # -- n annotation ------------------------------------------------------
        n_per = len(next(iter(all_results.values())))
        fig.text(0.99, 1.005, f"n = {n_per} per level",
                 ha="right", va="bottom", fontsize=7.5,
                 fontstyle="italic", transform=fig.transFigure)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.15)
            print(f"Saved -> {save_path}")
        plt.show()
# ── Peak Shape RMSE for matched pairs ─────────────────────────────────────────

def plot_shape_rmse(x, y_true, gt_params, rec_params,
                    amp_threshold=1e-5, center_tolerance=15.0,
                    title="Peak Shape RMSE"):
    """
    For each matched GT/fitted peak pair, compute the normalised shape RMSE:
        shape_rmse = sqrt(mean((wf_gt - wf_fit)^2)) / sqrt(mean(wf_gt^2))
    and produce four panels:
      (A) Spectrum overview with each peak coloured by its RMSE
      (B) Bar chart of RMSE per pair, sorted by peak centre
      (C) Histogram of RMSE values
      (D) RMSE vs GT amplitude   (E) RMSE vs peak centre
    """
    def _np(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    x          = _np(x)
    y_true     = _np(y_true)
    gt_params  = _np(gt_params)
    rec_params = _np(rec_params)

    p_gt  = gt_params [:len(gt_params)  // 4 * 4].reshape(-1, 4)
    p_rec = rec_params[:len(rec_params) // 4 * 4].reshape(-1, 4)

    gt_valid  = p_gt [np.abs(p_gt [:, 0]) > amp_threshold]
    rec_valid = p_rec[np.abs(p_rec[:, 0]) > amp_threshold]

    # ── Greedy nearest-centre matching (same logic as plot_voigt_fit_compare_detailed) ─
    matched_pairs = []
    unmatched_rec = list(range(len(rec_valid)))

    for i, (_, ctr_gt, _, _) in enumerate(gt_valid):
        if len(rec_valid) == 0:
            break
        dists    = np.abs(rec_valid[:, 1] - ctr_gt)
        best_idx = int(np.argmin(dists))
        if dists[best_idx] < center_tolerance and best_idx in unmatched_rec:
            matched_pairs.append((i, best_idx))
            unmatched_rec.remove(best_idx)

    n_matched = len(matched_pairs)
    print(f"Matched: {n_matched}  |  Missing: {len(gt_valid) - n_matched}"
          f"  |  Extra: {len(rec_valid) - n_matched}")

    if n_matched == 0:
        print("No matched pairs — nothing to plot.")
        return

    # ── Compute shape RMSE ────────────────────────────────────────────────────
    rmses, centers, amps_gt = [], [], []

    for gt_idx, rec_idx in matched_pairs:
        gt_p  = gt_valid [gt_idx]
        rec_p = rec_valid[rec_idx]
        wf_gt  = voigt_peak(x, gt_p [1], gt_p [0], gt_p [2], gt_p [3])
        wf_fit = voigt_peak(x, rec_p[1], rec_p[0], rec_p[2], rec_p[3])
        shape_rmse = (np.sqrt(np.mean((wf_gt - wf_fit) ** 2))
                      / (np.sqrt(np.mean(wf_gt ** 2)) + 1e-12))
        rmses  .append(shape_rmse)
        centers.append(gt_p[1])
        amps_gt.append(gt_p[0])

    rmses   = np.array(rmses)
    centers = np.array(centers)
    amps_gt = np.array(amps_gt)

    # ── Build plot ────────────────────────────────────────────────────────────
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0, vmax=min(1.0, rmses.max() * 1.1 + 1e-9))

    fig = plt.figure(figsize=(16, 13), dpi=300)
    gs  = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)

    # (A) Spectrum overview ────────────────────────────────────────────────────
    ax_spec = fig.add_subplot(gs[0, :])
    ax_spec.plot(x, y_true, 'k', lw=1.5, alpha=0.55, label='Spectrum', zorder=10)

    for i, (gt_idx, rec_idx) in enumerate(matched_pairs):
        gt_p  = gt_valid [gt_idx]
        rec_p = rec_valid[rec_idx]
        wf_gt  = voigt_peak(x, gt_p [1], gt_p [0], gt_p [2], gt_p [3])
        wf_fit = voigt_peak(x, rec_p[1], rec_p[0], rec_p[2], rec_p[3])
        col = cmap(norm(rmses[i]))
        ax_spec.plot(x, wf_gt,  color=col, lw=1.4, alpha=0.85)
        ax_spec.plot(x, wf_fit, color=col, lw=1.4, alpha=0.85, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_spec, label='Shape RMSE', pad=0.01)
    from matplotlib.lines import Line2D
    ax_spec.legend(handles=[
        Line2D([0],[0], color='k',    lw=2,                  label='Spectrum'),
        Line2D([0],[0], color='gray', lw=1.4,                label='GT peaks (solid)'),
        Line2D([0],[0], color='gray', lw=1.4, linestyle='--', label='Fitted peaks (dashed)'),
    ], fontsize=9)
    ax_spec.set_title(f"{title} — peaks coloured by shape RMSE", fontsize=11)
    ax_spec.set_xlabel("Raman shift"); ax_spec.set_ylabel("Intensity")
    ax_spec.grid(True, alpha=0.2)

    # (B) Bar chart sorted by centre ──────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    order  = np.argsort(centers)
    bar_x  = np.arange(n_matched)
    ax_bar.bar(bar_x, rmses[order],
               color=cmap(norm(rmses[order])), edgecolor='k', linewidth=0.4)
    ax_bar.axhline(np.median(rmses), color='steelblue', lw=1.5, linestyle='--',
                   label=f'Median {np.median(rmses):.3f}')
    ax_bar.axhline(np.mean(rmses),   color='crimson',   lw=1.5, linestyle=':',
                   label=f'Mean   {np.mean(rmses):.3f}')
    tick_step = max(1, n_matched // 10)
    ax_bar.set_xticks(bar_x[::tick_step])
    ax_bar.set_xticklabels(
        [f"{centers[order[i]]:.0f}" for i in range(0, n_matched, tick_step)],
        rotation=45, ha='right', fontsize=7)
    ax_bar.set_xlabel("Peak centre (sorted, cm⁻¹)"); ax_bar.set_ylabel("Shape RMSE")
    ax_bar.set_title("Shape RMSE per matched pair"); ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.25, axis='y')
    plt.colorbar(sm, ax=ax_bar, label='Shape RMSE')

    plt.suptitle(
        f"Peak Shape RMSE  |  {n_matched} matched pairs  |  "
        f"mean={np.mean(rmses):.3f}  median={np.median(rmses):.3f}  max={np.max(rmses):.3f}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()




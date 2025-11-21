import numpy as np
from numpy.polynomial.legendre import leggauss


# =======================
# KLE
# =======================
def compute_cov_integral(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Approximate the covariance kernel of a 1D signal using
    a sliding window of length `window_size`.
    """
    N = len(signal)
    X = np.array([signal[i:N - window_size + i] for i in range(window_size)]).T
    centered_X = X - np.mean(X, axis=0, keepdims=True)
    return (centered_X.T @ centered_X) / max(1, centered_X.shape[0])


def compute_kle_coefficients(
    signal: np.ndarray,
    window_size: int,
    quad_order: int,
) -> np.ndarray:
    """
    Perform a Karhunen–Loève expansion (KLE) of a 1D signal.
    """
    N = len(signal)
    X = np.array([signal[i:N - window_size + i] for i in range(window_size)]).T

    # covariance matrix and eigen-decomposition
    Rxx = compute_cov_integral(signal, window_size)
    evals, evecs = np.linalg.eigh(Rxx)
    evecs = evecs[:, np.argsort(-evals)]  # sort by descending eigenvalue

    # Gauss–Legendre quadrature nodes and weights on [0, window_size - 1]
    qx, qw = leggauss(quad_order)
    qx = 0.5 * (qx + 1.0) * (window_size - 1)
    qw = 0.5 * (window_size - 1) * qw

    # interpolate samples and basis to quadrature nodes
    centered_X = X - np.mean(X, axis=0, keepdims=True)
    interp_sample = np.array([
        np.interp(qx, np.arange(window_size), centered_X[i])
        for i in range(centered_X.shape[0])
    ])
    interp_basis = np.array([
        np.interp(qx, np.arange(window_size), evecs[:, i])
        for i in range(evecs.shape[1])
    ])

    # integral kernel: sum_t sample(t) * basis(t) * w(t)
    kernel = (
        qw[np.newaxis, np.newaxis, :]
        * interp_sample[:, np.newaxis, :]
        * interp_basis[np.newaxis, :, :]
    )
    return np.sum(kernel, axis=-1)  # [T', window_size]


# =======================
# Sliding-window PCC
# =======================
def sliding_pearson_corr(
    sig1: np.ndarray,
    sig2: np.ndarray,
    window_length: int,
    step_size: int,
):
    """
    Sliding-window Pearson correlation between two signals.

    Parameters
    ----------
    sig1, sig2 : np.ndarray
        Input time series (same length after any pre-processing).
    window_length : int
        Length of each sliding window.
    step_size : int
        Step size between successive windows.

    Returns
    -------
    starts : np.ndarray
        Start index of each window.
    centers : np.ndarray
        Center index of each window.
    r_values : np.ndarray
        Pearson correlation coefficient in each window.
    """
    centers, r_values, starts = [], [], []
    for start in range(0, len(sig1) - window_length + 1, step_size):
        end = start + window_length
        a, b = sig1[start:end], sig2[start:end]
        if np.std(a) == 0 or np.std(b) == 0:
            r = np.nan
        else:
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            denom = np.sqrt((a_centered @ a_centered) * (b_centered @ b_centered)) + 1e-12
            r = float((a_centered @ b_centered) / denom)
        centers.append(start + window_length // 2)
        r_values.append(r)
        starts.append(start)
    return np.array(starts), np.array(centers), np.array(r_values)


# =======================
# EWMA-based detection
# =======================
def ewma_persistent_deviation_detection(
    z_abs: np.ndarray,
    ewma_smoothing: float,
    baseline_fraction: float,
    sigma_threshold: float,
    consecutive_required: int,
):
    """
    Apply an EWMA-based persistent deviation detector to |z|-scores.

    Parameters
    ----------
    z_abs : np.ndarray
        Absolute Fisher z-transformed correlation values.
    ewma_smoothing : float
        EWMA smoothing factor (lambda in (0, 1)).
    baseline_fraction : float
        Fraction of the earliest windows used to estimate the baseline
        mean and variance of |z|.
    sigma_threshold : float
        Threshold in units of the EWMA standard deviation.
    consecutive_required : int
        Number of consecutive threshold crossings required to declare
        a persistent anomaly.

    Returns
    -------
    ewma_values : np.ndarray
        EWMA of |z| over time.
    scores : np.ndarray
        Normalized deviation scores (in units of sigma).
    anomaly_flags : np.ndarray of bool
        Boolean mask indicating detected anomalies.
    """
    # select baseline region
    num_windows = len(z_abs)
    num_baseline = max(1, int(baseline_fraction * num_windows))
    baseline = z_abs[:num_baseline]
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size == 0:
        baseline = z_abs[np.isfinite(z_abs)]

    # robust estimate of baseline mean and sigma
    mu_abs = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - mu_abs)))
    sigma_abs = 1.4826 * mad if np.isfinite(mad) else float(np.std(baseline, ddof=1) + 1e-12)
    sigma_abs_ewma = float(
        np.sqrt(ewma_smoothing / (2.0 - ewma_smoothing)) * max(sigma_abs, 1e-12)
    )

    # EWMA recursion and persistent deviation test
    ewma_values = np.empty_like(z_abs, dtype=float)
    scores = np.empty_like(z_abs, dtype=float)
    anomaly_flags = np.zeros_like(z_abs, dtype=bool)

    S_abs = mu_abs
    breach_run = 0

    for i, za in enumerate(z_abs):
        if np.isfinite(za):
            S_abs = ewma_smoothing * float(za) + (1.0 - ewma_smoothing) * S_abs
            drift_abs = abs(S_abs - mu_abs)
            score = drift_abs / (sigma_abs_ewma + 1e-12)
        else:
            score = np.nan

        ewma_values[i] = S_abs
        scores[i] = score

        if not np.isfinite(score):
            breach_run = 0
            anomaly_flags[i] = False
        else:
            is_breach = score >= sigma_threshold
            breach_run = breach_run + 1 if is_breach else 0
            anomaly_flags[i] = breach_run >= consecutive_required

    return ewma_values, scores, anomaly_flags


# =======================
# End-to-end example (KLE + PCC + EWMA) on two signals
# =======================
def run_kle_pcc_ewma(
    signal_1: np.ndarray,
    signal_2: np.ndarray,
    kle_window_size: int,
    kle_quad_order: int,
    pcc_window_length: int,
    pcc_step_size: int,
    ewma_smoothing: float,
    baseline_fraction: float,
    sigma_threshold: float,
    consecutive_required: int,
):
    """
    High-level pipeline combining:
      1) KLE on each signal (first KL mode),
      2) sliding-window Pearson correlation on KL coefficients,
      3) Fisher z-transform and absolute value,
      4) EWMA-based persistent deviation detection.
    """
    # first KL mode of each signal
    kle1 = compute_kle_coefficients(signal_1, window_size=kle_window_size,
                                    quad_order=kle_quad_order)[:, 0]
    kle2 = compute_kle_coefficients(signal_2, window_size=kle_window_size,
                                    quad_order=kle_quad_order)[:, 0]

    # sliding-window Pearson correlation on KL coefficients
    starts, centers, r_values = sliding_pearson_corr(
        kle1, kle2,
        window_length=pcc_window_length,
        step_size=pcc_step_size,
    )

    # Fisher z-transform and absolute value
    eps = 1e-6
    r_clipped = np.clip(r_values, -1.0 + eps, 1.0 - eps)
    z_values = np.arctanh(r_clipped)
    z_values[~np.isfinite(r_values)] = np.nan
    z_abs = np.abs(z_values)

    # EWMA-based persistent deviation detection
    ewma_values, scores, anomaly_flags = ewma_persistent_deviation_detection(
        z_abs=z_abs,
        ewma_smoothing=ewma_smoothing,
        baseline_fraction=baseline_fraction,
        sigma_threshold=sigma_threshold,
        consecutive_required=consecutive_required,
    )

    return {
        "window_starts": starts,
        "window_centers": centers,
        "pearson_r": r_values,
        "fisher_z": z_values,
        "abs_fisher_z": z_abs,
        "ewma_abs_z": ewma_values,
        "scores": scores,
        "anomaly_flags": anomaly_flags,
    }

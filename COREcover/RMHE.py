"""
This module provides:

  - Geometric/units helpers:
        deg_to_rad, degps_to_radps, angwrap, robust_prior_from_rows

  - Vehicle process / measurement models:
        f_process(x, params_drag)
        h_measure(x)

  - Parameter calibration on benign data:
        fit_simple_params(...)
        fit_drag_params_collocation_xyv(...)

  - Receding-horizon robust MHE with sparse measurement attacks:
        rmhe_with_inputs_masked(...)
        run_receding_mhe(...)
"""

from __future__ import annotations

import math
import time
from typing import Iterable, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


# ---------------------------------------------------------------------------
# Global configuration / hyperparameters
# ---------------------------------------------------------------------------

# Time step (seconds)
TIME_STEP = 0.05

# State layout used throughout:
#   [0] pos_x
#   [1] pos_y
#   [2] longitudinal speed (m/s)
#   [3] yaw (rad)
#   [4] longitudinal acceleration ax (m/s^2)
#   [5] yaw rate (rad/s)
STATE_COLS = ["x_used", "y_used", "v_used", "psi_used", "Accelerometer.x", "Gyroscope.z"]

# Measurement channels (must be a subset / reordering of STATE_COLS)
OBS_COLS = ["x_used", "y_used", "v_used", "psi_used", "Accelerometer.x", "Gyroscope.z"]
OBS_IDXS = [STATE_COLS.index(c) for c in OBS_COLS]

# Control input columns: steer, throttle, brake
CONTROL_COLS = ["steer", "throttle", "brake"]

# By default, treat *all* control inputs as trusted. In your experiment code
# you can set CONTROL_CUTOFF to the last index where control is still trusted.
CONTROL_CUTOFF = 10**9  # large default: no cut-off unless overridden

# How many frames before the first window do we look back when building the prior?
PRIOR_BACK = 3          # how far back the anchor index is
PRIOR_SMOOTH = 5        # local averaging window around the anchor

# Vehicle / tire parameters
VEHICLE_WHEELBASE = 4.8
FRONT_AXLE_OFFSET = VEHICLE_WHEELBASE / 2.0
REAR_AXLE_OFFSET = VEHICLE_WHEELBASE / 2.0

TIRE_FRICTION_COEFF = 0.3        # mu
YAW_RATE_TIME_CONSTANT = 0.45    # tau_r
SLIP_COEFFICIENT = 0.010         # k_beta
GRAVITY = 9.80665

# Steering and longitudinal actuation model
MAX_STEER_RAD = 0.6
THROTTLE_ACCEL_GAIN = 3.5        # m/s^2 at full throttle
BRAKE_DECEL_GAIN = 7.0           # m/s^2 at full brake

STEER_MIN = -1.0
STEER_MAX = 1.0
THROTTLE_MIN = 0.0
THROTTLE_MAX = 1.0
BRAKE_MIN = 0.0
BRAKE_MAX = 1.0
WHEELBASE_EPS = 1e-6

# Speed / yaw-rate bounds used as RMHE box constraints
SPEED_MIN_MS = 0.1
SPEED_MAX_MS_DEFAULT = 50.0
YAW_RATE_MAX_DEFAULT = 3.0

# RMHE solver verbosity and limits
VERBOSE_RMHE = True
RMHE_PRINT_EVERY = 50           # print every N residual evaluations
RMHE_MAX_NFEV = 500             # max function evaluations per window
RMHE_EVAL_LIMIT = 10000         # hard cap on residual calls in one solve

# Attack sparsity regularization per channel (aligned with OBS_COLS)
LAMBDA_VEC = np.array(
    [
        10.0,  # x
        10.0,  # y
        1.0,   # v
        10.0,  # psi
        5.0,   # ax
        10.0,  # gyro
    ],
    dtype=float,
)

# Before attack (trusted measurements): strong penalty on atk -> atk ~= 0
LAMBDA_SCALE_PRE = 1000.0

# During/after attack (potentially corrupted measurements): very weak penalty
LAMBDA_SCALE_POST = 1e-3

# Control consistency weight (a, omega)
U_INV = np.diag([3.0, 3.0])

# Process and arrival weights (inverse covariance)
Q_INV = np.diag(
    [
        200.0,  # x
        200.0,  # y
        200.0,  # v
        200.0,  # psi
        200.0,  # ax
        200.0,  # gyro
    ]
)

P0 = np.diag(
    [
        10.0,   # x
        10.0,   # y
        5.0,     # v
        10.0,  # psi
        5.0,     # ax
        10.0,  # gyro
    ]
)

# Measurement weights (inverse covariance)
R_INV = np.diag(
    [
        10.0,  # x
        10.0,  # y
        10.0,  # v
        10.0,  # psi
        10.0,  # ax
        10.0,  # gyro
    ]
)


# ---------------------------------------------------------------------------
# Unit / geometry helpers
# ---------------------------------------------------------------------------

def deg_to_rad(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Convert a yaw-like signal to radians if it appears to be in degrees."""
    s = series.astype(float).copy()
    was_deg = s.dropna().abs().max() > 2.0 * np.pi
    if was_deg:
        s = np.deg2rad(s)
    return s, was_deg


def degps_to_radps(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Convert a yaw-rate-like signal to rad/s if it appears to be in deg/s."""
    s = series.astype(float).copy()
    was_degps = s.dropna().abs().median() > 1.5  # heuristically treat as deg/s
    if was_degps:
        s = np.deg2rad(s)
    return s, was_degps


def angwrap(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(rad_arr: Iterable[float]) -> float:
    """Circular mean of a set of angles (radians)."""
    rad_arr = np.asarray(rad_arr, dtype=float)
    s = np.sin(rad_arr).mean()
    c = np.cos(rad_arr).mean()
    return math.atan2(s, c)


def robust_prior_from_rows(
    df: pd.DataFrame,
    idx_lo: int,
    idx_hi: int,
    state_cols: Iterable[str],
) -> np.ndarray:
    """
    Build a robust prior state by averaging over rows [idx_lo, idx_hi] inclusive.

    - x, y, v, ax, gyro use median.
    - psi uses circular mean.
    """
    seg = df[state_cols].iloc[max(0, idx_lo): max(0, idx_hi) + 1].astype(float)
    if len(seg) == 0:
        return np.zeros(len(state_cols), dtype=float)

    x_med = float(seg[state_cols[0]].median())
    y_med = float(seg[state_cols[1]].median())
    v_med = float(seg[state_cols[2]].median())
    psi_cm = float(circular_mean(seg[state_cols[3]].values))
    ax_med = float(seg[state_cols[4]].median())
    gyro_med = float(seg[state_cols[5]].median())

    prior = np.array(
        [x_med, y_med, v_med, angwrap(psi_cm), ax_med, gyro_med],
        dtype=float,
    )
    return prior


# ---------------------------------------------------------------------------
# Control interface: (steer, throttle, brake) ↔ (ax, yaw_rate)
# ---------------------------------------------------------------------------

def controls_to_cmds(
    steer: float,
    throttle: float,
    brake: float,
    v_ms: float,
) -> Tuple[float, float]:
    """
    Map normalized controls (steer, throttle, brake) to commanded (ax, yaw_rate).
    """
    s = float(np.clip(steer, STEER_MIN, STEER_MAX))
    t = float(np.clip(throttle, THROTTLE_MIN, THROTTLE_MAX))
    b = float(np.clip(brake, BRAKE_MIN, BRAKE_MAX))

    delta = s * MAX_STEER_RAD
    a_cmd = THROTTLE_ACCEL_GAIN * t - BRAKE_DECEL_GAIN * b
    omega_cmd = (v_ms / max(VEHICLE_WHEELBASE, WHEELBASE_EPS)) * math.tan(delta)
    return a_cmd, omega_cmd


def state_to_controls(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Infer an equivalent (steer, throttle, brake) triple from the current state x.

    This is used when the original control log is not trusted:
    we interpret ax and yaw_rate in x as "desired" inputs and invert the simple
    control model to find a plausible set of normalized controls.
    """
    v_ms = float(max(x[2], 0.1))
    a_des = float(x[4])
    om_des = float(x[5])

    # Longitudinal: split desired acceleration into throttle / brake.
    if a_des >= 0.0:
        throttle = np.clip(a_des / THROTTLE_ACCEL_GAIN, THROTTLE_MIN, THROTTLE_MAX)
        brake = 0.0
    else:
        throttle = 0.0
        brake = np.clip(-a_des / BRAKE_DECEL_GAIN, BRAKE_MIN, BRAKE_MAX)

    # Lateral: approximate steer from yaw rate.
    if abs(v_ms) < 0.2 or abs(om_des) < 1e-4:
        steer = 0.0
    else:
        delta_des = math.atan2(om_des * VEHICLE_WHEELBASE, v_ms)
        steer = delta_des / MAX_STEER_RAD

    steer = float(np.clip(steer, STEER_MIN, STEER_MAX))
    throttle = float(np.clip(throttle, THROTTLE_MIN, THROTTLE_MAX))
    brake = float(np.clip(brake, BRAKE_MIN, BRAKE_MAX))
    return steer, throttle, brake


# ---------------------------------------------------------------------------
# Process and measurement models
# ---------------------------------------------------------------------------

def f_process(x: np.ndarray, params_drag: np.ndarray) -> np.ndarray:
    """
    6D process model for the vehicle:

        x = [pos_x, pos_y, v, psi, ax, yaw_rate]

    - Longitudinal dynamics:
        v_dot = ax - drag(v)
        drag(v) = c2 * v^2 + c1 * v + c0

    - Yaw dynamics:
        yaw_rate is limited by a friction circle and first-order dynamics
        with time constant YAW_RATE_TIME_CONSTANT.

    - Lateral motion:
        use a simple slip angle approximation beta ≈ SLIP_COEFFICIENT * v * yaw_rate
        and integrate position with heading psi + beta.
    """
    pos_x, pos_y, v, psi, ax, omega = map(float, x)
    c2, c1, c0 = [float(p) for p in params_drag]

    dt = TIME_STEP
    mu = TIRE_FRICTION_COEFF
    tau_r = YAW_RATE_TIME_CONSTANT
    k_beta = SLIP_COEFFICIENT

    v_max = SPEED_MAX_MS_DEFAULT
    omega_max = YAW_RATE_MAX_DEFAULT

    # Longitudinal dynamics with drag
    a_drag = c2 * v * v + c1 * v + c0
    v_dot = ax - a_drag
    v_mid = max(0.0, min(v + v_dot * dt, v_max))

    # Yaw-rate dynamics with friction circle limit
    omega_cap = min(omega_max, mu * GRAVITY / max(v_mid, 0.1))
    omega_sat = max(-omega_cap, min(omega, omega_cap))

    omega_mid = omega_sat * math.exp(-0.5 * dt / tau_r)
    omega_next = omega_sat * math.exp(-dt / tau_r)

    # Slip angle approximation
    beta = max(-0.35, min(k_beta * v_mid * omega_mid, 0.35))

    # Integrate position using psi + beta
    psi_mid = angwrap(psi + 0.5 * omega_mid * dt)
    pos_x_next = pos_x + v_mid * math.cos(psi_mid + beta) * dt
    pos_y_next = pos_y + v_mid * math.sin(psi_mid + beta) * dt

    # Update v and psi
    v_next = max(0.0, min(v + v_dot * dt, v_max))
    psi_next = angwrap(psi + 0.5 * (omega_mid + omega_next) * dt)

    ax_next = ax
    omega_next = max(-omega_max, min(omega_next, omega_max))

    return np.array(
        [pos_x_next, pos_y_next, v_next, psi_next, ax_next, omega_next],
        dtype=float,
    )


def h_measure(x: np.ndarray) -> np.ndarray:
    """
    Measurement model: by default we directly observe selected components
    of the 6D state.

    OBS_COLS determines which channels are returned and in which order.
    """
    x = np.asarray(x, dtype=float)
    pos_x, pos_y, v_ms, psi, ax, omega = x[:6]
    out: list[float] = []
    for ch in OBS_COLS:
        if ch == "x_used":
            out.append(pos_x)
        elif ch == "y_used":
            out.append(pos_y)
        elif ch == "v_used":
            out.append(v_ms)
        elif ch == "psi_used":
            out.append(psi)
        elif ch == "Accelerometer.x":
            out.append(ax)
        elif ch == "Gyroscope.z":
            out.append(omega)
        else:
            out.append(x[STATE_COLS.index(ch)])
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# Parameter calibration on benign data
# ---------------------------------------------------------------------------

def fit_simple_params(
    df: pd.DataFrame,
    dt: float = TIME_STEP,
    params_drag: Iterable[float] = (0.0, 0.0, 0.0),
    n_head: int = 600,
) -> None:
    """
    Coarsely calibrate YAW_RATE_TIME_CONSTANT and SLIP_COEFFICIENT on a benign
    segment by minimizing the position prediction error of f_process.

    This function updates the global YAW_RATE_TIME_CONSTANT and SLIP_COEFFICIENT
    in-place.
    """
    global YAW_RATE_TIME_CONSTANT, SLIP_COEFFICIENT

    df_fit = df.iloc[:max(10, min(n_head, len(df)))].copy()

    # Position
    if "location.x" in df_fit.columns:
        xcol = "location.x"
    elif "x_used" in df_fit.columns:
        xcol = "x_used"
    else:
        raise ValueError("fit_simple_params: requires 'location.x' or 'x_used'.")

    if "location.y" in df_fit.columns:
        ycol = "location.y"
    elif "y_used" in df_fit.columns:
        ycol = "y_used"
    else:
        raise ValueError("fit_simple_params: requires 'location.y' or 'y_used'.")

    Xp = df_fit[xcol].astype(float).to_numpy()
    Yp = df_fit[ycol].astype(float).to_numpy()

    # Speed (m/s)
    if "v_used" in df_fit.columns:
        V = df_fit["v_used"].astype(float).to_numpy()
    elif "Speed" in df_fit.columns:
        V = df_fit["Speed"].astype(float).to_numpy()
        V = V / (3.6 if V.max() > 50.0 else 1.0)
    else:
        VX = np.gradient(Xp, dt)
        VY = np.gradient(Yp, dt)
        V = np.hypot(VX, VY)

    # Yaw
    if "psi_used" in df_fit.columns:
        psi = df_fit["psi_used"].astype(float).to_numpy()
        psi = angwrap(psi)
    else:
        psi = np.unwrap(np.arctan2(np.gradient(Yp), np.gradient(Xp)))

    # Yaw rate
    if "Gyroscope.z" in df_fit.columns:
        om = df_fit["Gyroscope.z"].astype(float).to_numpy()
        if np.nanmedian(np.abs(om)) > 1.5:
            om = np.deg2rad(om)
    else:
        om = np.gradient(psi, dt)

    # Longitudinal acceleration
    if "Accelerometer.x" in df_fit.columns:
        ax = df_fit["Accelerometer.x"].astype(float).to_numpy()
    else:
        ax = np.gradient(V, dt)

    x0 = np.array([Xp[0], Yp[0], max(0.0, V[0]), psi[0], ax[0], om[0]], dtype=float)

    tau_r_grid = np.linspace(0.25, 0.90, 9)
    k_beta_grid = np.linspace(0.006, 0.018, 7)

    best_tau_r = YAW_RATE_TIME_CONSTANT
    best_k_beta = SLIP_COEFFICIENT
    best_err = float("inf")

    for tau_r in tau_r_grid:
        for k_beta in k_beta_grid:
            YAW_RATE_TIME_CONSTANT = float(tau_r)
            SLIP_COEFFICIENT = float(k_beta)
            x = x0.copy()
            err = 0.0
            for k in range(len(df_fit) - 1):
                x[4] = float(ax[k])
                x[5] = float(om[k])
                x = f_process(x, params_drag)
                dx = x[0] - Xp[k + 1]
                dy = x[1] - Yp[k + 1]
                err += dx * dx + dy * dy
            if err < best_err:
                best_err = err
                best_tau_r, best_k_beta = tau_r, k_beta

    YAW_RATE_TIME_CONSTANT = float(best_tau_r)
    SLIP_COEFFICIENT = float(best_k_beta)
    print(
        f"[calib] tau_r={best_tau_r:.3f}, k_beta={best_k_beta:.4f} "
        f"(position error={best_err:.3g})"
    )


def fit_drag_params_collocation_xyv(
    df: pd.DataFrame,
    x_col: str = "location.x",
    y_col: str = "location.y",
    speed_col: str = "Speed",
    n_head: int = 400,
    dt: float = TIME_STEP,
    stride: int = 1,
    control_stride: int = 5,
    # tracking weights
    w_v: float = 10.0,
    w_xy: float = 1.0,
    # collocation residual weight
    w_dyn: float = 50.0,
    # curvature prior weight
    w_kappa: float = 1.0,
    # control magnitude / smoothness regularization
    alpha_u: float = 1.0,
    beta_u: float = 50.0,
    alpha_om: float = 2.0,
    beta_om: float = 80.0,
    # drag prior
    lambda_p: float = 100.0,
    p_prior: Tuple[float, float, float] = (1.6e-4, 5e-3, 0.10),
    # physical bounds
    a_bounds: Tuple[float, float] = (-5.0, 3.0),
    om_bounds: Tuple[float, float] = (-1.0, 1.0),
    theta0_bounds: Tuple[float, float] = (-np.pi, np.pi),
    # optimizer settings
    maxiter: int = 800,
    maxfun: int = 800000,
    ftol: float = 1e-12,
) -> np.ndarray:
    """
    Fit drag parameters (c2, c1, c0) using only (x, y, v) from a benign segment.

    The model is:

        v_dot = u_a - (c2 * v^2 + c1 * v + c0)
        theta_dot = omega
        x_dot = v cos(theta)
        y_dot = v sin(theta)

    where (u_a, omega) are treated as piecewise-constant "virtual controls" that
    are also optimized (with magnitude and smoothness regularization).
    """
    def softplus(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) + eps

    def moving_avg(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return x.copy()
        ker = np.ones(k) / float(k)
        return np.convolve(x, ker, mode="same")

    x_obs_full = df[x_col].iloc[:n_head].astype(float).values
    y_obs_full = df[y_col].iloc[:n_head].astype(float).values
    v_obs_full = df[speed_col].iloc[:n_head].astype(float).values

    idx = np.arange(0, len(v_obs_full), stride)
    x_obs = x_obs_full[idx]
    y_obs = y_obs_full[idx]
    v_obs = v_obs_full[idx]
    DT = dt * stride
    K = len(v_obs)
    if K < 5:
        raise ValueError("fit_drag_params_collocation_xyv: not enough samples (<5).")

    # Geometric curvature from XY
    dx = np.diff(x_obs)
    dy = np.diff(y_obs)
    ds = np.hypot(dx, dy) + 1e-9
    theta_geo = np.unwrap(np.arctan2(dy, dx))
    dtheta = np.diff(theta_geo)

    kappa = np.zeros(K - 1)
    if K >= 3:
        kappa[1:] = dtheta / ds[1:]
        kappa[0] = kappa[1]
    kappa = np.clip(kappa, -0.2, 0.2)

    # Control segmentation
    M = int(np.ceil((K - 1) / control_stride))

    def idx_ctrl(k: int) -> int:
        i = k // control_stride
        return i if i < M else M - 1

    # Initial guess for u_a (longitudinal “control”)
    a0_seq = np.diff(v_obs) / DT
    a0_seq = np.clip(a0_seq, a_bounds[0], a_bounds[1])
    a0_blocks = []
    for i in range(M):
        k0 = i * control_stride
        k1 = min((i + 1) * control_stride, K - 1)
        if k1 > k0:
            a0_blocks.append(np.median(a0_seq[k0:k1]))
        else:
            a0_blocks.append(0.0)
    u0 = moving_avg(np.array(a0_blocks, dtype=float), 3)
    u0 = np.clip(u0, a_bounds[0], a_bounds[1])

    # Initial guess for omega from curvature
    om0_per_step = v_obs[:-1] * kappa
    om0_blocks = []
    for i in range(M):
        k0 = i * control_stride
        k1 = min((i + 1) * control_stride, K - 1)
        if k1 > k0:
            om0_blocks.append(np.median(om0_per_step[k0:k1]))
        else:
            om0_blocks.append(0.0)
    om0 = moving_avg(np.array(om0_blocks, dtype=float), 3)
    om0 = np.clip(om0, om_bounds[0], om_bounds[1])

    # Initial yaw
    if K >= 2:
        theta0_init = float(np.arctan2(y_obs[1] - y_obs[0], x_obs[1] - x_obs[0]))
    else:
        theta0_init = 0.0

    def inv_softplus(y_val: float) -> float:
        return np.log(np.expm1(max(y_val, 1e-10)))

    p0 = np.array(
        [
            inv_softplus(p_prior[0]),
            inv_softplus(p_prior[1]),
            inv_softplus(p_prior[2]),
        ],
        dtype=float,
    )

    def unpack(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        p_t = z[:3]
        u = z[3 : 3 + M]
        om = z[3 + M : 3 + 2 * M]
        th0 = z[-1]
        return p_t, u, om, th0

    def rollout(
        p_t: np.ndarray,
        u: np.ndarray,
        om_ctrl: np.ndarray,
        th0: float,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float, Tuple[float, float, float]]:
        c2, c1, c0 = softplus(p_t)
        x = np.zeros(K)
        y = np.zeros(K)
        v = np.zeros(K)
        th = np.zeros(K)

        x[0], y[0], v[0], th[0] = x_obs[0], y_obs[0], v_obs[0], th0

        dyn_res = 0.0

        for k in range(K - 1):
            i = idx_ctrl(k)
            drag = c2 * (v[k] ** 2) + c1 * v[k] + c0
            v_dot = u[i] - drag
            th_dot = om_ctrl[i]
            x_dot = v[k] * math.cos(th[k])
            y_dot = v[k] * math.sin(th[k])

            v[k + 1] = v[k] + v_dot * DT
            th[k + 1] = th[k] + th_dot * DT
            x[k + 1] = x[k] + x_dot * DT
            y[k + 1] = y[k] + y_dot * DT

            v_mid = 0.5 * (v[k] + v[k + 1])
            th_mid = 0.5 * (th[k] + th[k + 1])
            drag_m = c2 * (v_mid ** 2) + c1 * v_mid + c0
            v_dot_m = u[i] - drag_m
            th_dot_m = om_ctrl[i]
            x_dot_m = v_mid * math.cos(th_mid)
            y_dot_m = v_mid * math.sin(th_mid)

            rx = (x[k + 1] - x[k]) - DT * x_dot_m
            ry = (y[k + 1] - y[k]) - DT * y_dot_m
            rv = (v[k + 1] - v[k]) - DT * v_dot_m
            rth = (th[k + 1] - th[k]) - DT * th_dot_m
            dyn_res += rx * rx + ry * ry + rv * rv + rth * rth

        return (x, y, v, th), dyn_res, (float(c2), float(c1), float(c0))

    def objective(z: np.ndarray) -> float:
        p_t, u, om_ctrl, th0 = unpack(z)
        (x_pred, y_pred, v_pred, th_pred), dyn_res, (c2, c1, c0) = rollout(
            p_t, u, om_ctrl, th0
        )

        e_v = v_pred - v_obs
        e_x = x_pred - x_obs
        e_y = y_pred - y_obs
        cost_track = w_v * float(e_v @ e_v) + w_xy * (
            float(e_x @ e_x) + float(e_y @ e_y)
        )

        cost_dyn = w_dyn * dyn_res

        om_per_step = np.array([om_ctrl[idx_ctrl(k)] for k in range(K - 1)])
        kappa_mis = om_per_step - v_pred[:-1] * kappa
        cost_kappa = w_kappa * float(kappa_mis @ kappa_mis)

        du = np.diff(u) if len(u) > 1 else np.array([0.0])
        dom = np.diff(om_ctrl) if len(om_ctrl) > 1 else np.array([0.0])
        reg_u = alpha_u * float(u @ u) + beta_u * float(du @ du)
        reg_om = alpha_om * float(om_ctrl @ om_ctrl) + beta_om * float(dom @ dom)

        c2p, c1p, c0p = p_prior
        rp = (
            ((c2 - c2p) / c2p) ** 2
            + ((c1 - c1p) / max(c1p, 1e-6)) ** 2
            + ((c0 - c0p) / c0p) ** 2
        )
        reg_p = lambda_p * rp

        penalty = 0.0
        if (v_pred < 0.0).any():
            v_neg = np.minimum(v_pred, 0.0)
            penalty += 1e6 * float(v_neg @ v_neg)

        return cost_track + cost_dyn + cost_kappa + reg_u + reg_om + reg_p + penalty

    z0 = np.concatenate([p0, u0, om0, np.array([theta0_init])])

    bounds = []
    bounds += [(-15.0, 5.0)] * 3
    bounds += [a_bounds] * M
    bounds += [om_bounds] * M
    bounds += [theta0_bounds]

    res = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "maxfun": maxfun, "ftol": ftol, "maxls": 50},
    )

    if not res.success:
        print("⚠️ Drag collocation optimization did not converge:", res.message)

    p_t_hat, _, _, _ = unpack(res.x)
    c2, c1, c0 = softplus(p_t_hat)
    print(
        f"✅ Drag (collocation) fit: c2={c2:.6g}, c1={c1:.6g}, c0={c0:.6g} | "
        f"iters={res.nit}, evals={res.nfev}"
    )
    return np.array([float(c2), float(c1), float(c0)], dtype=float)


# ---------------------------------------------------------------------------
# RMHE core: one window with optional control mask
# ---------------------------------------------------------------------------

def rmhe_with_inputs_masked(
    prior_x: np.ndarray,
    P_arr: np.ndarray,
    Y_obs: np.ndarray,
    U_obs: np.ndarray,
    ctrl_mask: np.ndarray,
    lambda_attack: float,
    params_drag: np.ndarray,
    z_init: Optional[np.ndarray] = None,
    final_xy_box: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Receding-horizon MHE for a single window with optional trusted controls.

    State:
        x_k = [x, y, v, psi, ax, omega]

    For each step k in the window:
        - If ctrl_mask[k] > 0.5:
            * Treat U_obs[k] = (steer, throttle, brake) as trusted controls.
            * Compute (ax_cmd, omega_cmd) via controls_to_cmds.
            * Use these commands in f_process, and also add a control-consistency
              residual so that (ax, omega) in the state track these commands.

        - If ctrl_mask[k] == 0:
            * Controls are not trusted.
            * Infer an equivalent (steer, throttle, brake) from x_k via
              state_to_controls, then map to (ax_cmd, omega_cmd) and feed
              f_process. The process model becomes “closed-loop” in terms of
              the estimated state.

    Each measurement channel has its own sparse attack variable atk_k[i].
    Before the attack (ctrl_mask=1) we strongly penalize atk_k → 0.
    During/after the attack (ctrl_mask=0) we relax the penalty so that atk_k
    can absorb large residuals.

    If final_xy_box is not None, the last state in the window is additionally
    constrained to lie within the given box:
        [x_min, x_max, y_min, y_max].
    """
    N = len(Y_obs)
    nx = 6
    ny = Y_obs.shape[1]

    assert U_obs.shape == (N, 3)
    assert ctrl_mask.shape == (N,)

    def unpack_z(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = z[: (N + 1) * nx].reshape(N + 1, nx)
        atk = z[(N + 1) * nx : (N + 1) * nx + N * ny].reshape(N, ny)
        return xs, atk

    # Initial guess: either provided or forward rollout with zero attack
    if z_init is not None and z_init.size == (N + 1) * nx + N * ny:
        z0 = z_init.copy()
    else:
        x0_seq = [prior_x.copy()]

        try:
            idx_ax = OBS_COLS.index("Accelerometer.x")
        except ValueError:
            idx_ax = None
        try:
            idx_gz = OBS_COLS.index("Gyroscope.z")
        except ValueError:
            idx_gz = None

        v_max = SPEED_MAX_MS_DEFAULT
        omega_max = YAW_RATE_MAX_DEFAULT

        for k in range(N):
            xk = x0_seq[-1].copy()

            # Fill ax / omega from measurements as a rough guess
            if idx_ax is not None:
                xk[4] = float(Y_obs[k, idx_ax])
            if idx_gz is not None:
                xk[5] = float(Y_obs[k, idx_gz])

            if ctrl_mask[k] > 0.5:
                steer_k, thr_k, brk_k = map(float, U_obs[k])
                v_k = float(xk[2])
                a_cmd, om_cmd = controls_to_cmds(steer_k, thr_k, brk_k, v_k)
            else:
                steer_k, thr_k, brk_k = state_to_controls(xk)
                v_k = float(xk[2])
                a_cmd, om_cmd = controls_to_cmds(steer_k, thr_k, brk_k, v_k)

            xk[4], xk[5] = a_cmd, om_cmd
            x_next = f_process(xk, params_drag)

            x_next[3] = angwrap(x_next[3])
            x_next[2] = np.clip(x_next[2], SPEED_MIN_MS, v_max)
            x_next[5] = np.clip(x_next[5], -omega_max, omega_max)

            x0_seq.append(x_next)

        xs0 = np.asarray(x0_seq, dtype=float)
        xs0[:, 3] = np.vectorize(angwrap)(xs0[:, 3])
        xs0[:, 2] = np.clip(xs0[:, 2], SPEED_MIN_MS, v_max)
        xs0[:, 5] = np.clip(xs0[:, 5], -omega_max, omega_max)

        atk0 = np.zeros((N, ny), dtype=float)
        z0 = np.concatenate([xs0.reshape(-1), atk0.reshape(-1)])

    # Weight matrices → per-component scaling
    Qw = np.sqrt(np.diag(Q_INV)).astype(float)
    Rw = np.sqrt(np.diag(R_INV)).astype(float)
    Pw = np.sqrt(np.diag(P0)).astype(float)
    Uw = np.sqrt(np.diag(U_INV)).astype(float)

    sqrt_lambda_vec = np.sqrt(LAMBDA_VEC).astype(float)
    sqrt_lambda_pre = LAMBDA_SCALE_PRE * sqrt_lambda_vec
    sqrt_lambda_post = LAMBDA_SCALE_POST * sqrt_lambda_vec

    try:
        theta_obs_idx = OBS_COLS.index("psi_used")
    except Exception:
        theta_obs_idx = None

    eval_counter = {"n": 0}
    last_z = {"z": z0.copy()}
    t_start = time.time()

    def residual(z: np.ndarray) -> np.ndarray:
        xs, atk = unpack_z(z)
        eval_counter["n"] += 1
        last_z["z"] = z.copy()

        if eval_counter["n"] > RMHE_EVAL_LIMIT:
            raise RuntimeError("RMHE evaluation limit reached")

        if VERBOSE_RMHE and (eval_counter["n"] % RMHE_PRINT_EVERY == 0):
            elapsed = time.time() - t_start
            last = xs[-1]
            print(
                f"[RMHE] eval={eval_counter['n']:4d}, "
                f"elapsed={elapsed:6.1f}s, "
                f"last(x,y,v,psi)=({last[0]:8.3f},{last[1]:8.3f},"
                f"{last[2]:5.2f},{last[3]:5.2f})"
            )

        res_list = []

        # Arrival cost at window start
        res_list.append(Pw * (xs[0] - prior_x))

        v_max = SPEED_MAX_MS_DEFAULT
        omega_max = YAW_RATE_MAX_DEFAULT

        for k in range(N):
            v_k = float(xs[k][2])

            if ctrl_mask[k] > 0.5:
                steer_k, thr_k, brk_k = map(float, U_obs[k])
                a_cmd, om_cmd = controls_to_cmds(steer_k, thr_k, brk_k, v_k)
            else:
                steer_k, thr_k, brk_k = state_to_controls(xs[k])
                a_cmd, om_cmd = controls_to_cmds(steer_k, thr_k, brk_k, v_k)

            x_dyn = xs[k].copy()
            x_dyn[4] = a_cmd
            x_dyn[5] = om_cmd

            fwd = f_process(x_dyn, params_drag)
            res_list.append(Qw * (xs[k + 1] - fwd))

            y_pred = h_measure(xs[k + 1]).astype(float)
            e_meas = (Y_obs[k] - y_pred - atk[k]).astype(float)
            if theta_obs_idx is not None and theta_obs_idx < len(e_meas):
                e_meas[theta_obs_idx] = angwrap(e_meas[theta_obs_idx])
            res_list.append(Rw * e_meas)

            if ctrl_mask[k] > 0.5:
                e_ctrl = np.array(
                    [xs[k][4] - a_cmd, xs[k][5] - om_cmd],
                    dtype=float,
                )
                res_list.append(Uw * e_ctrl)

            if ctrl_mask[k] > 0.5:
                sqrt_lambda_k = sqrt_lambda_pre
            else:
                sqrt_lambda_k = sqrt_lambda_post
            res_list.append(sqrt_lambda_k * atk[k])

        return np.concatenate([r.ravel() for r in res_list])

    nxs = (N + 1) * nx
    natk = N * ny
    lb = -np.inf * np.ones(nxs + natk)
    ub = np.inf * np.ones(nxs + natk)

    v_max = SPEED_MAX_MS_DEFAULT
    omega_max = YAW_RATE_MAX_DEFAULT

    for k in range(N + 1):
        base = k * nx
        lb[base + 2] = SPEED_MIN_MS
        ub[base + 2] = v_max
        lb[base + 3] = -np.pi
        ub[base + 3] = np.pi
        lb[base + 5] = -omega_max
        ub[base + 5] = omega_max

    if final_xy_box is not None:
        x_min, x_max, y_min, y_max = final_xy_box
        k_final = N
        base = k_final * nx
        lb[base + 0] = x_min
        ub[base + 0] = x_max
        lb[base + 1] = y_min
        ub[base + 1] = y_max

    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    z0 = np.minimum(z0, ub, where=finite_ub, out=z0.copy())
    z0 = np.maximum(z0, lb, where=finite_lb, out=z0)

    try:
        sol = least_squares(
            residual,
            z0,
            method="trf",
            bounds=(lb, ub),
            loss="huber",
            f_scale=1.0,
            max_nfev=RMHE_MAX_NFEV,
            verbose=0,
        )
    except RuntimeError as e:
        if "RMHE evaluation limit reached" in str(e):
            if VERBOSE_RMHE:
                print(
                    f"[RMHE] stopped early at eval={eval_counter['n']} "
                    f"due to RMHE_EVAL_LIMIT={RMHE_EVAL_LIMIT}"
                )

            class _Dummy:
                pass

            sol = _Dummy()
            sol.x = last_z["z"]
            sol.cost = np.nan
            sol.nfev = eval_counter["n"]
            sol.niter = -1
            sol.success = False
            sol.message = str(e)
        else:
            raise
    except np.linalg.LinAlgError as e:
        print("⚠️ least_squares(trf) SVD did not converge, retrying with dogbox:", e)
        try:
            sol = least_squares(
                residual,
                z0,
                method="dogbox",
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=RMHE_MAX_NFEV,
                verbose=0,
            )
        except np.linalg.LinAlgError as e2:
            print(
                "⚠️ least_squares(dogbox) also failed, falling back to last feasible z:",
                e2,
            )

            class _Dummy:
                pass

            sol = _Dummy()
            sol.x = last_z["z"]
            sol.cost = np.nan
            sol.nfev = eval_counter["n"]
            sol.niter = -1
            sol.success = False
            sol.message = f"LinAlgError in trf & dogbox: {e2}"

    elapsed_total = time.time() - t_start
    if not getattr(sol, "success", False):
        print(
            f"⚠️ least_squares did not report success: "
            f"{getattr(sol, 'message', 'unknown')} "
            f"(eval={eval_counter['n']}, elapsed={elapsed_total:.1f}s, "
            f"scipy_nfev={getattr(sol, 'nfev', -1)})"
        )
    elif VERBOSE_RMHE:
        print(
            f"[RMHE] done. eval={eval_counter['n']}, "
            f"elapsed={elapsed_total:.1f}s, cost={sol.cost:.3g}, "
            f"scipy_nfev={getattr(sol, 'nfev', -1)}"
        )

    xs_opt, atk_opt = unpack_z(sol.x)
    return xs_opt, atk_opt


# ---------------------------------------------------------------------------
# Sliding-window RMHE driver (online / receding horizon)
# ---------------------------------------------------------------------------

def run_receding_mhe(
    df: pd.DataFrame,
    state_cols: Iterable[str],
    obs_cols: Iterable[str],
    params_drag: np.ndarray,
    P_arr: np.ndarray,
    lambda_attack: float,
    dt: float = TIME_STEP,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    window_N: int = 40,
    terminal_xy_box: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Standard sliding-window RMHE (online):

      At global time t we use the last T = window_N measurements
      [t-T+1, ..., t] to estimate x_t (the state at the window end).
      The window is then shifted forward by one step.

    Parameters
    ----------
    df : DataFrame
        Log containing at least state_cols, obs_cols and CONTROL_COLS.
    state_cols : list of str
        Names of the 6D state columns.
    obs_cols : list of str
        Names of the measurement channels (subset / reorder of state_cols).
    params_drag : np.ndarray
        Drag parameters (c2, c1, c0) passed into f_process.
    P_arr : np.ndarray
        Arrival weight matrix for the prior (same shape as Q_INV).
    lambda_attack : float
        Base sparsity weight (currently unused but kept for API symmetry).
    dt : float
        Time step (seconds).
    start_idx, end_idx : int
        Global index range [start_idx, end_idx) over which RMHE is run.
        If None, defaults to the full range of df.
    window_N : int
        Number of measurements per MHE window.
    terminal_xy_box : (x_min, x_max, y_min, y_max) or None
        Optional terminal constraint for the *last* window only.
    """
    assert len(state_cols) == 6

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df)

    nx = len(state_cols)
    ny = len(obs_cols)

    # First time at which we can output an estimate (need window_N past measurements)
    INITIAL_OFFSET = 5  # small margin; can be tuned
    t0 = start_idx + window_N - INITIAL_OFFSET
    if t0 < start_idx:
        raise ValueError("window_N is too large for the chosen start_idx.")

    est_index = list(range(t0, end_idx))
    x_hat_list: list[np.ndarray] = []
    a_l1_list: list[float] = []
    loop_times: list[float] = []

    # First window
    t = t0
    y_start = t - window_N + 1
    y_end = t + 1  # right-open
    Y_obs = df[obs_cols].iloc[y_start:y_end].values.astype(float)
    U_obs = df[CONTROL_COLS].iloc[y_start:y_end].values.astype(float)

    global_idx = np.arange(y_start, y_end)
    ctrl_mask = (global_idx <= CONTROL_CUTOFF).astype(float)

    if y_start - 1 >= 0:
        prior_anchor = y_start - PRIOR_BACK
        idx_hi = max(0, prior_anchor)
        idx_lo = max(0, idx_hi - (PRIOR_SMOOTH - 1))
        prior_x = robust_prior_from_rows(df, idx_lo, idx_hi, state_cols)
    else:
        prior_x = df[state_cols].iloc[y_start].values.astype(float)

    is_last_window = t == end_idx - 1
    final_xy_box = terminal_xy_box if is_last_window else None

    t_loop_start = time.perf_counter()
    x_seq, a_seq = rmhe_with_inputs_masked(
        prior_x=prior_x,
        P_arr=P_arr,
        Y_obs=Y_obs,
        U_obs=U_obs,
        ctrl_mask=ctrl_mask,
        lambda_attack=lambda_attack,
        params_drag=params_drag,
        z_init=None,
        final_xy_box=final_xy_box,
    )
    t_loop_end = time.perf_counter()
    loop_times.append(t_loop_end - t_loop_start)

    x_hat_t = x_seq[-1].copy()
    x_hat_list.append(x_hat_t)
    a_l1_list.append(float(np.linalg.norm(a_seq[-1], ord=1)))

    prev_x_seq, prev_a_seq = x_seq, a_seq

    # Subsequent windows
    for t in range(t0 + 1, end_idx):
        y_start = t - window_N + 1
        y_end = t + 1
        Y_obs = df[obs_cols].iloc[y_start:y_end].values.astype(float)
        U_obs = df[CONTROL_COLS].iloc[y_start:y_end].values.astype(float)
        global_idx = np.arange(y_start, y_end)
        ctrl_mask = (global_idx <= CONTROL_CUTOFF).astype(float)

        prior_x = prev_x_seq[1].copy()

        xs_init = prev_x_seq[1:].copy()
        xs_init = np.vstack([xs_init, f_process(xs_init[-1], params_drag)])

        v_max = SPEED_MAX_MS_DEFAULT
        omega_max = YAW_RATE_MAX_DEFAULT

        xs_init[:, 3] = ((xs_init[:, 3] + np.pi) % (2.0 * np.pi)) - np.pi
        xs_init[:, 2] = np.clip(xs_init[:, 2].astype(float), SPEED_MIN_MS, v_max)
        xs_init[:, 5] = np.clip(xs_init[:, 5].astype(float), -omega_max, omega_max)

        prior_x[3] = angwrap(prior_x[3])
        prior_x[2] = np.clip(float(prior_x[2]), SPEED_MIN_MS, v_max)
        prior_x[5] = np.clip(float(prior_x[5]), -omega_max, omega_max)

        atk_init = prev_a_seq[1:].copy()
        if atk_init.shape[0] < Y_obs.shape[0]:
            atk_init = np.vstack([atk_init, np.zeros((1, atk_init.shape[1]))])

        z_init = np.concatenate([xs_init.reshape(-1), atk_init.reshape(-1)])

        is_last_window = t == end_idx - 1
        final_xy_box = terminal_xy_box if is_last_window else None

        t_loop_start = time.perf_counter()
        x_seq, a_seq = rmhe_with_inputs_masked(
            prior_x=prior_x,
            P_arr=P_arr,
            Y_obs=Y_obs,
            U_obs=U_obs,
            ctrl_mask=ctrl_mask,
            lambda_attack=lambda_attack,
            params_drag=params_drag,
            z_init=z_init,
            final_xy_box=final_xy_box,
        )
        t_loop_end = time.perf_counter()
        loop_times.append(t_loop_end - t_loop_start)

        x_hat_t = x_seq[-1].copy()
        x_hat_list.append(x_hat_t)
        a_l1_list.append(float(np.linalg.norm(a_seq[-1], ord=1)))

        prev_x_seq, prev_a_seq = x_seq, a_seq

    x_hat_arr = np.asarray(x_hat_list, dtype=float)
    est_cols = [f"R_MHE_{c}" for c in state_cols]

    est_df = pd.DataFrame(x_hat_arr, columns=est_cols, index=est_index)
    a_l1_series = pd.Series(a_l1_list, name="R_MHE_attack_L1", index=est_index)

    loop_times_arr = np.asarray(loop_times, dtype=float)
    if loop_times_arr.size > 0:
        timing_stats: Dict[str, Any] = {
            "loop_times": loop_times_arr,
            "mean": float(loop_times_arr.mean()),
            "std": float(loop_times_arr.std()),
            "max": float(loop_times_arr.max()),
        }
    else:
        timing_stats = {
            "loop_times": np.array([], dtype=float),
            "mean": float("nan"),
            "std": float("nan"),
            "max": float("nan"),
        }

    return est_df, a_l1_series, timing_stats

from __future__ import annotations

import sys
import math
import time
import weakref
from dataclasses import dataclass
from typing import Any

import argparse
import carla
import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt

# ===================== Configuration =====================

WIDTH, HEIGHT = 1280, 720
TICK_HZ = 20.0                 # control / rendering frequency
SPAWN_INDEX = 61               # spawn index on the map
FOV = 90

# Automatic stopping conditions
END_STOP_S_REM = 1.5
END_STOP_SPEED_KMH = 0.5
END_STOP_HOLD_SEC = 1.0
FOLLOW_SPEED_ONLY = True

# Lateral / longitudinal PID parameters
LAT_ARGS = {"K_P": 1.0, "K_I": 0.5, "K_D": 0.0}
LON_ARGS = {"K_P": 0.8, "K_I": 0.1, "K_D": 0.02}

# Dynamic lookahead: Ld = BASE + K * v[m/s]
LOOKAHEAD_BASE = 0.1
LOOKAHEAD_K = 0.1

# Steering shaping (low-pass + rate limiting)
STEER_TAU = 0.05               # low-pass filter time constant (s)
STEER_RATE_LIMIT = 5.0         # steering rate limit (rad/s)
STEER_ABS_LIMIT = 1.0          # absolute steering limit

# Minimal throttle to overcome static friction at very low speed
MIN_KICKSTART_THROTTLE = 0.30
KICKSTART_SPEED_KMH = 0.5


# ===================== Utility / control modules =====================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class FirstOrderFilter:
    """First-order low-pass filter to limit bandwidth and suppress noise."""
    def __init__(self, tau: float, dt: float, init: float = 0.0) -> None:
        self.alpha = clamp(dt / (tau + dt), 0.0, 1.0)
        self.y = init

    def reset(self, value: float = 0.0) -> None:
        self.y = value

    def step(self, x: float) -> float:
        self.y = self.y + self.alpha * (x - self.y)
        return self.y


class RateLimiter:
    """Rate limiter for steering commands."""
    def __init__(self, max_rate_per_sec: float, dt: float, init: float = 0.0) -> None:
        self.max_delta = max_rate_per_sec * dt
        self.prev = init

    def step(self, x: float) -> float:
        delta = clamp(x - self.prev, -self.max_delta, self.max_delta)
        self.prev += delta
        return self.prev


class AntiWindupPID:
    """Longitudinal PID controller with anti-windup."""
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        u_min: float,
        u_max: float,
    ) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self.integral = 0.0
        self.prev_err = 0.0

    def step(self, target: float, actual: float) -> float:
        err = target - actual
        d = (err - self.prev_err) / self.dt
        u = self.kp * err + self.ki * self.integral + self.kd * d
        u_sat = clamp(u, self.u_min, self.u_max)

        # Conditional integration to mitigate windup
        if (u == u_sat) or (
            math.copysign(1.0, u - u_sat) != math.copysign(1.0, err)
        ):
            self.integral += err * self.dt
        self.integral = clamp(self.integral, -10.0, 10.0)
        self.prev_err = err
        return u_sat


# ===================== Path / lookahead utilities =====================

def build_path_from_reconstruction(
    df: pd.DataFrame,
    speed_kmh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a polyline path and arc-length parameterization from state reconstruction results.

    The state reconstruction results are expected to contain at least two rows and columns
    ["location x", "location y"]. A simple de-duplication is applied
    to remove nearly coincident consecutive points.
    """
    pts_raw = df[["location x", "location y"]].to_numpy(dtype=float)
    if len(pts_raw) < 2:
        raise ValueError(
            "State reconstruction results need at least two position rows: "
            "columns=[location x, location y]"
        )

    mask = np.ones(len(pts_raw), dtype=bool)
    mask[1:] = np.linalg.norm(pts_raw[1:] - pts_raw[:-1], axis=1) > 1e-3
    pts = pts_raw[mask]

    ds = np.r_[0.0, np.linalg.norm(pts[1:] - pts[:-1], axis=1)]
    s = np.cumsum(ds)

    speed_path_kmh = np.asarray(speed_kmh, dtype=float)[mask]
    return pts, s, speed_path_kmh


# ===================== Display / camera =====================

@dataclass
class CameraDisplay:
    surface: pygame.Surface | None = None


def process_image(image: carla.Image, weak_ref):
    self = weak_ref()
    if self is None:
        return
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


# ===================== Stopping profile helper =====================

def stop_profile_within_distance(
    s: np.ndarray,
    i_anchor: int,
    v_ref_kmh: float,
    a_comf: float = 2.0,
) -> float:
    """
    Compute a speed (km/h) that can safely come to a stop within the
    remaining path distance using a comfortable deceleration.
    """
    s_rem = max(0.0, float(s[-1] - s[i_anchor]))
    v_ref_ms = max(0.0, float(v_ref_kmh) / 3.6)
    v_stop_ms = math.sqrt(max(0.0, 2.0 * a_comf * s_rem))
    v_ms = min(v_ref_ms, v_stop_ms)
    return 3.6 * v_ms


# ===================== Main playback loop =====================

def main() -> None:
    from recovery_controller import VehiclePIDController

    # ---- Load state reconstruction results generated from RMHE----
    df = pd.read_csv("state_reconstruction_results.csv")

    # Speed in the state reconstruction results is assumed to be in m/s; convert to km/h
    recon_speed_kmh = df["Speed"].to_numpy(dtype=float) * 3.6

    # Build path and speed profile from state reconstruction results
    pts, s, speed_path_kmh = build_path_from_reconstruction(df, recon_speed_kmh)

    # ---- UI / client setup ----
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF,
    )
    pygame.display.set_caption("CARLA - State Reconstruction Playback Controller")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    prev_settings = world.get_settings()
    settings = carla.WorldSettings(
        no_rendering_mode=prev_settings.no_rendering_mode,
        synchronous_mode=True,
        fixed_delta_seconds=1.0 / TICK_HZ,
        substepping=prev_settings.substepping,
        max_substep_delta_time=prev_settings.max_substep_delta_time,
        max_substeps=prev_settings.max_substeps,
    )
    world.apply_settings(settings)
    dt = settings.fixed_delta_seconds or (1.0 / TICK_HZ)

    COLLISION_HOLD_FRAMES = max(1, int(COLLISION_HOLD_SEC / dt))
    MAX_RUNTIME_FRAMES = int(MAX_RUNTIME_SEC / dt) if MAX_RUNTIME_SEC > 0 else 0

    # ---- Vehicle and camera ----
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    spawn_point = world.get_map().get_spawn_points()[SPAWN_INDEX]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print("Vehicle spawn failed.")
        world.apply_settings(prev_settings)
        pygame.quit()
        sys.exit(1)

    def ensure_drive(ctrl: carla.VehicleControl) -> carla.VehicleControl:
        ctrl.hand_brake = False
        ctrl.reverse = False
        ctrl.manual_gear_shift = False
        return ctrl

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(WIDTH))
    camera_bp.set_attribute("image_size_y", str(HEIGHT))
    camera_bp.set_attribute("fov", str(FOV))
    camera_transform = carla.Transform(
        carla.Location(x=-6.0, z=3.0),
        carla.Rotation(pitch=-15.0),
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    cam_disp = CameraDisplay()
    camera.listen(lambda image: process_image(image, weakref.ref(cam_disp)))

    # Collision sensor
    collision_bp = blueprint_library.find("sensor.other.collision")
    collision_sensor = world.spawn_actor(
        collision_bp, carla.Transform(), attach_to=vehicle
    )
    collision_info = {"tripped": False, "impulse": 0.0, "other": "", "frames": 0}

    def on_collision(event: carla.CollisionEvent):
        imp = math.sqrt(
            event.normal_impulse.x**2
            + event.normal_impulse.y**2
            + event.normal_impulse.z**2
        )
        if imp >= COLLISION_IMPULSE_MIN:
            collision_info["tripped"] = True
            collision_info["impulse"] = imp
            collision_info["other"] = getattr(event.other_actor, "type_id", "")

    collision_sensor.listen(on_collision)

    # IMU sensor
    imu_bp = blueprint_library.find("sensor.other.imu")
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    imu_state = {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0}

    def on_imu(meas):
        try:
            imu_state["ax"] = float(meas.accelerometer.x)
            imu_state["ay"] = float(meas.accelerometer.y)
            imu_state["az"] = float(meas.accelerometer.z)
            imu_state["gx"] = float(meas.gyroscope.x)
            imu_state["gy"] = float(meas.gyroscope.y)
            imu_state["gz"] = float(meas.gyroscope.z)
        except Exception:
            pass

    imu_sensor.listen(on_imu)

    # ---- Controllers and filters ----
    pid_controller = VehiclePIDController(
        vehicle,
        LAT_ARGS,
        LON_ARGS,
        offset=0.0,
        max_throttle=0.85,
        max_brake=1.0,
        max_steering=STEER_ABS_LIMIT,
    )

    steer_lpf = FirstOrderFilter(tau=STEER_TAU, dt=dt, init=0.0)
    steer_rl = RateLimiter(max_rate_per_sec=STEER_RATE_LIMIT, dt=dt, init=0.0)
    speed_pid = AntiWindupPID(
        kp=LON_ARGS["K_P"],
        ki=LON_ARGS["K_I"],
        kd=LON_ARGS["K_D"],
        dt=dt,
        u_min=-1.0,
        u_max=1.0,
    )

    # ---- Logs ----
    control_history: list[dict[str, Any]] = []
    trajectory_history: list[dict[str, Any]] = []
    speed_log: list[dict[str, Any]] = []
    position_log: list[dict[str, Any]] = []
    debug_log: list[dict[str, Any]] = []

    idx_recon = 0  # index into state reconstruction results
    frames_run = 0

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            frame = world.tick()
            frames_run += 1
            if MAX_RUNTIME_FRAMES and frames_run >= MAX_RUNTIME_FRAMES:
                print("[AUTO-STOP] Max runtime reached. Exiting.")
                break

            snapshot = world.get_snapshot()

            # Current state
            tf = vehicle.get_transform()
            vel = vehicle.get_velocity()
            ang = vehicle.get_angular_velocity()

            cur_xy = np.array([tf.location.x, tf.location.y], dtype=float)
            cur_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            v_ms = cur_speed / 3.6

            # Target index from state reconstruction results
            i_anchor = min(max(idx_recon, 0), len(s) - 1)
            base_target = float(speed_path_kmh[i_anchor])
            if FOLLOW_SPEED_ONLY:
                target_speed = base_target
            else:
                target_speed = stop_profile_within_distance(
                    s, i_anchor, base_target, a_comf=2.0
                )

            # Lookahead point along the reconstructed path
            Ld = LOOKAHEAD_BASE + LOOKAHEAD_K * max(0.0, v_ms)
            s_target_recon = s[i_anchor] + Ld
            if s_target_recon <= s[-1]:
                j = int(np.searchsorted(s, s_target_recon, side="left"))
                j = min(j, len(pts) - 1)
                target_xy = pts[j]
            else:
                if len(pts) >= 2:
                    tvec = pts[-1] - pts[-2]
                    n = float(np.linalg.norm(tvec))
                    if n < 1e-6:
                        target_xy = pts[-1]
                    else:
                        target_xy = pts[-1] + tvec / n * (s_target_recon - s[-1])
                else:
                    target_xy = pts[-1]

            target_loc = carla.Location(
                x=float(target_xy[0]),
                y=float(target_xy[1]),
                z=spawn_point.location.z,
            )

            # Lateral control from high-level PID
            raw = pid_controller.run_step(target_speed, target_loc)

            # Longitudinal control with AntiWindupPID
            u = speed_pid.step(target_speed, cur_speed)
            throttle = clamp(u, 0.0, 1.0)
            brake = clamp(-u, 0.0, 1.0)

            # Kickstart at very low speed if target_speed is non-zero
            if (target_speed > 1.0) and (cur_speed < KICKSTART_SPEED_KMH):
                throttle = max(throttle, MIN_KICKSTART_THROTTLE)
                brake = 0.0

            # Optional collision-based stopping
            if STOP_ON_COLLISION and collision_info["tripped"]:
                collision_info["frames"] += 1
                throttle = 0.0
                brake = 1.0
                if collision_info["frames"] >= COLLISION_HOLD_FRAMES:
                    print(
                        f"[AUTO-STOP] Collision impulse={collision_info['impulse']:.1f} "
                        f"with {collision_info['other']}. Exiting."
                    )
                    break

            # Steering shaping
            steer_cmd = clamp(raw.steer, -1.0, 1.0)
            steer_cmd = steer_lpf.step(steer_cmd)
            steer_cmd = steer_rl.step(steer_cmd)
            steer_cmd = clamp(steer_cmd, -STEER_ABS_LIMIT, STEER_ABS_LIMIT)

            raw.throttle = throttle
            raw.brake = brake
            raw.steer = steer_cmd
            raw = ensure_drive(raw)
            vehicle.apply_control(raw)
            # HUD
            if getattr(cam_disp, "surface", None) is not None:
                display.blit(cam_disp.surface, (0, 0))
            hud_text = [
                f"Speed: {cur_speed:.1f} km/h",
                f"Target: {target_speed:.1f} km/h",
                f"Step: {i_anchor+1}/{len(pts)}",
            ]
            for i, txt in enumerate(hud_text):
                display.blit(
                    font.render(txt, True, (255, 255, 255)),
                    (10, 10 + 30 * i),
                )
            pygame.display.flip()
            clock.tick(TICK_HZ)

    except KeyboardInterrupt:
        pass
    finally:

        # Cleanup
        try:
            collision_sensor.stop()
        except Exception:
            pass
        try:
            imu_sensor.stop()
        except Exception:
            pass
        try:
            collision_sensor.destroy()
        except Exception:
            pass
        try:
            imu_sensor.destroy()
        except Exception:
            pass
        try:
            camera.stop()
        except Exception:
            pass
        try:
            camera.destroy()
        except Exception:
            pass
        try:
            vehicle.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(prev_settings)
        except Exception:
            pass
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()

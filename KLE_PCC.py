from collections import deque
import math
import numpy as np
import carla
from misc import get_speed


class VehiclePIDController:
    """Combine longitudinal & lateral PID."""

    def __init__(
        self,
        vehicle,
        args_lateral,
        args_longitudinal,
        offset=0.0,
        max_throttle=0.75,
        max_brake=0.3,
        max_steering=0.8,
    ):
        self._vehicle = vehicle
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_steering = max_steering

        self._lon_controller = PIDLongitudinalController(
            vehicle, **args_longitudinal
        )
        self._lat_controller = PIDLateralController(
            vehicle, offset=offset, **args_lateral
        )

    def run_step(self, target_speed, target):
        # speed control
        accel_cmd = self._lon_controller.run_step(target_speed)

        # steering control
        if isinstance(target, carla.Waypoint):
            steer_cmd = self._lat_controller.run_step_waypoint(target)
        elif isinstance(target, carla.Location):
            steer_cmd = self._lat_controller.run_step_location(target)
        else:
            raise ValueError("Target must be carla.Waypoint or carla.Location")

        control = carla.VehicleControl()

        # map accel_cmd ∈ [-1,1] to throttle / brake
        if accel_cmd >= 0.0:
            control.throttle = min(accel_cmd, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel_cmd, self.max_brake)

        # saturate steering
        steer_cmd = float(steer_cmd)
        steer_cmd = max(-self.max_steering, min(self.max_steering, steer_cmd))
        control.steer = steer_cmd

        control.hand_brake = False
        control.manual_gear_shift = False
        return control


class PIDLongitudinalController:
    """PID on speed."""

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        current_speed = get_speed(self._vehicle)
        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        error = float(target_speed - current_speed)
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            ie = sum(self._error_buffer) * self._dt
        else:
            de = 0.0
            ie = 0.0

        cmd = self._k_p * error + self._k_d * de + self._k_i * ie
        return float(np.clip(cmd, -1.0, 1.0))


class PIDLateralController:
    """PID on heading / lateral error toward target location."""

    def __init__(self, vehicle, offset=0.0, K_P=6.0, K_I=0.0, K_D=0.1, dt=0.02):
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step_waypoint(self, waypoint):
        return self._pid_control(
            waypoint.transform.location, self._vehicle.get_transform()
        )

    def run_step_location(self, location):
        return self._pid_control(location, self._vehicle.get_transform())

    def _pid_control(self, target_location, vehicle_transform):
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0], dtype=float)

        w_vec = np.array(
            [target_location.x - ego_loc.x,
             target_location.y - ego_loc.y,
             0.0],
            dtype=float,
        )

        denom = np.linalg.norm(v_vec) * np.linalg.norm(w_vec)
        if denom < 1e-6:
            angle = 0.0
        else:
            cos_angle = np.clip(np.dot(v_vec, w_vec) / denom, -1.0, 1.0)
            angle = math.acos(cos_angle)

        cross = np.cross(v_vec, w_vec)
        if cross[2] < 0:
            angle *= -1.0

        angle += self._offset

        self._e_buffer.append(angle)
        if len(self._e_buffer) >= 2:
            de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            ie = sum(self._e_buffer) * self._dt
        else:
            de = 0.0
            ie = 0.0

        steer = self._k_p * angle + self._k_d * de + self._k_i * ie
        return float(np.clip(steer, -1.0, 1.0))

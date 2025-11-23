# COREcover-a-unifed-recovery-framework


This repository provides a **implementation of three modules, detection module, state reconstruction module, recovery module of COREcover** for robotic / autonomous vehicles under potential sensor attacks.


---

## Features of state reconstruction module

- **6D vehicle state model**
  - State: \\( x = [x, y, v, \psi, a_x, \omega]^T \\)
  - Bicycle-style kinematics with slip, yaw-rate dynamics, and drag

- **Drag parameter fitting from benign data**
  - `fit_drag_params_collocation_xyv(...)`
  - Uses collocation on \\( (x, y, v) \\) trajectories
  - Optimizes drag parameters \\( (c_2, c_1, c_0) \\) and virtual controls

- **RMHE with sparse sensor attacks**
  - `rmhe_with_inputs_masked(...)`
  - `run_receding_mhe(...)` (sliding / online)
  - Each measurement channel has its own sparse attack variable
  - Can **ignore untrusted control inputs during the attack** and instead close the loop using the estimated state

- **Attack-awareness via control mask**
  - Before the attack: trust control inputs
  - During / after the attack: optionally treat controls as untrusted and infer them from the states

---

## Installation

This project uses Python and standard scientific libraries.

### Requirements

- Python
- `numpy`
- `pandas`
- `scipy`
- `math`
- `carla`
- `os`
- `time`
- `scipy.optimize`

You can install dependencies via:

```bash
pip install numpy pandas scipy

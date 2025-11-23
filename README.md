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


## CARLA 0.9.15 Installation

Below are the steps to install and configure CARLA 0.9.15 on Windows.

### 1. Prerequisites

- OS: Windows 10 / 11 (64-bit)
- GPU: NVIDIA GPU recommended (for better rendering)
- Python: 3.7.x
- Visual C++ Redistributable for Visual Studio 2015–2019
- Git, pip, etc.

### 2. Download CARLA 0.9.15

1. Go to the official CARLA release page, download and unzip it
   https://github.com/carla-simulator/carla/releases
2. Start CARLA by `CarlaUE4.exe -windowed -carla-port=2000`
3. Test the Python API by `python manual_control.py`

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
- `sys`
- `weakref`
- `argparse`

You can install dependencies via:

```bash
pip install numpy pandas scipy

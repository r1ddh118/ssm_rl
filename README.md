# Efficient Model-Based Reinforcement Learning with Structured State-Space World Models

> B.Tech AIML Reinforcement Learning Course Project
> Cross-platform target: macOS Apple Silicon or Linux Fedora with NVIDIA GPU
> Tested with Python 3.13 and the current dependency set in `requirements.txt`

## Project Overview

This repo trains reinforcement learning baselines on DeepMind Control Suite tasks using a custom `dm_control` to Gymnasium wrapper. The setup now supports both:

- macOS with Apple Silicon using `mps`
- Linux with NVIDIA GPUs using `cuda`

If no accelerator is available, the scripts fall back to CPU automatically.

## Project Structure

```text
rl/
├── main.py                 # Unified CLI entry point
├── run_layout.py           # Shared run directory initialization
├── artifact_logging.py     # JSONL artifact callback
├── device_utils.py         # Cross-platform device detection: CUDA, MPS, CPU
├── env_setup.py            # dm_control -> Gymnasium wrapper
├── train_ppo_mac.py        # PPO training task
├── train_sac_mac.py        # SAC training task
├── tdmpc2/
│   ├── model.py            # Phase 1 TD-MPC2 world model components
│   ├── replay_buffer.py    # Sequence replay buffer
│   ├── trainer.py          # TD-MPC2 training loop and MPPI planner
│   └── train_tdmpc2.py     # TD-MPC2 training entry point
├── test_env.py             # Lightweight environment smoke test
├── plot_results.py         # Plot saved evaluation curves
├── requirements.txt        # Python dependencies
├── logs/                   # Model checkpoints and evaluation outputs
│   ├── ppo_walker/
│   └── sac_cheetah/
├── artifacts/              # Structured run artifacts
│   ├── ppo_walker/
│   └── sac_cheetah/
└── README.md
```

## What Changed For Cross-Platform Support

- Training now auto-selects `cuda`, then `mps`, then `cpu`
- The code no longer depends on `dmc2gym`
- Each run now initializes its own labeled directories under `logs/` and `artifacts/`
- Plotting works with both new cross-platform logs and older `_mac` logs
- The environment smoke test avoids viewer/render assumptions that often break on Linux

## Fedora + NVIDIA Setup

These steps fit your ASUS ProArt PX13 with Ryzen 9 and RTX 4050 laptop.

### 1. Install system packages

```bash
sudo dnf update -y
sudo dnf install -y python3 python3-virtualenv git gcc gcc-c++ make \
    glfw glfw-devel mesa-libGL mesa-libGL-devel libXcursor libXi libXinerama libXrandr
```

If your NVIDIA drivers are already installed and `nvidia-smi` works, keep using them. If not, install the correct Fedora NVIDIA driver stack first.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch with CUDA support

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio
```

### 4. Install the remaining Python dependencies

```bash
pip install --no-deps dm_control
pip install -r requirements.txt
```

If you previously hit a `labmaze` build error, rerun with:

```bash
pip uninstall -y labmaze dm_control shimmy
pip install --no-deps dm_control
pip install -r requirements.txt
```

### 5. Verify CUDA and MuJoCo

```bash
python - <<'PY'
import torch
import mujoco
from device_utils import describe_device
from env_setup import make_env

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Selected device:", describe_device())
print("MuJoCo:", mujoco.__version__)

env = make_env("walker", "walk")
obs = env.reset()
print("Env OK, obs shape:", obs[0].shape)
PY
```

Expected on your laptop:

- `CUDA available: True`
- selected device should show `cuda`

## macOS Apple Silicon Setup

### 1. Install system packages

```bash
brew install python glfw git
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install Python packages

```bash
pip install --no-deps dm_control
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Verify MPS and MuJoCo

```bash
python - <<'PY'
import torch
import mujoco
from device_utils import describe_device
from env_setup import make_env

print("Torch:", torch.__version__)
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("Selected device:", describe_device())
print("MuJoCo:", mujoco.__version__)

env = make_env("walker", "walk")
obs = env.reset()
print("Env OK, obs shape:", obs[0].shape)
PY
```

## Run The Project

From the repo root after activating the virtual environment:

### Smoke test the environment

```bash
python main.py test
```

### Train PPO on Walker-Walk

```bash
python main.py ppo
```

The script stays quiet during training and writes artifacts to:

- `artifacts/ppo_walker/metrics.jsonl`
- `artifacts/ppo_walker/summary.json`

Default training length: `50,000` timesteps.

### Train SAC on Cheetah-Run

```bash
python main.py sac
```

The script stays quiet during training and writes artifacts to:

- `artifacts/sac_cheetah/metrics.jsonl`
- `artifacts/sac_cheetah/summary.json`

Default training length: `100,000` timesteps.

### Plot results

```bash
python main.py plot
```

### Train TD-MPC2 with MLP dynamics on Walker-Walk

```bash
python main.py tdmpc
```

To train the same Phase 1 baseline on other domains directly:

```bash
python -m tdmpc2.train_tdmpc2 --env-name cheetah
python -m tdmpc2.train_tdmpc2 --env-name hopper
```

The training scripts print the selected runtime device automatically.

## Output Layout

Each algorithm now gets its own labeled directory tree:

```text
logs/
├── ppo_walker/
│   ├── best/
│   ├── eval/
│   └── final_model.zip
└── sac_cheetah/
    ├── best/
    ├── eval/
    └── final_model.zip

artifacts/
├── ppo_walker/
│   ├── metrics.jsonl
│   └── summary.json
└── sac_cheetah/
    ├── metrics.jsonl
    └── summary.json
```

## Notes For Linux Rendering

For normal training, no special rendering setup is required.

If you later add visual rendering on Linux and hit OpenGL issues, these environment variables are the usual fixes:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Use those only when needed for headless or EGL-backed rendering.

# EGR-PO: Effective Goal-Reaching Policy Optimization

Official PyTorch implementation of **"Goal-Reaching Policy Learning from Non-Expert Observations via Effective Subgoal Guidance"** (CoRL 2024).

EGR-PO learns goal-reaching policies from offline datasets that contain only non-expert (sub-optimal) trajectories. A hierarchical architecture automatically discovers waypoints as intermediate subgoals, guiding both offline pretraining and online fine-tuning toward the desired goal.

---

## Overview

Most offline goal-conditioned RL methods assume access to expert demonstrations. EGR-PO relaxes this requirement by learning a **two-level hierarchy**:

- **High-level policy** — a diffusion model that generates an intermediate subgoal (waypoint) on the path from the current state to the desired goal.
- **Low-level policy** — a Gaussian policy trained with IQL-style advantage-weighted regression to reach the predicted subgoal.

Both levels are pretrained offline with goal-conditioned value functions. The low-level policy can then be further improved online with a guided SAC agent that uses intrinsic rewards derived from the pretrained value function.

---

## Method

### Offline Pretraining (`pretrain.py`)

1. **Dataset loading** — D4RL or custom offline datasets are loaded and wrapped in a `GCSDataset` that samples goals using a mixture of random, trajectory, and current-state goals.
2. **Value learning** — A twin value function is trained with expectile regression on Bellman targets:

   ```
   V(s, g) ← r + γ · V(s', g)    (expectile loss weighted by advantage sign)
   ```

3. **Low-level actor** — A Gaussian policy is updated with exponential advantage weighting:

   ```
   π(a|s, s_sub) ← exp(β · (V(s', s_sub) − V(s, s_sub))) · log π(a|s, s_sub)
   ```

4. **High-level actor (subgoal generator)** — A diffusion model learns to propose the next waypoint `s_sub` conditioned on the current state and the final goal, guided by the same advantage signal.

### Online Fine-tuning (`online_train.py`)

After offline pretraining, the low-level policy is further refined with a guided SAC agent:

- The pretrained **high-level diffusion policy** proposes subgoals at every step.
- An **intrinsic reward** is computed as the value improvement `sigmoid(V(s', s_sub) − V(s, s_sub))`.
- A separate **goal-critic** provides auxiliary signal for subgoal-directed learning.
- **HER (Hindsight Experience Replay)** relabels episode goals for sample efficiency.

---

## Repository Structure

```
EGR-PO/
├── pretrain.py              # Offline pretraining entry point
├── online_train.py          # Online fine-tuning entry point
├── model.py                 # SAC Actor / Double-Q-Critic (PyTorch)
├── sac_agent.py             # Guided SAC agent + HER replay buffer
│
├── src/
│   ├── agents/
│   │   ├── hdql.py          # JointTrainAgent: value + actor + high-actor losses
│   │   └── iql.py           # IQL base class (for reference)
│   ├── special_networks.py  # HierarchicalActorCritic, RelativeRepresentation, MonolithicVF
│   ├── gc_dataset.py        # GCSDataset: goal sampling with waypoint support
│   ├── d4rl_utils.py        # Dataset loading, environment creation
│   ├── utils.py             # CSV logger, video recording
│   └── envs/                # Calvin, Procgen, Kitchen environment wrappers
│
├── dql/
│   └── agents/
│       ├── ql_diffusion.py  # Diffusion_QL: diffusion-based subgoal generator
│       ├── diffusion.py     # Core DDPM diffusion sampling
│       ├── model.py         # MLP with sinusoidal time embeddings
│       └── helpers.py       # EMA, beta schedules, loss functions
│
├── rl_m/                    # RL backbone: networks, dataset, evaluation utilities
├── d4rl_ext/                # D4RL environment extensions (AntMaze-ultra, etc.)
├── datasets/                # Additional dataset utilities and normalizers
└── script/                  # Ready-to-run training scripts per environment
```

---

## Installation

### Core dependencies

```bash
pip install torch torchvision
pip install gym d4rl
pip install wandb tqdm ml_collections absl-py
```

### Optional environment dependencies

**MetaWorld** (for Sawyer tasks):
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git
```

**WGCSL** (for Sawyer environments):
```bash
pip install git+https://github.com/YangRui2015/AWGCSL.git
```

**CALVIN** (for robot manipulation):
Follow the [CALVIN installation guide](https://github.com/mees/calvin).

---

## Quickstart

### 1. Offline Pretraining

```bash
# AntMaze Large Diverse
python pretrain.py \
  --env_name antmaze-large-diverse-v2 \
  --pretrain_steps 1000000 \
  --use_waypoints 1 \
  --way_steps 25 \
  --value_hidden_dim 512 \
  --value_num_layers 3 \
  --use_layer_norm 1 \
  --batch_size 1024 \
  --save_interval 250000
```

Or use a pre-written script:
```bash
bash script/antmaze_large_diverse.sh
```

### 2. Online Fine-tuning

```bash
python online_train.py \
  --env_name antmaze-large-diverse-v2 \
  --pretrain_path <path/to/params_XXXXXX.pkl> \
  --use_waypoints 1 \
  --way_steps 25 \
  --use_rep 1 \
  --rep_dim 10
```

Or use the universal online script:
```bash
env_name=antmaze-large-diverse-v2 \
pretrain_path=<path> \
bash script/online_universe.sh
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `use_waypoints` | `1` | Enable hierarchical waypoint guidance |
| `way_steps` | `25` | Steps ahead to predict as waypoint |
| `p_trajgoal` | `0.5` | Fraction of goals sampled from the same trajectory |
| `p_randomgoal` | `0.3` | Fraction of goals sampled randomly from dataset |
| `p_currgoal` | `0.2` | Fraction of goals set to the current state (self-play) |
| `high_p_randomgoal` | `0.3` | Random goal fraction for the high-level policy |
| `pretrain_expectile` | `0.7` | Expectile for value learning (>0.5 emphasises upper quantile) |
| `temperature` | `1.0` | Advantage temperature for low-level actor |
| `high_temperature` | `1.0` | Advantage temperature for high-level actor |
| `discount` | `0.99` | RL discount factor |
| `value_hidden_dim` | `512` | Width of value / actor MLP hidden layers |
| `value_num_layers` | `3` | Depth of value / actor MLP |
| `use_layer_norm` | `1` | Enable LayerNorm in all networks |
| `use_rep` | `0` | Learn a compact goal representation (bottleneck) |
| `rep_dim` | `10` | Bottleneck representation dimension |
| `rep_type` | `state` | Goal encoding: `state`, `diff` (relative), or `concat` |

---

## Supported Environments

| Environment | Dataset source | Notes |
|---|---|---|
| `antmaze-{umaze,medium,large,ultra}-*-v2` | D4RL | Primary benchmark |
| `kitchen-{partial,mixed,complete}-v0` | D4RL | Multi-task manipulation |
| `maze2d-{umaze,medium,large}-v1` | D4RL | Dense maze navigation |
| `FetchReach-v1`, `FetchPush-v1`, `FetchPickAndPlace-v1`, `FetchSlide-v1` | Custom `.npy` | Requires `--load_path` |
| `SawyerReach`, `SawyerDoor` | Custom `.npy` | Requires WGCSL + `--load_path` |
| `calvin` | Custom `.gz` | Requires CALVIN install |
| `procgen-500`, `procgen-1000` | Custom `.npz` | Procedural maze levels |

---

## Citation

```bibtex
@inproceedings{egrpo2024,
  title     = {Goal-Reaching Policy Learning from Non-Expert Observations via Effective Subgoal Guidance},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2024},
}
```

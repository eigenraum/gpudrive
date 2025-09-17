# Random Map Training in GPUDrive

This document describes how to train policies on procedurally generated maps
instead of relying on pre-recorded traffic scenarios. The random map generator
creates European-inspired urban grids as well as multi-lane highway layouts and
initializes agents with realistic poses and velocities.

## Key Components

- `gpudrive.env.random_map.generate_random_map(map_type)`: produces a single
  map dictionary (`"urban"` or `"highway"`).
- `gpudrive.env.random_map.RandomMapDataLoader`: yields batches of on-disk JSON
  scenes that the simulator can load directly.
- Generated agents are spaced along the road network to avoid collisions and
  receive velocities consistent with the selected road type.

## Quick Start

1. **Create a data loader for random maps**

```python
from gpudrive.env import RandomMapDataLoader

random_loader = RandomMapDataLoader(
    batch_size=8,          # number of parallel worlds
    dataset_size=64,       # maps regenerated after the loader is exhausted
    map_types=("urban", "highway"),
    seed=0,
)
```

2. **Instantiate the environment with the loader**

```python
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv

config = EnvConfig()
env = GPUDriveTorchEnv(
    config=config,
    data_loader=random_loader,
    max_cont_agents=64,
    device="cuda"  # or "cpu" if CUDA is unavailable
)
```

3. **Run a short training or rollout**

```python
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated.any() or truncated.any():
        obs, _ = env.reset()
```

The environment will automatically regenerate fresh random maps whenever the
loader is exhausted. The generated files are cached in
`data/generated/random_maps/` (configurable via the loader).

## Tips

- Restrict the generator to a single mode by passing `map_types=("urban",)` or
  `map_types=("highway",)`.
- Increase `dataset_size` for more unique scenes before regeneration.
- To reuse generated scenes across runs, specify `output_dir` when constructing
  `RandomMapDataLoader`.
- The loader mirrors the behaviour of `SceneDataLoader`, so you can drop it into
  existing training scripts without further changes.

Happy training!

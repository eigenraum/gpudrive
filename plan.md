# Plan for Random Map Generation and Training Support

## Context and Key Components
- **Simulator bindings** (`src/`): `MapReader`, `level_gen.cpp`, and the C++ `Map`/`MapObject` structs describe how map JSON files are parsed and turned into simulator entities.
- **Python environment** (`gpudrive/env/env_torch.py`, `base_env.py`): the gym wrapper feeds scene paths to the simulator through a `SceneDataLoader`.
- **Existing dataset loader** (`gpudrive/env/dataset.py`): expects on-disk JSON scenes and batches their file paths for the simulator.

## Goals
1. Programmatically generate realistic random maps representing European-style urban grids and highways, including appropriate road geometries.
2. Populate each generated map with non-overlapping agents placed on the road network with plausible headings and initial velocities.
3. Provide a data loader / helper that can yield an arbitrary number of such random maps to the simulator.
4. Integrate the generator so the gym environment can swap to random-map training without manual JSON files.
5. Document how to launch training with the random map generator and provide a runnable smoke test that exercises the new mode.

## Implementation Steps
1. **Design map abstractions** in a new Python module (e.g. `gpudrive/env/random_map.py`) to create lane polylines, road edges, and crosswalks. Implement utilities for rotating/jittering intersections to mimic European city layouts and for constructing multi-lane highways.
2. **Agent initialization**: add helper functions that interpolate positions along lanes, compute headings, assign realistic speeds (urban vs. highway), and guarantee spacing to avoid collisions.
3. **Random map dataset loader**: implement a `RandomMapDataLoader` mirroring the interface of `SceneDataLoader`, pre-generating the requested number of scenes into a cache directory and yielding batches of file paths.
4. **Environment integration**: allow `GPUDriveTorchEnv` (and other entry points if needed) to accept the new loader so `swap_data_batch` works unchanged.
5. **Testing / validation**: create a short script or unit test that instantiates the env with the random loader, runs a brief rollout with sampled actions, and ensure no crashes. Use this as the requested “test training”.
6. **Documentation**: author `README_randommap.md` with setup instructions and code snippets showing how to train with random maps.

## Validation Plan
- Run the new smoke-test training command to verify map generation and agent initialization succeed end-to-end.
- Execute existing test suite (`pytest`) if feasible or at least targeted tests touching the new components.

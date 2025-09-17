"""Utilities for procedurally generating random urban and highway maps.

The generator produces map dictionaries that follow the JSON schema expected by
GPUDrive and saves them as standalone scene files. A lightweight data loader is
provided to integrate the generated scenes with the existing simulator API.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

LANE_WIDTH_METERS = 3.6
URBAN_SPEED_RANGE = (6.0, 14.0)  # m/s -> roughly 20-50 km/h
HIGHWAY_SPEED_RANGE = (20.0, 33.0)  # m/s -> roughly 70-120 km/h
URBAN_CLEARANCE = 9.0
HIGHWAY_CLEARANCE = 18.0
URBAN_PADDING = 14.0
HIGHWAY_PADDING = 30.0
DEFAULT_TIMESTEP_S = 0.6
HIGHWAY_TIMESTEP_S = 0.5


@dataclass
class LaneDefinition:
    """Representation of a driveline polyline and its traversal metadata."""

    id: int
    points: List[Tuple[float, float]]
    map_element_id: int
    entity_type: str
    speed_range: Tuple[float, float]
    _segment_lengths: List[float] = field(init=False, repr=False)
    _total_length: float = field(init=False, repr=False)
    occupied_distances: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("LaneDefinition requires at least two polyline points")

        self._segment_lengths = []
        total = 0.0
        for start, end in zip(self.points[:-1], self.points[1:]):
            seg_length = math.dist(start, end)
            self._segment_lengths.append(seg_length)
            total += seg_length
        self._total_length = total

    @property
    def total_length(self) -> float:
        return self._total_length

    def point_and_heading_at(self, distance: float) -> Tuple[Tuple[float, float], float]:
        """Return coordinates and heading (radians) at a distance along the lane."""

        if distance <= 0:
            start, nxt = self.points[0], self.points[1]
            heading = math.atan2(nxt[1] - start[1], nxt[0] - start[0])
            return start, heading

        remaining = distance
        last_heading = math.atan2(
            self.points[1][1] - self.points[0][1], self.points[1][0] - self.points[0][0]
        )

        for idx, seg_length in enumerate(self._segment_lengths):
            start = self.points[idx]
            end = self.points[idx + 1]
            if seg_length <= 1e-6:
                continue

            if remaining <= seg_length:
                ratio = remaining / seg_length
                x = start[0] + (end[0] - start[0]) * ratio
                y = start[1] + (end[1] - start[1]) * ratio
                heading = math.atan2(end[1] - start[1], end[0] - start[0])
                return (x, y), heading

            remaining -= seg_length
            last_heading = math.atan2(end[1] - start[1], end[0] - start[0])

        return self.points[-1], last_heading

    def try_reserve_distance(
        self, rng: random.Random, min_clearance: float, padding: float
    ) -> Optional[float]:
        """Sample a distance along the lane ensuring minimum spacing."""

        if self._total_length < 2 * padding:
            return None

        for _ in range(32):
            distance = rng.uniform(padding, self._total_length - padding)
            if all(abs(distance - other) >= min_clearance for other in self.occupied_distances):
                self.occupied_distances.append(distance)
                return distance

        return None


def _offset_polyline(points: Sequence[Tuple[float, float]], offset: float) -> List[Tuple[float, float]]:
    """Offset a polyline by a constant amount using local normals."""

    if len(points) < 2:
        return list(points)

    offset_points: List[Tuple[float, float]] = []
    for idx, (x, y) in enumerate(points):
        if idx == 0:
            dx = points[1][0] - x
            dy = points[1][1] - y
        elif idx == len(points) - 1:
            dx = x - points[idx - 1][0]
            dy = y - points[idx - 1][1]
        else:
            dx = points[idx + 1][0] - points[idx - 1][0]
            dy = points[idx + 1][1] - points[idx - 1][1]

        length = math.hypot(dx, dy)
        if length <= 1e-6:
            offset_points.append((x, y))
            continue

        nx = -dy / length
        ny = dx / length
        offset_points.append((x + nx * offset, y + ny * offset))

    return offset_points


def _polyline_with_curvature(
    start: Tuple[float, float],
    end: Tuple[float, float],
    rng: random.Random,
    curvature_ratio: float,
) -> List[Tuple[float, float]]:
    """Create a slightly curved polyline between two points."""

    points = [start]
    if curvature_ratio > 0:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 1e-6:
            mid_fraction = rng.uniform(0.35, 0.65)
            mx = start[0] + dx * mid_fraction
            my = start[1] + dy * mid_fraction
            nx = -dy / length
            ny = dx / length
            amplitude = curvature_ratio * length * rng.uniform(-0.4, 0.4)
            mx += nx * amplitude
            my += ny * amplitude
            points.append((mx, my))
    points.append(end)
    return points


def _round_point(point: Tuple[float, float]) -> dict:
    return {"x": round(point[0], 3), "y": round(point[1], 3)}


class RandomMapBuilder:
    def __init__(self, rng: random.Random, map_type: str, seed: Optional[int] = None) -> None:
        self.rng = rng
        self.map_type = map_type.lower()
        self.seed = seed if seed is not None else rng.randint(0, 2**31 - 1)
        self.roads: List[dict] = []
        self.lanes: List[LaneDefinition] = []
        self.road_id_counter = 0

    def _next_road_id(self) -> int:
        self.road_id_counter += 1
        return self.road_id_counter

    def _add_lane(
        self,
        points: Sequence[Tuple[float, float]],
        map_element_id: int,
        speed_range: Tuple[float, float],
    ) -> None:
        lane = LaneDefinition(
            id=self._next_road_id(),
            points=list(points),
            map_element_id=map_element_id,
            entity_type="lane",
            speed_range=speed_range,
        )
        self.lanes.append(lane)
        self.roads.append(
            {
                "geometry": [_round_point(p) for p in lane.points],
                "type": "lane",
                "map_element_id": map_element_id,
                "id": lane.id,
            }
        )

    def _add_road_edge(self, points: Sequence[Tuple[float, float]]) -> None:
        self.roads.append(
            {
                "geometry": [_round_point(p) for p in points],
                "type": "road_edge",
                "map_element_id": 15,
                "id": self._next_road_id(),
            }
        )

    def _add_road_line(self, points: Sequence[Tuple[float, float]], element_id: int) -> None:
        self.roads.append(
            {
                "geometry": [_round_point(p) for p in points],
                "type": "road_line",
                "map_element_id": element_id,
                "id": self._next_road_id(),
            }
        )

    # ------------------------------------------------------------------
    # Network construction helpers
    # ------------------------------------------------------------------
    def _build_urban_network(self) -> None:
        rows = self.rng.randint(3, 5)
        cols = self.rng.randint(3, 5)

        def _axis_positions(num: int) -> List[float]:
            distances = [self.rng.uniform(45.0, 90.0) for _ in range(num - 1)]
            coords = [0.0]
            for dist in distances:
                coords.append(coords[-1] + dist)
            total = coords[-1]
            centered = [c - total / 2 for c in coords]
            return centered

        x_coords = _axis_positions(cols)
        y_coords = _axis_positions(rows)

        jitter_x = (max(x_coords) - min(x_coords)) * 0.08
        jitter_y = (max(y_coords) - min(y_coords)) * 0.08
        angle = math.radians(self.rng.uniform(-18, 18))
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        intersections: dict[Tuple[int, int], Tuple[float, float]] = {}
        for r, y in enumerate(y_coords):
            for c, x in enumerate(x_coords):
                jittered_x = x + self.rng.uniform(-jitter_x, jitter_x)
                jittered_y = y + self.rng.uniform(-jitter_y, jitter_y)
                rot_x = jittered_x * cos_a - jittered_y * sin_a
                rot_y = jittered_x * sin_a + jittered_y * cos_a
                intersections[(r, c)] = (rot_x, rot_y)

        lane_width = LANE_WIDTH_METERS * self.rng.uniform(0.85, 1.05)
        for r in range(rows):
            for c in range(cols - 1):
                start = intersections[(r, c)]
                end = intersections[(r, c + 1)]
                curvature = 0.25 if self.rng.random() < 0.4 else 0.0
                base = _polyline_with_curvature(start, end, self.rng, curvature)
                lanes_per_dir = 1 if self.rng.random() < 0.7 else 2
                for lane_idx in range(lanes_per_dir):
                    offset = lane_width * (lane_idx + 0.5)
                    forward = _offset_polyline(base, offset)
                    backward = list(reversed(_offset_polyline(base, -offset)))
                    self._add_lane(forward, 2, URBAN_SPEED_RANGE)
                    self._add_lane(backward, 2, URBAN_SPEED_RANGE)

                boundary_offset = lane_width * (lanes_per_dir + 0.6)
                self._add_road_edge(_offset_polyline(base, boundary_offset))
                self._add_road_edge(_offset_polyline(base, -boundary_offset))
                self._add_road_line(base, 11)

        for c in range(cols):
            for r in range(rows - 1):
                start = intersections[(r, c)]
                end = intersections[(r + 1, c)]
                curvature = 0.25 if self.rng.random() < 0.4 else 0.0
                base = _polyline_with_curvature(start, end, self.rng, curvature)
                lanes_per_dir = 1 if self.rng.random() < 0.75 else 2
                for lane_idx in range(lanes_per_dir):
                    offset = lane_width * (lane_idx + 0.5)
                    forward = _offset_polyline(base, offset)
                    backward = list(reversed(_offset_polyline(base, -offset)))
                    self._add_lane(forward, 2, URBAN_SPEED_RANGE)
                    self._add_lane(backward, 2, URBAN_SPEED_RANGE)

                boundary_offset = lane_width * (lanes_per_dir + 0.6)
                self._add_road_edge(_offset_polyline(base, boundary_offset))
                self._add_road_edge(_offset_polyline(base, -boundary_offset))
                self._add_road_line(base, 11)

        # Add a couple of diagonal connectors to mimic irregular European streets.
        diag_attempts = self.rng.randint(1, 2)
        all_keys = list(intersections.keys())
        for _ in range(diag_attempts):
            start_idx = self.rng.choice(all_keys)
            end_idx = self.rng.choice(all_keys)
            if start_idx == end_idx:
                continue
            start = intersections[start_idx]
            end = intersections[end_idx]
            if math.dist(start, end) < 40:
                continue
            base = _polyline_with_curvature(start, end, self.rng, 0.15)
            forward = _offset_polyline(base, lane_width * 0.5)
            backward = list(reversed(_offset_polyline(base, -lane_width * 0.5)))
            self._add_lane(forward, 2, URBAN_SPEED_RANGE)
            self._add_lane(backward, 2, URBAN_SPEED_RANGE)
            boundary_offset = lane_width * 1.6
            self._add_road_edge(_offset_polyline(base, boundary_offset))
            self._add_road_edge(_offset_polyline(base, -boundary_offset))

    def _build_highway_network(self) -> None:
        length = self.rng.uniform(650.0, 900.0)
        segments = self.rng.randint(6, 9)
        x_values = [(-length / 2) + (length / (segments - 1)) * i for i in range(segments)]
        amplitude = self.rng.uniform(15.0, 35.0)
        frequency = self.rng.uniform(0.004, 0.009)
        phase = self.rng.uniform(0, 2 * math.pi)
        y_offset = self.rng.uniform(-40.0, 40.0)

        base: List[Tuple[float, float]] = []
        for idx, x in enumerate(x_values):
            y = y_offset + amplitude * math.sin(frequency * x + phase)
            if 0 < idx < segments - 1:
                y += self.rng.uniform(-8.0, 8.0)
            base.append((x, y))

        lane_width = LANE_WIDTH_METERS * self.rng.uniform(0.95, 1.1)
        lanes_per_dir = self.rng.choice([2, 3])

        for lane_idx in range(lanes_per_dir):
            offset = lane_width * (lane_idx + 0.5)
            forward = _offset_polyline(base, offset)
            backward = list(reversed(_offset_polyline(base, -offset)))
            self._add_lane(forward, 1, HIGHWAY_SPEED_RANGE)
            self._add_lane(backward, 1, HIGHWAY_SPEED_RANGE)

        boundary_offset = lane_width * (lanes_per_dir + 0.9)
        self._add_road_edge(_offset_polyline(base, boundary_offset))
        self._add_road_edge(_offset_polyline(base, -boundary_offset))
        self._add_road_line(base, 12)

        # Optional on-ramp / off-ramp connectors
        if self.rng.random() < 0.85:
            entry_sign = self.rng.choice([-1, 1])
            entry_start = base[1]
            mid_x = entry_start[0] - entry_sign * 60
            mid_y = entry_start[1] + entry_sign * 55
            ramp = [
                (entry_start[0] - entry_sign * 140, entry_start[1] + entry_sign * 95),
                (mid_x, mid_y),
                entry_start,
            ]
            self._add_lane(ramp, 1, (14.0, 22.0))

        if self.rng.random() < 0.85:
            exit_sign = self.rng.choice([-1, 1])
            exit_start = base[-2]
            mid_x = exit_start[0] + exit_sign * 70
            mid_y = exit_start[1] + exit_sign * 60
            ramp = [
                exit_start,
                (mid_x, mid_y),
                (exit_start[0] + exit_sign * 150, exit_start[1] + exit_sign * 110),
            ]
            self._add_lane(ramp, 1, (14.0, 24.0))

    # ------------------------------------------------------------------
    # Agent generation
    # ------------------------------------------------------------------
    def _vehicle_size(self) -> Tuple[float, float, float]:
        if self.rng.random() < 0.82:
            length = self.rng.uniform(4.0, 4.9)
            width = self.rng.uniform(1.7, 1.95)
            height = self.rng.uniform(1.4, 1.6)
        else:
            length = self.rng.uniform(5.8, 8.5)
            width = self.rng.uniform(2.3, 2.6)
            height = self.rng.uniform(2.6, 3.5)
        return length, width, height

    def _generate_agents(self) -> List[dict]:
        if not self.lanes:
            raise RuntimeError("No lanes were generated for the random map")

        if self.map_type == "urban":
            num_agents = self.rng.randint(10, 22)
            clearance = URBAN_CLEARANCE
            padding = URBAN_PADDING
            dt = DEFAULT_TIMESTEP_S
        else:
            num_agents = self.rng.randint(18, 34)
            clearance = HIGHWAY_CLEARANCE
            padding = HIGHWAY_PADDING
            dt = HIGHWAY_TIMESTEP_S

        lane_weights = [max(l.total_length, 1.0) for l in self.lanes]
        objects: List[dict] = []

        for agent_id in range(num_agents):
            lane = self.rng.choices(self.lanes, weights=lane_weights, k=1)[0]
            distance = lane.try_reserve_distance(self.rng, clearance, padding)
            attempts = 0
            while distance is None and attempts < 6:
                lane = self.rng.choices(self.lanes, weights=lane_weights, k=1)[0]
                distance = lane.try_reserve_distance(self.rng, clearance, padding)
                attempts += 1
            if distance is None:
                continue

            position, heading = lane.point_and_heading_at(distance)
            speed_min, speed_max = lane.speed_range
            speed = self.rng.uniform(speed_min, speed_max)
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)

            length, width, height = self._vehicle_size()

            positions: List[Tuple[float, float]] = []
            headings: List[float] = []
            velocities: List[Tuple[float, float]] = []
            valids: List[bool] = []

            steps = max(6, min(18, int((lane.total_length - distance) / max(speed * dt, 1.0)) + 3))
            current_distance = distance
            for _ in range(steps):
                pos, hdg = lane.point_and_heading_at(current_distance)
                positions.append(pos)
                headings.append(hdg)
                velocities.append((vx, vy))
                valids.append(True)
                current_distance = min(current_distance + speed * dt, lane.total_length - 1e-3)
                if current_distance >= lane.total_length - 1.0:
                    break

            goal = positions[-1]

            objects.append(
                {
                    "position": [_round_point(p) for p in positions],
                    "width": round(width, 2),
                    "length": round(length, 2),
                    "height": round(height, 2),
                    "heading": [round(h, 4) for h in headings],
                    "velocity": [
                        {"x": round(vx, 3), "y": round(vy, 3)} for _ in positions
                    ],
                    "valid": valids,
                    "goalPosition": _round_point(goal),
                    "type": "vehicle",
                    "id": agent_id,
                    "mark_as_expert": False,
                }
            )

        return objects

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self) -> dict:
        if self.map_type == "urban":
            self._build_urban_network()
        elif self.map_type == "highway":
            self._build_highway_network()
        else:
            raise ValueError(f"Unsupported map type '{self.map_type}'")

        objects = self._generate_agents()
        if not objects:
            raise RuntimeError("Random map generator produced no agents")

        suffix = f"{self.seed % 1_000_000:06d}"
        scenario_id = f"random_{self.map_type}_{suffix}"
        return {
            "name": scenario_id,
            "scenario_id": scenario_id,
            "objects": objects,
            "roads": self.roads,
            "tl_states": [],
            "metadata": {
                "sdc_track_index": -1,
                "objects_of_interest": [],
                "tracks_to_predict": [],
            },
        }


def generate_random_map(map_type: str = "urban", seed: Optional[int] = None) -> dict:
    """Generate a single random map dictionary."""

    rng = random.Random(seed)
    builder = RandomMapBuilder(rng=rng, map_type=map_type, seed=seed)
    return builder.generate()


def save_random_map(map_data: dict, path: Path | str) -> Path:
    """Persist a generated map to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(map_data, f)
    return path


class RandomMapDataLoader:
    """Data loader that yields batches of procedurally generated maps."""

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        map_types: Sequence[str] = ("urban", "highway"),
        *,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        shuffle: bool = True,
        loop_forever: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not map_types:
            raise ValueError("map_types must contain at least one entry")

        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.map_types = tuple(mt.lower() for mt in map_types)
        self.random_gen = random.Random(seed)
        self.shuffle = shuffle
        self.loop_forever = loop_forever
        self.output_dir = Path(output_dir) if output_dir else Path("data/generated/random_maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = f"random_{seed}"

        self.dataset: List[str] = []
        self.indices: List[int] = []
        self.current_index = 0
        self._refresh_dataset()

    def _refresh_dataset(self) -> None:
        total = max(self.dataset_size, self.batch_size)
        self.dataset = []
        for idx in range(total):
            map_type = self.random_gen.choice(self.map_types)
            map_seed = self.random_gen.randint(0, 2**31 - 1)
            map_data = generate_random_map(map_type=map_type, seed=map_seed)
            path = self.output_dir / f"{self.prefix}_{idx:05d}.json"
            save_random_map(map_data, path)
            self.dataset.append(str(path))
        self._reset_indices()

    def _reset_indices(self) -> None:
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.random_gen.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self) -> "RandomMapDataLoader":
        self._reset_indices()
        return self

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __next__(self) -> List[str]:
        if self.current_index + self.batch_size > len(self.indices):
            if not self.loop_forever:
                raise StopIteration
            self._refresh_dataset()

        start = self.current_index
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        if len(batch_indices) < self.batch_size:
            if not self.loop_forever:
                raise StopIteration
            self._refresh_dataset()
            return self.__next__()

        self.current_index = end
        return [self.dataset[i] for i in batch_indices]


__all__ = [
    "generate_random_map",
    "save_random_map",
    "RandomMapDataLoader",
]

import json
from pathlib import Path

from gpudrive.env.random_map import RandomMapDataLoader, generate_random_map


def test_generate_random_map_structure():
    urban_map = generate_random_map("urban", seed=1234)
    assert "objects" in urban_map and urban_map["objects"], "Urban map should contain agents"
    assert "roads" in urban_map and urban_map["roads"], "Urban map should contain roads"

    highway_map = generate_random_map("highway", seed=5678)
    assert "objects" in highway_map and highway_map["objects"], "Highway map should contain agents"
    assert "roads" in highway_map and highway_map["roads"], "Highway map should contain roads"

    for map_data in (urban_map, highway_map):
        assert map_data["metadata"]["sdc_track_index"] == -1
        assert map_data["metadata"]["tracks_to_predict"] == []
        assert map_data["metadata"]["objects_of_interest"] == []


def test_random_map_data_loader(tmp_path):
    loader = RandomMapDataLoader(
        batch_size=2,
        dataset_size=3,
        map_types=("urban", "highway"),
        seed=99,
        output_dir=tmp_path,
        shuffle=False,
        loop_forever=False,
    )

    iterator = iter(loader)
    batch = next(iterator)
    assert len(batch) == 2

    for file_path in batch:
        data_path = Path(file_path)
        assert data_path.exists()
        scene = json.loads(data_path.read_text())
        assert "objects" in scene and scene["objects"]
        assert "roads" in scene and scene["roads"]
        assert scene["name"] == scene["scenario_id"]

"""Check the online adapter against the real NAVSIM simulator and evaluator."""

import csv
from dataclasses import dataclass
import lzma
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.maneuverflow.online_candidate_supervision import (
    OfficialPDMFactorScorer,
    interpolate_relative_hypotheses_to_state_array,
)
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def _initial_state(*, heading=0.0):
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, heading),
        rear_axle_velocity_2d=StateVector2D(2.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(1_000_000),
        vehicle_parameters=get_pacifica_parameters(),
    )


def test_online_interpolation_matches_official_pose_interpolation_with_yaw_wrap():
    initial = _initial_state(heading=2.8)
    hypotheses = np.zeros((2, 8, 3), dtype=np.float32)
    hypotheses[:, :, 0] = np.arange(1, 9)[None, :]
    hypotheses[0, :, 2] = np.linspace(2.9, 3.4, 8)
    hypotheses[1, :, 2] = np.linspace(-3.4, -2.9, 8)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    official = [
        get_trajectory_as_array(transform_trajectory(Trajectory(row), initial), sampling, initial.time_point)
        for row in hypotheses
    ]
    vectorized = interpolate_relative_hypotheses_to_state_array(
        hypotheses, initial_ego_state=initial, expert_states=official[0], proposal_sampling=sampling,
    )
    assert vectorized.shape == (3, 41, 11)
    np.testing.assert_allclose(vectorized[0], official[0], atol=1e-6)
    for index, reference in enumerate(official, start=1):
        # Official interpolation quantizes floating timestamps to integer
        # microseconds; vectorized interpolation retains exact 0.1s offsets.
        np.testing.assert_allclose(vectorized[index, :, :3], reference[:, :3], atol=1e-5)
        np.testing.assert_allclose(vectorized[index, 0], reference[0], atol=1e-6)


def test_official_adapter_scores_64_and_matches_two_proposal_navsim_evaluation():
    initial = _initial_state()
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    observation = PDMObservation(sampling, sampling, map_radius=1000, observation_sample_res=1)
    observation.update_detections_tracks([DetectionsTracks(TrackedObjects([])) for _ in range(51)])
    drivable_map = PDMDrivableMap(
        tokens=["road", "lane"],
        map_types=[SemanticMapLayer.ROADBLOCK, SemanticMapLayer.LANE],
        geometries=np.array([box(-100, -10, 100, 10), box(-100, -10, 100, 10)], dtype=object),
    )
    expert = np.zeros((8, 3), dtype=np.float32)
    expert[:, 0] = np.arange(1, 9)
    cache = SimpleNamespace(
        ego_state=initial,
        trajectory=transform_trajectory(Trajectory(expert), initial),
        observation=observation,
        centerline=PDMPath([StateSE2(-100, 0, 0), StateSE2(100, 0, 0)]),
        route_lane_ids=["lane"],
        drivable_area_map=drivable_map,
    )
    loader = SimpleNamespace(get_from_token=lambda token: cache)
    adapter = OfficialPDMFactorScorer(loader)
    hypotheses = np.zeros((64, 8, 3), dtype=np.float32)
    hypotheses[:, :, 0] = np.linspace(0.5, 4.0, 64)[:, None] * np.arange(1, 9)[None, :]
    factors = adapter.score("synthetic_scene", hypotheses)
    assert factors.shape == (5, 64)
    assert np.isfinite(factors).all()
    for index in (0, 31, 63):
        reference = pdm_score(
            cache, Trajectory(hypotheses[index]), sampling, adapter._simulator, adapter._scorer,
        )
        expected = np.array([
            reference.no_at_fault_collisions, reference.drivable_area_compliance,
            reference.ego_progress, reference.time_to_collision_within_bound, reference.comfort,
        ])
        np.testing.assert_allclose(factors[:, index], expected, atol=1e-6)
        nc, dac, ep, ttc, comfort = factors[:, index]
        np.testing.assert_allclose(nc * dac * (5 * ep + 5 * ttc + 2 * comfort) / 12,
                                   reference.score, atol=1e-6)


def test_metric_cache_reads_all_worker_metadata_and_quoted_paths(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for index, token in enumerate(("first", "second")):
        cache_path = tmp_path / "log,with-comma" / token / "metric_cache.pkl"
        cache_path.parent.mkdir(parents=True)
        with lzma.open(cache_path, "wb") as stream:
            pickle.dump(token, stream)
        with (metadata / f"cache_metadata_node_{index}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file_name"])
            writer.writerow([cache_path.as_posix()])
    loader = MetricCacheLoader(tmp_path)
    assert loader.tokens == ["first", "second"]
    assert [loader.get_from_token(token) for token in loader.tokens] == ["first", "second"]


def test_metric_cache_missing_metadata_reports_the_cache_path(tmp_path):
    (tmp_path / "metadata").mkdir()
    with pytest.raises(FileNotFoundError, match="metadata"):
        MetricCacheLoader(tmp_path)


@pytest.mark.parametrize("requires_scene", [False, True])
def test_evaluation_predicts_from_agent_input_before_reading_reference_cache(tmp_path, monkeypatch, requires_scene):
    from navsim.planning.script import run_pdm_score as evaluation

    @dataclass
    class Result:
        score: float = 0.75

    events = []
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    cache_path = tmp_path / "log/scene/metric_cache.pkl"
    cache_path.parent.mkdir(parents=True)
    with lzma.open(cache_path, "wb") as stream:
        pickle.dump("reference_only", stream)
    with (metadata / "cache_metadata.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name"])
        writer.writerow([cache_path.as_posix()])

    current_input = object()
    trajectory = Trajectory(np.zeros((8, 3), dtype=np.float32))

    def get_input(token):
        assert token == "scene"
        events.append("input")
        return current_input

    def predict(agent_input):
        assert agent_input is current_input
        assert events == ["initialize", "input"]
        events.append("predict")
        return trajectory

    original_load = pickle.load

    def load_reference(stream):
        assert events == ["initialize", "input", "predict"]
        events.append("reference")
        return original_load(stream)

    def score(**kwargs):
        assert kwargs["model_trajectory"] is trajectory
        assert kwargs["metric_cache"] == "reference_only"
        events.append("score")
        return Result()

    agent = SimpleNamespace(
        requires_scene=requires_scene, initialize=lambda: events.append("initialize"),
        get_sensor_config=lambda: object(), compute_trajectory=predict,
    )
    sampling = object()
    cfg = SimpleNamespace(
        agent=agent, simulator=SimpleNamespace(proposal_sampling=sampling),
        scorer=SimpleNamespace(proposal_sampling=sampling), metric_cache_path=str(tmp_path),
        train_test_split=SimpleNamespace(scene_filter=SimpleNamespace()),
        sensor_blobs_path=str(tmp_path), navsim_log_path=str(tmp_path),
    )
    loader = SimpleNamespace(tokens=["scene"], get_agent_input_from_token=get_input)
    loader.get_scene_from_token = lambda token: pytest.fail("Annotated scenes must never reach inference.")
    monkeypatch.setattr(evaluation, "instantiate", lambda config: config)
    monkeypatch.setattr(evaluation, "SceneLoader", lambda **kwargs: loader)
    monkeypatch.setattr(pickle, "load", load_reference)
    monkeypatch.setattr(evaluation, "pdm_score", score)

    if requires_scene:
        with pytest.raises(ValueError, match="AgentInput"):
            evaluation.run_pdm_score([{"cfg": cfg, "log_file": "log", "tokens": ["scene"]}])
        assert events == []
    else:
        rows = evaluation.run_pdm_score([{"cfg": cfg, "log_file": "log", "tokens": ["scene"]}])
        assert rows == [{"token": "scene", "valid": True, "score": 0.75}]
        assert events == ["initialize", "input", "predict", "reference", "score"]


@pytest.mark.parametrize("all_fail", [False, True])
def test_submission_scoring_returns_failed_rows_for_csv_export(tmp_path, monkeypatch, all_fail):
    from navsim.planning.script import run_pdm_score_from_submission as scoring

    @dataclass
    class Result:
        score: float = 0.75

    submission = tmp_path / "submission.pkl"
    with submission.open("wb") as stream:
        pickle.dump({"predictions": [{} if all_fail else {"good": "trajectory"}]}, stream)
    monkeypatch.setattr(scoring, "MetricCacheLoader", lambda path: SimpleNamespace(
        tokens=["good", "missing"], get_from_token=lambda token: object(),
    ))
    monkeypatch.setattr(scoring, "pdm_score", lambda **kwargs: Result())
    scores = scoring.run_pdm_score(
        submission, SimpleNamespace(proposal_sampling=None), object(), tmp_path,
    )
    assert scores is not None
    assert scores["token"].tolist() == ["good", "missing"]
    assert scores["valid"].tolist() == [not all_fail, False]
    if all_fail:
        assert np.isnan(scores.loc[0, "score"])
    else:
        assert scores.loc[0, "score"] == 0.75
    assert np.isnan(scores.loc[1, "score"])
    output = tmp_path / "scores.csv"
    scores.to_csv(output, index=False)
    assert output.is_file()

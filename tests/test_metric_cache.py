"""Check reference-cache selection independently of final PDM evaluation."""

import numpy as np
from shapely.geometry import box

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex, WeightedMetricIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def _score_open_road(scorer, speeds, *, oncoming=False):
    sampling = scorer.proposal_sampling
    times = np.arange(sampling.num_poses + 1) * sampling.interval_length
    states = np.zeros((len(speeds), len(times), StateIndex.size()), dtype=np.float64)
    states[:, :, StateIndex.X] = np.asarray(speeds)[:, None] * times
    states[:, :, StateIndex.VELOCITY_X] = np.asarray(speeds)[:, None]
    if oncoming:
        states[:, :, StateIndex.Y] = 8.0
    observation = PDMObservation(sampling, sampling, map_radius=1000, observation_sample_res=1)
    observation.update_detections_tracks([DetectionsTracks(TrackedObjects([])) for _ in range(51)])
    drivable_map = PDMDrivableMap(
        tokens=["road", "lane"],
        map_types=[SemanticMapLayer.ROADBLOCK, SemanticMapLayer.LANE],
        geometries=np.array([box(-1000, -20, 1000, 20), box(-1000, -3, 1000, 3)], dtype=object),
    )
    return scorer.score_proposals(
        states, observation, PDMPath([StateSE2(-1000, 0, 0), StateSE2(1000, 0, 0)]),
        ["lane"], drivable_map,
    )


def test_reference_cache_uses_its_own_progress_threshold(tmp_path):
    processor = MetricCacheProcessor(str(tmp_path), force_feature_computation=True)
    reference = processor._pdm_closed._scorer
    evaluation = PDMScorer(reference.proposal_sampling)
    assert reference._config.progress_distance_threshold == 0.1
    assert evaluation._config.progress_distance_threshold == 5.0


def test_reference_cache_normalizes_short_progress(tmp_path):
    reference = MetricCacheProcessor(str(tmp_path), force_feature_computation=True)._pdm_closed._scorer
    _score_open_road(reference, [0.05, 0.1])
    np.testing.assert_allclose(reference._weighted_metrics[WeightedMetricIndex.PROGRESS], [0.5, 1.0])


def test_reference_direction_gate_does_not_change_final_evaluation(tmp_path):
    reference = MetricCacheProcessor(str(tmp_path), force_feature_computation=True)._pdm_closed._scorer
    evaluation = PDMScorer(reference.proposal_sampling)
    speeds = np.linspace(10.0, 20.0, 15)
    reference_scores = _score_open_road(reference, speeds, oncoming=True)
    evaluation_scores = _score_open_road(evaluation, speeds, oncoming=True)
    assert reference_scores[-1] == 0.0
    assert evaluation_scores[-1] > 0.9

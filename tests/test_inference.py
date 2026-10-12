"""BEV geometry, inference conditioning, and published TF inference."""

import math
from pathlib import Path

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.maneuverflow.architecture import ManeuverFlowHypothesizer
from navsim.agents.maneuverflow.checkpoint_schema import checkpoint_file_sha256
from navsim.agents.maneuverflow.config import ManeuverFlowConfig
from navsim.agents.maneuverflow.model import (
    AgentPredictionHead,
    ManeuverFlowModel,
    StatusModulatedManeuverDecoder,
    _warp_future_bev_target_to_current_frame,
)
from navsim.agents.maneuverflow.trajectory_evidence_probe import (
    _sample_bev_along_trajectory,
)
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "checkpoints/maneuverflow_tf_0.906.pt"


@pytest.fixture
def cpu_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 8))
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(previous_threads)


def _physical_bev_coordinates(config, height=64, width=32):
    # The histogram extent consists of cells; each feature sits at its center.
    x_edges = torch.linspace(config.bev_minimum_x, config.bev_maximum_x, height + 1)
    y_edges = torch.linspace(config.bev_minimum_y, config.bev_maximum_y, width + 1)
    return torch.meshgrid(
        0.5 * (x_edges[1:] + x_edges[:-1]),
        0.5 * (y_edges[1:] + y_edges[:-1]),
        indexing="ij",
    )


@pytest.mark.parametrize("decay", [0.0, 0.5, 0.998, 0.9999])
def test_joint_training_requires_released_ema_decay(decay):
    with pytest.raises(ValueError, match="ema_target_decay=0.999"):
        ManeuverFlowConfig(training_phase="joint_finetuning", ema_target_decay=decay)


@pytest.mark.parametrize("phase", ["generation_pretraining", "joint_finetuning", "inference"])
def test_bev_probe_reads_forward_x_and_histogram_pixel_centers(phase):
    config = ManeuverFlowConfig(training_phase=phase, ecfr_final_pool_composition="source_replacement")
    x, y = _physical_bev_coordinates(config)
    bev = torch.stack([x, y]).unsqueeze(0)
    points = torch.tensor([[8.0, 4.0], [-6.0, 10.0], [-31.5, -31.0], [31.5, 31.0]])
    coordinates = points.view(1, -1, 1, 1, 2)
    sampled, valid = _sample_bev_along_trajectory(bev, coordinates, config, return_valid=True)
    torch.testing.assert_close(sampled[0, :, 0, 0], points, rtol=0.0, atol=1e-5)
    assert bool(valid.all())


def test_legacy_bev_probe_preserves_descending_x_and_endpoint_alignment():
    config = ManeuverFlowConfig(training_phase="inference", ecfr_final_pool_composition="independent_ranked_retention")
    x, y = torch.meshgrid(torch.linspace(-32.0, 32.0, 9), torch.linspace(-32.0, 32.0, 7), indexing="ij")
    bev = torch.stack([x, y]).unsqueeze(0)
    points = torch.tensor([[8.0, 4.0], [-32.0, -32.0], [32.0, 32.0]])
    sampled = _sample_bev_along_trajectory(bev, points.view(1, -1, 1, 1, 2), config)
    expected = points.clone()
    expected[:, 0] *= -1.0
    torch.testing.assert_close(sampled[0, :, 0, 0], expected, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("legacy", [False, True])
def test_future_bev_warp_identity_preserves_all_pixels(legacy):
    config = ManeuverFlowConfig(
        training_phase="inference",
        ecfr_final_pool_composition="independent_ranked_retention" if legacy else "source_replacement",
    )
    generator = torch.Generator().manual_seed(7)
    future_bev = torch.randn(2, 3, 2, 16, 32, generator=generator)
    warped, valid = _warp_future_bev_target_to_current_frame(future_bev, torch.zeros(2, 3, 3), config)
    torch.testing.assert_close(warped, future_bev, rtol=0.0, atol=1e-5)
    assert bool(valid.all())


@pytest.mark.parametrize("offset", [(8.0, -4.0, 0.0), (4.0, -2.0, math.pi / 2.0), (-3.0, 5.0, 0.35)])
def test_future_bev_warp_matches_inverse_se2_on_affine_field(offset):
    config = ManeuverFlowConfig(training_phase="joint_finetuning")
    x, y = _physical_bev_coordinates(config)
    future_bev = (x + 2.0 * y).view(1, 1, 1, *x.shape)
    warped, valid = _warp_future_bev_target_to_current_frame(future_bev, torch.tensor([[offset]]), config)
    dx, dy, heading = offset
    future_x = math.cos(heading) * (x - dx) + math.sin(heading) * (y - dy)
    future_y = -math.sin(heading) * (x - dx) + math.cos(heading) * (y - dy)
    expected = future_x + 2.0 * future_y
    interior = (future_x.abs() <= 31.5) & (future_y.abs() <= 31.0)
    torch.testing.assert_close(warped[0, 0, 0][interior], expected[interior], rtol=1e-5, atol=2e-5)
    physical_valid = (future_x.abs() <= 32.0) & (future_y.abs() <= 32.0)
    assert torch.equal(valid[0, 0, 0], physical_valid)
    assert not bool(warped[0, 0, 0][~physical_valid].any())


@pytest.mark.parametrize("legacy", [False, True])
def test_translated_future_bev_has_expected_forward_origin_value(legacy):
    config = ManeuverFlowConfig(
        training_phase="inference",
        ecfr_final_pool_composition="independent_ranked_retention" if legacy else "source_replacement",
    )
    if legacy:
        x, _ = torch.meshgrid(torch.linspace(-32.0, 32.0, 65), torch.linspace(-32.0, 32.0, 33), indexing="ij")
    else:
        x, _ = _physical_bev_coordinates(config)
    warped, _ = _warp_future_bev_target_to_current_frame(
        x.view(1, 1, 1, *x.shape), torch.tensor([[[8.0, 0.0, 0.0]]]), config,
    )
    origin_sample = _sample_bev_along_trajectory(warped, torch.zeros(1, 1, 1, 1, 2), config)
    torch.testing.assert_close(origin_sample.squeeze(), torch.tensor(8.0 if legacy else -8.0))


class _CoordinateBackbone(torch.nn.Module):
    def forward(self, camera, lidar):
        high = torch.arange(64.0).view(1, 1, 64, 1).expand(camera.shape[0], 1, 64, 64)
        low = torch.zeros(camera.shape[0], 4, 8, 8)
        return high, low, None


class _QueryPassthrough(torch.nn.Module):
    def forward(self, query, keyval):
        return query


def _small_perception_model(config):
    # Exercise the real perception control flow without building the full encoders.
    model = ManeuverFlowModel.__new__(ManeuverFlowModel)
    torch.nn.Module.__init__(model)
    model._config = config
    model._query_splits = [1, 2]
    model.hypothesizer = ManeuverFlowHypothesizer()
    perception = model.hypothesizer.perception
    perception.transfuser_backbone = _CoordinateBackbone()
    perception.bev_projection = torch.nn.Identity()
    perception.state_navigation_encoder = torch.nn.Linear(8, 4)
    perception.bev_memory_embedding = torch.nn.Embedding(65, 4)
    perception.query_embedding = torch.nn.Embedding(3, 4)
    perception.query_decoder = _QueryPassthrough()
    perception.bev_semantic_head = torch.nn.Identity()
    perception.agent_prediction_head = AgentPredictionHead(2, 8, 4)
    model.hypothesizer.maneuver_decoder = StatusModulatedManeuverDecoder(4, num_classes=8, dropout=0.0)
    features = {
        "camera_feature": torch.zeros(1, 3, 1, 1),
        "lidar_feature": torch.zeros(1, 1, 1, 1),
        "status_feature": torch.tensor([[0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]]),
    }
    return model, features


@pytest.mark.parametrize("phase,composition,first_row", [
    ("generation_pretraining", "independent_ranked_retention", 32),
    ("joint_finetuning", "source_replacement", 32),
    ("inference", "source_replacement", 32),
    ("inference", "independent_ranked_retention", 0),
])
def test_semantic_head_crops_the_release_profile_half(phase, composition, first_row, cpu_runtime):
    model, features = _small_perception_model(ManeuverFlowConfig(training_phase=phase, ecfr_final_pool_composition=composition))
    output, _ = model._forward_perception(features)
    expected = torch.arange(float(first_row), float(first_row + 32))
    torch.testing.assert_close(output["bev_semantic_map"][0, 0, :, 0], expected)


@pytest.mark.parametrize("phase,train_mode,masked", [
    ("inference", True, False),
    ("generation_pretraining", True, True),
    ("joint_finetuning", True, True),
    ("generation_pretraining", False, False),
])
def test_navigation_dropout_is_limited_to_training_phases(phase, train_mode, masked, monkeypatch, cpu_runtime):
    model, features = _small_perception_model(ManeuverFlowConfig(training_phase=phase))
    torch.nn.Module.train(model, train_mode)
    commands = []
    handle = model.hypothesizer.maneuver_decoder.dynamic_query_proj.register_forward_pre_hook(
        lambda module, inputs: commands.append(inputs[0].detach().clone())
    )
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
    try:
        model._forward_perception(features)
    finally:
        handle.remove()
    expected = torch.zeros(1, 4) if masked else features["status_feature"][:, :4]
    torch.testing.assert_close(commands[0], expected)


def test_released_weights_produce_finite_pool_and_selected_trajectory(cpu_runtime):
    if not WEIGHTS.is_file() or WEIGHTS.stat().st_size < 1024:
        pytest.skip("Run git lfs pull to fetch the published TF checkpoint.")
    assert WEIGHTS.stat().st_size == 643879555
    assert checkpoint_file_sha256(WEIGHTS) == (
        "0d780bdf95253e08ad2a0048bcdfe984108f048a60961f9c1cc21dca1844becc"
    )
    config = OmegaConf.load(
        ROOT / "navsim/planning/script/config/common/agent/maneuverflow_agent.yaml"
    )
    config.checkpoint_path = str(WEIGHTS)
    config.config.inference_device = "cpu"
    agent = instantiate(config)
    agent.initialize()
    assert agent.model.ema_target_encoder is None

    rng = np.random.default_rng(20261009)
    images = [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    empty = Camera()
    cameras = Cameras(
        cam_f0=Camera(image=images[0]), cam_l0=Camera(image=images[1]),
        cam_r0=Camera(image=images[2]), cam_l1=empty, cam_l2=empty,
        cam_r1=empty, cam_r2=empty, cam_b0=empty,
    )
    cloud = np.zeros((6, 4096), dtype=np.float32)
    cloud[:2] = rng.uniform(-31, 31, (2, 4096))
    cloud[2] = rng.uniform(0.3, 2.0, 4096)
    status = EgoStatus(
        ego_pose=np.zeros(3, dtype=np.float64),
        ego_velocity=np.array([5.0, 0.0], dtype=np.float32),
        ego_acceleration=np.zeros(2, dtype=np.float32),
        driving_command=np.array([0, 1, 0, 0], dtype=np.int64),
    )
    inputs = AgentInput([status] * 4, [cameras] * 4, [Lidar(cloud)] * 4)
    features = agent.get_feature_builders()[0].compute_features(inputs)
    torch.random.default_generator.manual_seed(20261009)
    with torch.no_grad():
        output = agent.model.inference({key: value.unsqueeze(0) for key, value in features.items()})
    for key, shape in {
        "generated_base_hypotheses": (1, 64, 8, 3),
        "retained_base_hypotheses": (1, 48, 8, 3),
        "refined_hypotheses": (1, 16, 8, 3),
        "final_hypotheses": (1, 64, 8, 3),
        "final_hypothesis_scores": (1, 64),
    }.items():
        assert tuple(output[key].shape) == shape
        assert bool(torch.isfinite(output[key]).all())
    assert int(output["final_hypothesis_is_refined"].sum()) == 16
    expected = output["final_hypotheses"][0, output["final_hypothesis_scores"][0].argmax()].numpy()
    torch.random.default_generator.manual_seed(20261009)
    selected = agent.compute_trajectory(inputs).poses
    assert selected.shape == (8, 3)
    assert np.isfinite(selected).all()
    np.testing.assert_allclose(selected, expected, rtol=1e-5, atol=1e-5)

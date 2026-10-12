# ManeuverFlow-TF method and code map

ManeuverFlow follows a Hypothesize–Verify–Improve pipeline. Current scene
observations support trajectory generation, future-scene prediction,
candidate verification, and refinement.

## Architecture

1. **Encode the scene.** TransFuser combines camera and LiDAR features with
   navigation and ego-state context.
2. **Hypothesize.** The Maneuver Gaussian Mixture Prior (M-GMP) supplies
   maneuver-conditioned initial states. Maneuver-routed MeanFlow (M³F)
   generates trajectory hypotheses in one step.
3. **Verify.** Trajectory Evidence Memory (TEM) combines current and predicted
   future evidence. Metric Prism predicts NAVSIM planning factors for each candidate.
4. **Improve and select.** Evidence-conditioned Frenet refinement (ECFR)
   improves promising candidates. Joint scoring selects the final trajectory.

During joint training, an EMA target encoder provides future-BEV supervision
from raw future observations. NAVSIM supplies candidate scores for verification
and refinement. Inference uses current observations and predicted future evidence.

## Architecture-to-code map

Paths below are under `navsim/agents/maneuverflow`.

| Component | Implementation |
| --- | --- |
| Inputs and perception | [features.py](../navsim/agents/maneuverflow/features.py), [transfuser_backbone.py](../navsim/agents/maneuverflow/transfuser_backbone.py) |
| M-GMP and trajectory representation | [mgmp_prior.py](../navsim/agents/maneuverflow/mgmp_prior.py), [trajectory_codec.py](../navsim/agents/maneuverflow/trajectory_codec.py) |
| M³F and model orchestration | [m3f.py](../navsim/agents/maneuverflow/m3f.py), [model.py](../navsim/agents/maneuverflow/model.py) |
| Future-BEV and target encoder | [future_bev_model.py](../navsim/agents/maneuverflow/future_bev_model.py), [targets.py](../navsim/agents/maneuverflow/targets.py), [ema_target_encoder.py](../navsim/agents/maneuverflow/ema_target_encoder.py) |
| TEM and Metric Prism | [trajectory_evidence_probe.py](../navsim/agents/maneuverflow/trajectory_evidence_probe.py), [metric_prism.py](../navsim/agents/maneuverflow/metric_prism.py) |
| ECFR and online supervision | [ecfr.py](../navsim/agents/maneuverflow/ecfr.py), [online_candidate_supervision.py](../navsim/agents/maneuverflow/online_candidate_supervision.py), [refinement_supervision.py](../navsim/agents/maneuverflow/refinement_supervision.py) |
| Objectives | [losses.py](../navsim/agents/maneuverflow/losses.py) |
| Agent and checkpoint export | [agent.py](../navsim/agents/maneuverflow/agent.py), [checkpoint_export.py](../navsim/agents/maneuverflow/checkpoint_export.py) |

See the [training and inference guide](train_eval.md) for runnable commands.

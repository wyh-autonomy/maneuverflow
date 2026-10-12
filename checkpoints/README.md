# Released TF checkpoint

| Checkpoint | Agent configuration |
| --- | --- |
| [maneuverflow_tf_0.906.pt](maneuverflow_tf_0.906.pt) | `maneuverflow_agent` |

From the repository root, download the Git LFS payload and verify it:

```bash
git lfs pull --include="checkpoints/maneuverflow_tf_0.906.pt"
sha256sum checkpoints/maneuverflow_tf_0.906.pt
```

Expected SHA-256:
`0d780bdf95253e08ad2a0048bcdfe984108f048a60961f9c1cc21dca1844becc`.

The checkpoint includes its M-GMP prior. Follow the
[inference tutorial](../docs/train_eval.md#evaluation-and-submission) to evaluate it
or save predictions. For Stage-2 exports, use `maneuverflow_ema_agent` as shown in
[export and inference](../docs/train_eval.md#export-and-evaluate-trained-weights).

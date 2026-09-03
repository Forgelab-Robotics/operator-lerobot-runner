# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting for this repository. Include affected versions,
reproduction steps, impact, and any suggested mitigation.

## Deployment boundary

LeRobot Inference runs policy inference as a CLI or Dora node and does not
provide authentication or transport encryption for its dataflow interfaces.

- Run inference nodes only on trusted networks; do not expose Dora dataflows
  directly to the public internet.
- Treat configuration files, recordings, runtime state, and Tool results as
  potentially sensitive.
- Checkpoint loading uses `torch.load(weights_only=True)` for legacy `.ckpt`
  files; still verify checkpoint provenance before deployment.
- Validate safety controls independently before connecting a physical robot.

Security fixes are supported on the latest released version.

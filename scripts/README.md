# Scripts

This folder contains runnable reference scripts for the local `rsl_rl` tree.

## AMP Example

`amp_plugin_case.py` shows the recommended plugin-driven AMP path:

- `PPO + AMPPlugin + ExternalAMPProvider`
- transition expert data under `{"amp": ..., "next_amp": ...}`
- runner-side automatic expert-source injection from the environment

Run from the repository root:

```bash
python rsl_rl/scripts/amp_plugin_case.py
```

To inspect the equivalent explicit plugin block instead of `algorithm.amp_cfg` translation:

```bash
python rsl_rl/scripts/amp_plugin_case.py --explicit-plugin
```

What the script demonstrates:

- how to expose `"amp"` observations separately from policy observations
- how to provide both `get_amp_expert_observations()` and `sample_amp_expert_observations()`
- how to configure AMP with transition-first discriminator inputs in the new framework

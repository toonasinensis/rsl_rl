# RSL-RL Plugin Branch

This branch reorganizes PPO extensions around a small plugin interface. The goal is to keep the core PPO loop clean while making research features easy to add, test, turn on, or remove from configuration.

[中文说明](README.zh-CN.md)

## Why Plugins

- **Lower coupling:** extra rewards, losses, metrics, parameters, and checkpoint states live in plugins instead of being hard-coded into PPO.
- **Configurable experiments:** plugins are created from `algorithm.plugins`, so different training recipes can share the same PPO implementation.
- **Clear extension points:** hooks cover rollout, update, gradient clipping, train/eval mode, save, and load.
- **Easier growth:** new methods can be added as new `PPOPlugin` subclasses without rewriting the runner or duplicating PPO.

## Included Plugins

### AMPPlugin

`AMPPlugin` integrates Adversarial Motion Priors into PPO. During rollout it stores AMP observations, computes discriminator-based rewards, and logs reward components. During update it trains the discriminator together with the policy through the plugin loss hooks.

### TeacherKLPlugin

`TeacherKLPlugin` adds a frozen teacher-policy KL loss for student policy training. It loads a teacher checkpoint, maps teacher observation groups when needed, applies a scheduled KL coefficient, and saves plugin progress in checkpoints.

## Minimal Config Shape

```python
algorithm = {
    "class_name": "PPO",
    "plugins": [
        {
            "class_name": "AMPPlugin",
            "amp_reward_coef": 2.0,
            "amp_discr_hidden_dims": [1024, 512],
            "amp_motion_files": "/path/to/motions",
            "amp_body_names": ["pelvis", "left_foot", "right_foot"],
            "amp_anchor_name": "pelvis",
        },
        {
            "class_name": "TeacherKLPlugin",
            "teacher_checkpoint_path": "/path/to/teacher.pt",
            "teacher_actor": {...},
            "teacher_obs_groups": {"actor": ["proprio"]},
        },
    ],
}
```

Plugins are optional. Leave `algorithm.plugins` empty to run plain PPO.

## Installation

```bash
git clone https://github.com/toonasinensis/rsl_rl.git
cd rsl_rl
pip install -e .
```

## Tests

```bash
pytest tests/algorithms/test_amp_motion_loader.py \
       tests/algorithms/test_amp_plugin_step_metrics.py \
       tests/algorithms/test_teacher_kl_plugin.py
```

## Citation

This repository is based on RSL-RL. If you use it in research, please cite the original project.

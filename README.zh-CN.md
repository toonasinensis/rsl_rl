# RSL-RL 插件分支

这个分支把 PPO 的扩展能力整理成了轻量级插件接口。核心目标是保持 PPO 主循环简洁，同时让研究功能可以通过配置添加、测试、关闭或替换。

[English README](README.md)

## 插件式编程的优点

- **低耦合：** 额外奖励、额外 loss、日志指标、插件参数和 checkpoint 状态都放在插件里，不再硬塞进 PPO。
- **配置化实验：** 插件通过 `algorithm.plugins` 创建，同一个 PPO 可以复用在不同训练方案中。
- **扩展点清晰：** hook 覆盖 rollout、update、梯度裁剪、训练/推理模式、保存和加载。
- **方便继续扩展：** 新方法只需要继承 `PPOPlugin`，不用复制或重写 PPO 和 runner。

## 当前两个插件

### AMPPlugin

`AMPPlugin` 将 Adversarial Motion Priors 接入 PPO。rollout 阶段保存 AMP 观测、计算判别器奖励并记录奖励分量；update 阶段通过插件 loss hook 与策略一起训练判别器。

### TeacherKLPlugin

`TeacherKLPlugin` 用冻结的 teacher policy 给 student policy 增加 KL 约束。它负责加载 teacher checkpoint、按需映射 teacher 观测组、调度 KL loss 系数，并把插件进度写入 checkpoint。

## 最小配置示例

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

插件是可选的。`algorithm.plugins` 为空时就是普通 PPO。

## 安装

```bash
git clone https://github.com/toonasinensis/rsl_rl.git
cd rsl_rl
pip install -e .
```

## 测试

```bash
pytest tests/algorithms/test_amp_motion_loader.py \
       tests/algorithms/test_amp_plugin_step_metrics.py \
       tests/algorithms/test_teacher_kl_plugin.py
```

## 引用

本仓库基于 RSL-RL。如用于研究，请引用原项目。

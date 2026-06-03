rsl_rl 项目架构与使用指南
============================

本文档面向正在使用或准备扩展 ``rsl_rl`` 的开发者，目标不是逐行解释源码，而是帮助你快速建立以下认知：

1. 训练是如何从环境一路流到算法、模型、存储和日志系统的。
2. ``rsl_rl/rsl_rl/`` 下各个模块分别负责什么。
3. 目前有哪些算法变体，它们和通用架构的关系是什么。
4. 在实际项目中应该怎样接入环境、配置训练、导出策略和扩展新能力。

项目定位
--------

当前这份 ``rsl_rl`` 已经不只是一个“标准 PPO 实现”，而是一个以 ``PPO`` 为核心、向多种机器人学习场景扩展的训练框架。它有三个明显特征：

1. 以 ``runner -> algorithm -> model/storage`` 为主链路，结构相对稳定。
2. 通过 ``plugins``、``extensions`` 和专用算法类承载场景化能力。
3. 保持了面向工程使用的能力，包括多 GPU、日志、断点保存、JIT/ONNX 导出。

如果只看最核心的抽象，这个项目可以理解为下面五层：

1. ``VecEnv``: 提供批量环境交互接口。
2. ``OnPolicyRunner``: 驱动采样、更新、保存、导出。
3. ``PPO`` 或其他算法类: 负责 rollout 数据消费、loss 计算、参数更新。
4. ``RolloutStorage``: 缓存采样得到的轨迹并生成 mini-batch。
5. ``Model/Module``: 定义 actor、critic、student、teacher 以及附属网络。

整体训练主链路
--------------

以最常见的 on-policy 训练为例，主流程如下：

1. ``runner`` 向环境读取初始观测 ``obs = env.get_observations()``。
2. ``algorithm.act(obs)`` 用 actor 采样动作，并把 value、log-prob、distribution 参数等写入当前 transition。
3. ``runner`` 在 ``env.step(actions)`` 前后触发 plugin hook。
4. ``algorithm.process_env_step(...)`` 把奖励、done、timeout bootstrap 后的结果写入 ``RolloutStorage``。
5. 一个 iteration 内累计 ``num_steps_per_env`` 步后，``algorithm.compute_returns(obs)`` 计算 GAE return 和 advantage。
6. ``algorithm.update()`` 从 ``RolloutStorage`` 生成 mini-batch，反复执行 forward、loss、backward、gradient sync、clip、optimizer step。
7. ``Logger`` 汇总 loss、episode 指标和 ``step_metrics``，写入 TensorBoard/WandB/Neptune。
8. ``runner.save()`` 可在迭代间隔或训练结束时保存 checkpoint，``export_policy_to_jit()`` 和 ``export_policy_to_onnx()`` 可导出部署模型。

这条链路里最重要的两个切入点是：

1. rollout 侧切入点：``plugin.on_after_act()``、``plugin.on_after_step()``。
2. update 侧切入点：``plugin.on_update_start()``、``plugin.on_per_batch_extra_loss()``、``plugin.on_after_backward()``、``plugin.on_post_backward()``、``plugin.on_post_update()``。

核心模块分层
------------

``env``
^^^^^^^

``env/vec_env.py`` 定义了整个库依赖的环境抽象 ``VecEnv``。任何外部环境只要满足这个接口，就可以接入当前训练栈。

环境至少需要提供：

1. ``num_envs``、``num_actions``、``max_episode_length``、``episode_length_buf``、``device``、``cfg``。
2. ``get_observations() -> TensorDict``。
3. ``step(actions) -> (obs, rewards, dones, extras)``。

其中最关键的是两类约定：

1. 观测必须是 ``TensorDict``，并按 observation group 组织。
2. ``extras`` 中建议提供 ``time_outs``、``log``，现在也允许 plugins 写入 ``step_metrics``。

``env/env_wrapper.py`` 中的 ``IsaaclabVecEnvWrapper`` 负责把 Isaac Lab 风格环境适配成 ``VecEnv``。

``runners``
^^^^^^^^^^^

``runners`` 负责“把算法跑起来”，是最接近业务脚本入口的一层。

``on_policy_runner.py``
   当前通用训练主入口。负责：

   1. 多 GPU 初始化。
   2. 根据配置构造算法。
   3. 执行 rollout/update 主循环。
   4. 触发 plugin 的 rollout hook。
   5. 配置 AMP expert source。
   6. 保存、加载、JIT/ONNX 导出。

``amp_on_policy_runner.py``
   兼容 ``AMPPPO`` 的轻量包装，主要复用通用 ``OnPolicyRunner``。

``smp_on_policy_runner.py``
   在通用 runner 基础上补充 SMP expert data 和 expert sampler 的注入逻辑。

``moe_on_policy_runner.py``
   在通用 runner 基础上补充 expert checkpoint 加载和 expert freeze。

``distillation_runner.py``
   基于 ``OnPolicyRunner``，但在训练前强制检查 teacher 权重是否已加载。

``deparse_policy_runner.py``
   保留一套独立 runner，服务于 ``DeparsePPO`` 这种历史/专用训练路径。它和通用 ``OnPolicyRunner`` 的主循环类似，但保留了 Deparse 侧对 RND、对称性和日志的兼容逻辑。

``algorithms``
^^^^^^^^^^^^^^

算法层决定“如何使用采样数据更新模型参数”。

``ppo.py``
   当前最核心的基座算法，负责：

   1. actor/critic rollout 交互。
   2. GAE return/advantage 计算。
   3. PPO clipped surrogate 和 value loss。
   4. adaptive KL learning rate。
   5. plugin 生命周期。
   6. checkpoint save/load。
   7. 多 GPU 参数同步。

``amp_ppo.py``
   ``AMPPPO`` 是 AMP 的兼容算法包装。它保留“独立 discriminator + AMP reward/loss”的经典入口，但内部已经复用共享的 ``AMPPlugin`` 核心来计算 reward、loss 和 checkpoint 状态。

``smp_ppo.py``
   ``SMPPPO`` 在 PPO 上附加一个 diffusion 风格的 ``SMPDiffusionModel``：

   1. rollout 阶段把 SMP reward 加到环境 reward 上。
   2. update 阶段对 expert SMP observation 计算 denoising loss。

``moe_ppo.py``
   ``MoEPPO`` 使用 ``BackboneMoE`` 作为 actor backbone，额外暴露：

   1. router/expert 相关辅助 loss。
   2. MoE router 指标日志。
   3. expert freeze / load_experts。

``deparse_ppo.py``
   ``DeparsePPO`` 是一条相对独立的专用 PPO 分支，特点是：

   1. 自带 RND 和 symmetry 逻辑，而不是走 plugin 架构。
   2. actor/critic 分离优化器。
   3. 多 GPU 下只同步 actor 和 RND 的梯度，critic 允许各 rank 本地更新。

``distillation.py``
   ``Distillation`` 不是 RL，而是 student-teacher imitation：

   1. student 在环境中执行动作。
   2. teacher 输出 privileged action 作为监督目标。
   3. update 时按时间步序列做行为克隆。

``algorithms/plugins``
^^^^^^^^^^^^^^^^^^^^^^

这部分是当前项目里最值得优先理解的新扩展层。它把很多“附加能力”从专用算法里抽离成了标准生命周期。

``base.py``
   定义两类基类：

   1. ``PPOPlugin``: 提供 rollout hook、update hook、train/eval、save/load hook。
   2. ``AuxLossPlugin``: 面向“往 PPO loss 里加一项辅助损失”的标准基类。

``amp_plugin.py``
   ``AMPPlugin`` 是标准化 AMP 入口，职责包括：

   1. 在 rollout 阶段读取 ``amp`` 观测。
   2. 记录 policy transition。
   3. 计算 AMP reward decomposition 并写入 ``extras["step_metrics"]``。
   4. 在 update 阶段通过 ``on_per_batch_extra_loss()`` 注入 discriminator loss。
   5. 在 ``on_after_backward()`` 里做 discriminator grad clip 和 policy std clamp。
   6. 保存/恢复 provider 和 discriminator 相关状态。

``amp_provider.py``
   把“AMP expert 数据从哪里来”单独抽象成 ``AMPExpertProvider``。

   当前默认实现 ``ExternalAMPProvider`` 支持：

   1. 注入固定 expert data。
   2. 注入 expert sampler。
   3. 可选 policy replay buffer。

``obs_reconstruction.py``
   ``ObsReconstructionPlugin`` 是一个典型的 ``AuxLossPlugin``，它从 actor 的 latent 重建指定观测组，把 reconstruction MSE 作为 PPO 的辅助正则。

``storage``
^^^^^^^^^^^

``storage/rollout_storage.py`` 是所有训练路径共享的数据缓冲区。

它有三个作用：

1. 用 ``Transition`` 记录单步采样数据。
2. 用 ``add_transition()`` 把数据写入 time-major buffer。
3. 在 update 阶段生成 feedforward 或 recurrent mini-batch。

其中需要特别注意：

1. RL 模式会保存 ``values``、``actions_log_prob``、``distribution_params``、``returns``、``advantages``。
2. recurrent 模式会额外保存 actor/critic hidden state，并通过 ``split_and_pad_trajectories()`` 生成带 mask 的批次。
3. distillation 模式走单独的 ``generator()``。

``models``
^^^^^^^^^^

``models`` 层负责把 observation groups 变成 actor/critic/student/teacher 需要的输出。

``backbone_base.py``
   ``BaseModel`` 定义统一模型接口：

   1. 解析 observation groups。
   2. 维护 observation normalization。
   3. 支持 distribution 相关接口。
   4. 预留 recurrent reset、hidden state、JIT/ONNX 导出接口。

``wrapper_stochastic.py``
   ``StochasticWrapper`` 把“确定性 backbone”包装成“带动作分布的 actor”。这使得 actor 的 stochastic shell 和 backbone 可以解耦。

``backbone_mlp.py``
   最常用 backbone，适合纯 1D 观测。

``backbone_cnn.py``
   处理 1D + 2D 混合观测；多个 2D group 可映射到多个 CNN encoder，并支持 actor/critic 共享 CNN。

``backbone_rnn.py``
   在 1D 观测上增加 GRU/LSTM 状态，适合部分可观测任务。

``backbone_moe.py``
   Mixture-of-Experts actor backbone，包含 router、expert、top-k gate、load balance/router z-loss 等逻辑。

``backbone_fsq.py``
   面向编码器/解码器和离散潜变量的专用 backbone，适合更研究型的 latent modeling 路径。

``modules``
^^^^^^^^^^^

``modules`` 是神经网络构件层，供 ``models`` 组合使用。

``mlp.py``
   通用多层感知机。

``cnn.py``
   通用卷积栈，支持多层 conv、norm、pool 和 flatten。

``rnn.py``
   对 GRU/LSTM 做了一层包装，既支持 rollout 时的内部 hidden state，也支持 update 时的显式 hidden state 输入。

``distribution.py``
   定义 actor 动作分布。目前默认提供：

   1. ``GaussianDistribution``。
   2. ``HeteroscedasticGaussianDistribution``。

``normalization.py``
   提供经验均值方差归一化和 discounted reward normalization。

``extensions``
^^^^^^^^^^^^^^

``extensions`` 目前主要是历史上直接耦合到算法内的两类增强模块。

``rnd.py``
   ``RandomNetworkDistillation`` 提供 intrinsic reward、状态归一化、奖励归一化和权重调度。

``symmetry.py``
   负责把 symmetry 配置解析为算法可用结构，本体的 data augmentation/mirror loss 逻辑仍主要写在专用算法里。

从当前架构看，``extensions`` 更偏“旧式增强模块”，而 ``plugins`` 是更推荐的新增能力落点。

``utils``
^^^^^^^^^

``utils`` 集中放各种胶水逻辑，其中最重要的是：

``utils.py``
   1. ``resolve_callable()``: 从字符串解析类或函数。
   2. ``construct_actor_with_shell()``: 构造 actor backbone + stochastic wrapper。
   3. ``resolve_obs_groups()``: 校验和补全 observation set。
   4. ``resolve_optimizer()``、``resolve_nn_activation()``。
   5. recurrent batch 的 split/unpad 工具。

``logger.py``
   ``Logger`` 已经形成比较完整的训练指标体系，职责包括：

   1. episode 统计。
   2. loss 和性能指标记录。
   3. ``step_metrics`` 聚合。
   4. TensorBoard/WandB/Neptune 后端适配。
   5. 代码仓 git diff 存档。

特别是 ``step_metrics`` 现在允许环境或 plugin 在每一步上报命名指标，随后按 episode 汇总并在 ``Train/mean_*`` 路径下输出，这对 AMP reward decomposition 一类能力很重要。

当前算法家族怎么选
------------------

如果你只是想知道“该用哪条路径”，可以按下面理解：

``PPO``
   默认首选。适合标准 actor-critic 连续控制，也是 plugin 化能力的主承载者。

``PPO + plugins``
   当前最推荐的扩展方式。适合加入 AMP、辅助重建、额外正则、训练侧监控等能力，而不必复制一整份算法。

``AMPPPO``
   适合兼容已有 AMP 训练配置或已有 ``amp_cfg``/``amp_runner`` 工作流。它在外观上仍是“专有 AMP 算法”，但内部正逐步复用共享的 AMP plugin 核心。

``SMPPPO``
   适合需要 score matching prior / diffusion prior 的 locomotion 风格任务。

``MoEPPO``
   适合希望用专家混合 actor，并控制 router-only 训练、expert 冻结或 expert checkpoint 注入的场景。

``DeparsePPO``
   适合当前项目里那条独立实验分支，尤其是“单 actor + 多 critic 风格”的分布式训练逻辑。它保留了较多定制行为，不是通用基座首选。

``Distillation``
   适合部署蒸馏、student-teacher imitation 或 privileged-to-policy 迁移。

Plugin 架构与专有算法的关系
--------------------------

当前项目里实际上存在两种扩展思路并行：

1. ``PPO + plugin`` 的通用扩展方式。
2. ``AMPPPO/SMPPPO/MoEPPO/DeparsePPO`` 这样的专用算法分支。

从代码形态上看，它们的匹配情况如下。

完全匹配 plugin 架构的
^^^^^^^^^^^^^^^^^^^^^^^

``PPO``
   plugin 生命周期的基座。

``AMPPlugin``
   已经完整接入 rollout hook、extra loss、save/load、step metrics 和 backward 时序。

``ObsReconstructionPlugin``
   是标准的 ``AuxLossPlugin`` 用法示例。

部分向 plugin 架构收敛的
^^^^^^^^^^^^^^^^^^^^^^^^^

``AMPPPO``
   虽然仍保留专用算法壳，但 AMP reward 和 discriminator training 已经通过内部 ``_amp_plugin_core`` 复用通用实现。这意味着它正在从“重型独立算法”向“兼容包装层”收敛。

尚未迁入 plugin 主架构的
^^^^^^^^^^^^^^^^^^^^^^^^^

``SMPPPO``
   仍然把 reward shaping、expert sampling 和 diffusion loss 写在算法类里。

``MoEPPO``
   虽然已经能把 actor backbone 额外发出的 ``aux_losses``、``moe_metrics`` 接入 update/logging，但它本身不是 plugin 化的。

``DeparsePPO``
   RND、symmetry、分离优化器、多 GPU actor-only sync 都直接写在算法里，是一条相对独立的专用路径。

对后续扩展的建议是：

1. 能写成 plugin 的能力，优先不要再复制一份专用 PPO。
2. 只有在优化器拓扑、分布式策略、数据流本身完全不同的时候，再保留专用算法类。

典型使用方式
------------

标准 PPO 训练
^^^^^^^^^^^^^

这是最基础也最推荐的入口。最小配置通常包括：

1. ``runner.num_steps_per_env``。
2. ``runner.obs_groups``。
3. ``runner.algorithm.class_name = PPO``。
4. ``runner.actor`` 和 ``runner.critic``。

一个最小 YAML 结构可以写成：

.. code-block:: yaml

   runner:
     num_steps_per_env: 24
     save_interval: 100
     obs_groups:
       actor: ["policy"]
       critic: ["policy", "critic"]
     algorithm:
       class_name: PPO
     actor:
       class_name: StochasticWrapper
       distribution_cfg:
         class_name: GaussianDistribution
         init_std: 1.0
       backbone:
         class_name: BackboneMLP
         hidden_dims: [256, 256, 256]
     critic:
       class_name: BackboneMLP
       hidden_dims: [256, 256, 256]

Python 侧调用通常是：

.. code-block:: python

   from rsl_rl.runners import OnPolicyRunner

   runner = OnPolicyRunner(env, train_cfg, log_dir="logs/exp", device="cuda:0")
   runner.learn(num_learning_iterations=1000)

Plugin 化 AMP
^^^^^^^^^^^^

如果希望把 AMP 当成可组合能力，而不是专用算法，优先用 ``PPO + AMPPlugin`` 或 ``algorithm.amp_cfg``。

两种入口本质等价：

1. 在 ``algorithm.plugins`` 中显式配置 ``AMPPlugin``。
2. 在 ``algorithm.amp_cfg`` 中写 AMP 配置，让 ``PPO.construct_algorithm()`` 自动翻译成 ``AMPPlugin``。

然后由 ``OnPolicyRunner`` 自动完成 expert data / sampler 的注入，来源可以是：

1. ``amp_runner.expert_data``。
2. ``amp_runner.expert_data_path``。
3. ``env.get_amp_expert_observations()``。
4. ``env.sample_amp_expert_observations()``。

兼容式 AMPPPO
^^^^^^^^^^^^^

如果已有历史 AMP 配置、旧 checkpoint 或旧训练脚本依赖 ``AMPPPO``，就继续使用：

1. ``algorithm.class_name = AMPPPO``。
2. ``algorithm.amp_cfg`` 中提供 reward/discriminator 相关参数。
3. ``AMPOnPolicyRunner`` 或通用 ``OnPolicyRunner`` 负责 expert source 注入。

这种方式的优势是兼容旧工作流，代价是算法壳仍然更重一些。

SMP / MoE / Distillation
^^^^^^^^^^^^^^^^^^^^^^^^

这些变体都延续同一个使用习惯：先选算法类，再选匹配的 runner 和配置块。

1. ``SMPPPO`` 配 ``smp_cfg``，通常配 ``SMPOnPolicyRunner``。
2. ``MoEPPO`` 配 MoE actor backbone，必要时加 ``moe_runner.freeze_experts`` 或 ``expert_checkpoint_paths``。
3. ``Distillation`` 使用 ``student``/``teacher`` 配置，并通过 ``DistillationRunner`` 启动。

环境接入时需要满足什么
----------------------

一个新环境要稳定接入当前项目，建议满足以下约定：

1. ``get_observations()`` 返回 ``TensorDict``。
2. observation group 命名稳定，便于 ``obs_groups`` 做映射。
3. ``step()`` 返回的 ``rewards`` 和 ``dones`` 形状分别兼容 ``[num_envs]``。
4. 对 time-limit termination 提供 ``extras["time_outs"]``。
5. 对 episode 统计提供 ``extras["log"]`` 或 ``extras["episode"]``。

如果要支持专门能力，还可以额外提供：

1. ``get_amp_expert_observations()`` / ``sample_amp_expert_observations()``。
2. ``get_smp_expert_observations()`` / ``sample_smp_expert_observations()``。

如何扩展新能力
--------------

新增环境
^^^^^^^^

优先实现 ``VecEnv`` 约定，而不是改算法本体。只要 ``TensorDict`` 观测和 ``extras`` 约定正确，大多数训练路径都能复用。

新增 actor/critic 结构
^^^^^^^^^^^^^^^^^^^^^^

优先在 ``models`` 和 ``modules`` 层扩展：

1. 新 backbone 继承 ``BaseModel``。
2. 需要 stochastic actor 时继续复用 ``StochasticWrapper``。
3. 如果要支持导出，补齐 ``as_jit()`` 和 ``as_onnx()``。

新增辅助损失或 rollout 逻辑
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

优先写成 plugin：

1. 只加训练损失，用 ``AuxLossPlugin``。
2. 需要改 rollout 奖励或记录额外 transition，用 ``PPOPlugin``。
3. 需要 checkpoint 能力，复用 ``on_save()``/``on_load()``。

只有在以下情况，才更适合新建专用算法类：

1. rollout 数据结构本身不同。
2. 优化器数量或同步策略明显不同。
3. update 流程不再是 PPO 风格。

新增日志指标
^^^^^^^^^^^^

优先走两种方式：

1. update 侧指标：返回到 ``loss_dict``。
2. rollout 侧逐步指标：写入 ``extras["step_metrics"]``。

前者适合 loss、accuracy、KL 等训练指标，后者适合 reward decomposition、逐步统计量和 episode 内积分指标。

阅读代码时的推荐顺序
--------------------

如果你刚接手这个项目，建议按下面顺序阅读：

1. ``runners/on_policy_runner.py``：先看训练主循环。
2. ``algorithms/ppo.py``：再看 rollout 数据如何进入 PPO update。
3. ``storage/rollout_storage.py``：理解 batch 是怎么生成的。
4. ``models/backbone_base.py``、``backbone_mlp.py``、``wrapper_stochastic.py``：理解 actor/critic 输入输出。
5. ``algorithms/plugins/base.py``、``amp_plugin.py``：理解当前推荐扩展方式。
6. 再按需要阅读 ``amp_ppo.py``、``smp_ppo.py``、``moe_ppo.py``、``deparse_ppo.py``、``distillation.py``。

总结
----

当前 ``rsl_rl`` 的核心逻辑可以概括为一句话：

   以 ``OnPolicyRunner + PPO + RolloutStorage + BaseModel/StochasticWrapper`` 为稳定主干，
   以 ``plugins`` 为通用扩展层，以 ``AMPPPO/SMPPPO/MoEPPO/DeparsePPO/Distillation`` 为场景化分支。

如果你的目标是“稳定使用这个库”，优先掌握 ``VecEnv``、``OnPolicyRunner``、``PPO``、``RolloutStorage`` 和 ``obs_groups`` 配置。

如果你的目标是“继续演进这个库”，优先沿着 ``plugin`` 架构扩展，而不是继续复制更多专用 PPO 变体。

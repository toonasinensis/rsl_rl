# Runners 模块边界定义

`rsl_rl.runners` 的职责是编排训练流程。
它负责把 `env`、`algorithms`、`models`、`logger` 串成一个可执行的 collection + update 工作流，但不负责具体神经网络前向、loss 计算、环境物理推进或 rollout 数据结构实现。

## 1. 模块定位

`runners` 是训练系统的 orchestration 层，核心职责有三类：

1. 初始化训练运行时上下文
2. 定义 rollout / update / logging / checkpoint 的时序关系
3. 在训练期间协调 `env`、`algorithm`、`logger` 之间的数据流

一句话说，runner 决定“什么时候调用谁”，而不决定“每一步具体怎么算”。

## 2. 依赖方向

依赖方向应保持单向：

- `runners` 可以依赖 `env`、`algorithms`、`models`、`utils.logger`
- `runners` 不应被 `algorithms` 或 `models` 反向依赖
- `runners` 不应下沉到底层仿真器 SDK

边界约束：

- 算法细节留在 `algorithms`
- 环境状态推进留在 `env`
- 模型前向与分布逻辑留在 `models`
- 指标缓存与外部日志后端适配留在 `logger`

## 3. 当前对外暴露面

`rsl_rl.runners.__init__` 当前导出：

- `OnPolicyRunner`
- `DistillationRunner`

当前稳定公共抽象以 `OnPolicyRunner` 为主。
`DistillationRunner` 是其特化子类，但当前实现仍处于未完成状态，不应视为和 `OnPolicyRunner` 同等级成熟的公共能力。

## 4. Runner 的核心职责

runner 当前负责：

- 持有训练环境 `env`
- 持有训练配置 `cfg`
- 根据环境首帧观测构建算法对象 `alg`
- 构建并驱动 `Logger`
- 控制训练/评估模式切换
- 驱动 rollout 收集、policy update、日志输出、模型保存与加载
- 在分布式场景下完成进程 rank 配置与参数同步入口

runner 不负责：

- 动作采样数学逻辑
- value / advantage / return 计算
- PPO / distillation loss 计算
- rollout storage 的具体写入结构
- 观测归一化统计更新
- 环境 reset / step 内部实现

## 5. OnPolicyRunner 契约

`OnPolicyRunner` 是当前 `runners` 模块的主契约实现。

### 5.1 构造契约

```python
OnPolicyRunner(
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
)
```

初始化阶段 runner 负责：

1. 保存 `env`、`cfg`、`device`
2. 调用 `_configure_multi_gpu()`
3. 通过 `env.get_observations()` 获取首帧观测
4. 通过 `algorithm.class_name` 解析算法类，并调用 `construct_algorithm(obs, env, cfg, device)`
5. 构造 `Logger`
6. 初始化 `current_learning_iteration`

边界约束：

- runner 不自己构造 actor / critic / storage；这些由 `algorithm.construct_algorithm(...)` 负责
- runner 可以读取首帧观测用于算法构建，但不应缓存额外环境内部状态

### 5.2 `learn()` 契约

`learn(num_learning_iterations, init_at_random_ep_len=False)` 是 runner 的主流程入口。

其职责顺序是：

1. 可选随机化 `env.episode_length_buf`
2. 获取初始观测并切换算法到 train mode
3. 在分布式模式下广播参数
4. 初始化日志 writer
5. 迭代执行：
   - rollout collection
   - return computation
   - policy update
   - logging
   - periodic checkpoint
6. 在训练结束后保存最终 checkpoint 并停止 logger

### 5.3 rollout 编排职责

在每个 rollout step 中，runner 当前负责按固定顺序调用：

1. `actions = self.alg.act(obs)`
2. `plugin.on_after_act(self, obs)` for each plugin
3. `obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))`
4. `check_nan(obs, rewards, dones)` if enabled
5. 将 `obs/rewards/dones` 移到训练 `device`
6. `rewards = plugin.on_after_step(self, obs, rewards, dones, extras)` for each plugin
7. `self.alg.process_env_step(obs, rewards, dones, extras)`
8. `self.logger.process_env_step(rewards, dones, extras, None)`

这部分是 runner 的强时序契约。
插件、算法、logger 都默认依赖这个调用顺序。

### 5.4 update 编排职责

rollout 完成后，runner 负责：

1. 调用 `self.alg.compute_returns(obs)`
2. 调用 `loss_dict = self.alg.update()`
3. 统计 `collect_time` 与 `learn_time`
4. 调用 `self.logger.log(...)`

runner 不解释 `loss_dict` 的内部结构，只负责将其透传给 logger。

## 6. Runner 对 Algorithm 的契约要求

当前 `OnPolicyRunner` 默认算法对象至少需要提供以下接口：

- `construct_algorithm(obs, env, cfg, device)`
- `act(obs)`
- `process_env_step(obs, rewards, dones, extras)`
- `compute_returns(obs)`
- `update()`
- `train_mode()`
- `eval_mode()`
- `save()`
- `load(loaded_dict, load_cfg, strict)`
- `get_policy()`

若启用分布式训练，还要求：

- `broadcast_parameters()`

若算法启用了插件机制，runner 还默认算法对象提供：

- `plugins: list`

其中每个 plugin 可能实现：

- `on_after_act(runner, obs)`
- `on_after_step(runner, obs, rewards, dones, extras)`

边界约束：

- runner 可以调度这些接口
- runner 不应理解算法内部参数、loss 分解、storage 布局
- runner 只消费 plugin 在 on_after_step 后的 rewards 输出

## 7. Runner 对 Env 的契约要求

runner 当前依赖 `env` 提供：

- `get_observations()`
- `step(actions)`
- `device`
- `cfg`
- `num_envs`
- `num_actions`
- `episode_length_buf`
- `max_episode_length`

这意味着：

- runner 不应自己处理环境内部 reset
- runner 只消费统一 `VecEnv` 输出
- runner 不应依赖具体仿真器类名或 SDK 类型

## 8. Runner 对 Logger 的契约要求

runner 当前依赖 `Logger` 提供：

- `init_logging_writer()`
- `process_env_step(rewards, dones, extras, intrinsic_rewards)`
- `log(...)`
- `save_model(path, it)`
- `stop_logging_writer()`

runner 与 logger 的边界是：

- runner 提供时序和原始训练指标
- logger 负责缓冲、聚合、打印和上传
- runner 不应实现重复的统计逻辑

## 9. 设备与分布式边界

### 9.1 设备责任

runner 负责协调“环境 device”和“训练 device”之间的切换：

- `env.step()` 前把 action 移到 `env.device`
- `env.step()` 后把 `obs/rewards/dones` 移到 `runner.device`

这属于 runner 的稳定职责，因为它位于环境执行与算法训练的边界上。

### 9.2 分布式责任

`_configure_multi_gpu()` 负责：

- 读取 `WORLD_SIZE` / `LOCAL_RANK` / `RANK`
- 判定是否分布式训练
- 写回 `cfg["multi_gpu"]`
- 校验 `device` 与 local rank 对齐
- 初始化 `torch.distributed`
- 设置当前 CUDA device

在分布式场景下，runner 还负责在训练开始前调用：

- `self.alg.broadcast_parameters()`

边界约束：

- runner 负责“训练进程级”分布式编排
- 算法负责参数与梯度如何同步

## 10. Checkpoint 与导出边界

### 10.1 保存与加载

`save(path, infos=None)` 的职责是：

- 从 `alg.save()` 获取算法保存内容
- 附加 `iter` 和 `infos`
- 调用 `torch.save`
- 通知 logger 上传模型

`load(path, load_cfg=None, strict=True, map_location=None)` 的职责是：

- 调用 `torch.load`
- 把已加载字典交给 `alg.load(...)`
- 如果算法允许恢复 iteration，则同步 `current_learning_iteration`

runner 不应自行解释 actor / critic / optimizer 的 state_dict 细节。

### 10.2 推理与导出

runner 当前还暴露：

- `get_inference_policy(device=None)`
- `export_policy_to_jit(path, filename="policy.pt")`
- `export_policy_to_onnx(path, filename="policy.onnx", verbose=False)`

这些方法的职责是“组织导出流程”，而不是重新实现导出逻辑：

- JIT / ONNX 兼容模型由 `policy.as_jit()` / `policy.as_onnx()` 提供
- runner 只负责选择输出目录、触发导出与落盘

## 11. DistillationRunner 的边界

`DistillationRunner` 当前继承 `OnPolicyRunner`，只额外做了一件事：

- 在 `learn()` 开始前检查 `self.alg.teacher_loaded`

但当前实现中，`super().learn(...)` 之后会直接抛出：

```python
NotImplementedError("Teacher studet distillation mechanism is not implemented.")
```

因此当前应将其视为：

- 一个“结构上存在”的 runner 扩展点
- 还不是完整稳定的 distillation 训练 runner

文档层面的边界约束应明确：

- 可以继承 `OnPolicyRunner` 复用主流程
- 但若扩展成新 runner，必须保证 `learn()` 在功能上闭合，不能在成功完成主流程后再无条件抛出未实现异常

## 12. 新增 Runner 的实现要求

任何新增 runner 都应满足：

1. 明确自己的训练编排职责，而不是承载算法实现
2. 尽量复用 `OnPolicyRunner` 的共性流程
3. 只通过稳定接口与 `env` / `algorithm` / `logger` 交互
4. 不直接读取模型内部私有字段
5. 不直接操作 rollout storage 的内部 tensor 布局，除非该 runner 明确就是为某类 storage 流程定制
6. 在分布式训练下明确说明谁负责参数同步、谁负责梯度同步
7. 对外暴露的方法语义要完整闭合，不能以“先跑一半再抛异常”的形式提供伪接口

推荐做法：

- 把变体差异限制在 `learn()` 的局部时序
- 把具体数值计算继续下沉到算法层
- 把日志聚合继续交给 `Logger`

## 13. 当前实现状态说明

就当前仓库而言：

- `OnPolicyRunner` 是主要稳定 runner 实现
- `DistillationRunner` 是未完成的扩展态实现
- `runners` 模块当前没有额外的公共抽象基类

因此现阶段 `runners` 模块的稳定边界，应以 `OnPolicyRunner` 已实现并被上游实际依赖的行为为准。

## 14. 一句话原则

`runners` 模块的稳定边界是：
只负责训练流程编排、状态切换和模块协同，不吸收环境计算、模型前向、算法损失或日志聚合这些下层职责。

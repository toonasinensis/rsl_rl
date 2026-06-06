# Env 模块边界定义

`rsl_rl.env` 的职责是把底层仿真器包装成 `rsl_rl` 可消费的统一向量化环境接口。
它负责规范化观测、奖励、终止信号和附加信息的输出格式，但不负责训练算法、模型前向或 rollout 存储。

## 1. 模块定位

`env` 模块是 `simulator -> rsl_rl` 的适配层，核心目标有两点：

1. 向上游暴露统一的 `VecEnv` 抽象
2. 向下游提供稳定的 `TensorDict` 观测与 step 结果格式

依赖方向应保持为：

- `runners` / `algorithms` / `models` 可以依赖 `env`
- `env` 可以依赖具体仿真器 SDK
- `env` 不应反向依赖 `algorithms`、`models`、`storage`

## 2. 当前对外暴露面

`rsl_rl.env.__init__` 当前只导出：

- `VecEnv`

这意味着当前稳定公共接口是 `VecEnv` 抽象类，而不是某个具体仿真器适配实现。

当前目录中的实现归类如下：

- `vec_env.py`
  `env` 模块的核心抽象契约定义。
- `isaacsim_env.py`
  面向 Isaac Sim 的预留适配层，目前仍是占位实现。

边界约束：

- 上游代码应面向 `VecEnv` 编程。
- 新增具体环境类时，不应要求 runner 或 algorithm 感知仿真器差异。
- 未通过 `__init__.py` 显式导出的具体环境类，不应被视为稳定公共 API。

## 3. VecEnv 抽象契约

### 3.1 必备属性

所有 `VecEnv` 实现都必须维护以下公共属性：

- `num_envs: int`
  并行环境数量。
- `num_actions: int`
  单个环境的动作维度。
- `max_episode_length: int | torch.Tensor`
  最大 episode 长度，可以是全局标量，也可以是按环境给出的张量。
- `episode_length_buf: torch.Tensor`
  当前各环境已运行步数。
- `device: torch.device | str`
  环境内部张量所在设备。
- `cfg: dict | object`
  环境配置对象，供 logger 和上游流程读取。

这些属性属于 `env` 对上游的强契约；runner 和 algorithm 已直接依赖它们。

### 3.2 必备方法

`VecEnv` 当前只要求两个抽象方法：

1. `get_observations() -> TensorDict`
2. `step(actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]`

这是 `env` 模块当前唯一稳定的方法级公共接口。

## 4. `get_observations()` 契约

### 4.1 职责

`get_observations()` 返回“当前时刻可供策略/价值网络消费的观测快照”。

它的职责是：

- 将底层仿真器观测整理为 `TensorDict`
- 保证各观测组名称稳定
- 保证 batch 维与 `num_envs` 对齐
- 让上游可以基于 `obs_groups` 对不同模型输入进行分组

它不负责：

- 采样动作
- 拼装 rollout 数据
- 计算 loss
- 维护模型隐状态

### 4.2 返回格式

返回值必须是 `TensorDict`，且每个 key 对应一个“观测组”。

每个观测组必须满足：

- 第一维批大小与 `num_envs` 一致
- 值必须是 `torch.Tensor`
- key 名称稳定且可被配置项 `obs_groups` 引用

示意：

```python
TensorDict(
    {
        "policy": Tensor[num_envs, obs_dim],
        "critic": Tensor[num_envs, critic_obs_dim],
    },
    batch_size=[num_envs],
)
```

### 4.3 与 `models` 的边界

`models` 模块不关心底层仿真器，只关心：

- `TensorDict` 的 key 是否存在
- 每个 key 的 shape 是否稳定
- `obs_groups` 能否把这些 key 映射到 actor / critic / student / teacher 等 observation set

因此 `env` 的稳定承诺是“观测组接口”，不是“具体物理含义”。

## 5. `step()` 契约

### 5.1 输入契约

统一签名：

```python
step(actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]
```

输入 `actions` 必须满足：

- shape 为 `(num_envs, num_actions)`
- 位于环境可接受的 device 上
- 语义上对应“每个并行环境各执行一步动作”

`env` 可以自行完成必要的动作裁剪、类型转换或仿真器格式适配，但这些都是 `env` 内部责任，不应泄漏给上游。

### 5.2 输出契约

`step()` 必须返回四元组：

1. `observations: TensorDict`
2. `rewards: torch.Tensor`
3. `dones: torch.Tensor`
4. `extras: dict`

其中：

- `observations`
  表示执行动作后的下一时刻观测。
- `rewards`
  shape 应与 `num_envs` 对齐；当前上游逻辑接受一维张量，并在 storage 中 reshape 成 `(num_envs, 1)`。
- `dones`
  shape 应与 `num_envs` 对齐；语义为该环境在本 step 后是否终止。
- `extras`
  用于携带 timeout、日志、调试信息等非核心输出。

### 5.3 `step()` 的最小语义保证

任意实现必须保证：

- `observations`、`rewards`、`dones` 来自同一个 step 的一致结果
- `dones[i] == True` 表示第 `i` 个环境本步已终止或重置边界已触发
- 返回值中不得包含 NaN；上游会显式检查 `obs/rewards/dones`
- batch 维必须始终与 `num_envs` 对齐

## 6. `extras` 契约

`extras` 是扩展通道，不是随意字典。
当前代码中已经存在以下稳定约定：

### 6.1 `time_outs`

- 类型：`torch.Tensor`
- 含义：由于时间上限而终止，而非真正进入 terminal state
- 用途：PPO 在 `process_env_step()` 中使用它做 timeout bootstrap

约束：

- 若提供，必须能按环境索引对齐到 `num_envs`
- 语义必须是“time limit termination”，不能混入其他 done 原因

### 6.2 `log`

- 类型：`dict[str, float | torch.Tensor]`
- 含义：额外日志与调试指标

约束：

- key 应为稳定字符串
- value 可以是标量或张量
- 若 value 是张量，上游 logger 会按均值处理

### 6.3 其他扩展字段

允许存在其他字段，但必须遵守：

- 不能破坏已有 `time_outs` / `log` 语义
- 不能把必须字段偷偷塞进 `extras`，而不体现在正式返回值中
- 字段名和 shape 应保持稳定，便于插件或上游流程读取

## 7. episode 生命周期边界

`env` 模块负责维护环境级 episode 生命周期，包括：

- 在每步推进时更新内部仿真器状态
- 维护 `episode_length_buf`
- 判定哪些环境 done
- 在 done 后处理对应环境的 reset 或等价重置逻辑

但 `env` 不负责：

- 模型隐状态 reset
- rollout storage 清理
- return / advantage 计算

这些职责在当前代码中分别由：

- `algorithm.process_env_step()` 调用 `actor.reset(dones)` / `critic.reset(dones)`
- `storage` 负责 transition 记录
- `algorithm.compute_returns()` 负责回报计算

## 8. 与上游模块的接口关系

### 8.1 对 runner 的承诺

`OnPolicyRunner` 当前依赖 `env` 提供：

- `get_observations()`
- `step(actions)`
- `num_envs`
- `num_actions`
- `device`
- `cfg`
- `episode_length_buf`
- `max_episode_length`

因此这些项都应视为稳定公共边界。

### 8.2 对 algorithm 的承诺

`PPO` / `Distillation` 当前依赖：

- `get_observations()` 返回的 `TensorDict` 可直接构建模型和 storage
- `step()` 返回的 `rewards` / `dones` / `extras`
- `extras["time_outs"]` 的 timeout 语义

### 8.3 对 models 的承诺

`models` 当前只依赖观测结构，不依赖具体环境类。
因此 `env` 模块需要保证的是：

- 观测 key 稳定
- 每个 key 的维度稳定
- 与 `obs_groups` 配置兼容

## 9. 新增环境适配器的实现要求

任何新增的仿真器适配类都必须：

1. 继承 `VecEnv`
2. 完整实现 `get_observations()` 与 `step()`
3. 维护 `num_envs`、`num_actions`、`device`、`cfg`、`episode_length_buf`、`max_episode_length`
4. 返回 `TensorDict` 观测，而不是裸 `dict`
5. 保证所有输出 batch 维与 `num_envs` 一致
6. 明确区分真正 terminal 与 `time_outs`
7. 不把算法层逻辑耦合到环境内，例如 advantage、GAE、policy loss、normalization update

推荐但非强制：

- 在环境内部尽量完成底层 SDK 的数据格式转换
- 对外暴露稳定的观测组命名，而不是把底层字段名直接泄漏给上游
- 保持 `cfg` 中有足够信息供 logger 记录环境配置

## 10. 当前实现状态说明

就当前仓库而言：

- `VecEnv` 是唯一明确稳定的 env 公共抽象
- `Isaacsim45Env` 仍是占位类，尚未形成可依赖的具体实现契约
- `env` 模块当前没有定义额外的标准辅助方法，例如 `reset()`、`close()`、`seed()`

因此在现阶段，`env` 模块的稳定边界应以 `VecEnv` 中已经声明并被上游实际使用的属性和方法为准，而不是以潜在的仿真器能力为准。

## 11. 一句话原则

`env` 模块的稳定边界是：
把任意底层仿真器适配成统一的向量化 step/observation 接口，并把训练流程、模型语义和 rollout 管理严格留在上游模块。

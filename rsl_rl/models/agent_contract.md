# Models 模块边界定义

`rsl_rl.models` 的职责是把观测 `obs` 转成策略/价值网络可消费的神经网络输出。
它只负责“模型表达”这一层，不负责 PPO 训练流程、rollout 缓存、环境交互、奖励计算或优化器调度。

## 1. 模块定位

当前 `models` 模块分为两层：

1. `BaseModel` 及其各类 `Backbone*`
2. `StochasticWrapper`

分层职责如下：

### 1.1 Backbone 层

backbone 的职责是：

- 接收 `TensorDict` 形式的观测输入
- 按 `obs_groups` / `obs_set` 组织并拼接所需观测
- 可选执行观测归一化
- 产出动作分布所需的特征或确定性动作头输入
- 在训练态按需返回附加中间结果，例如 `aux_losses`、`moe_gates`、`fsq_level_indices`
- 在需要时管理隐状态、导出 JIT / ONNX 兼容模型

backbone 不负责：

- 动作采样
- log-prob / entropy / KL 计算
- PPO/AMP/MoE 等算法级损失聚合
- rollout buffer 写入

### 1.2 Wrapper 层

`StochasticWrapper` 的职责是：

- 持有一个 `BaseModel` 实例
- 持有一个 `Distribution` 实例
- 用 backbone 输出更新分布参数
- 提供采样、确定性输出、log-prob、entropy、KL 等随机策略语义
- 透传 backbone 的隐状态管理、归一化更新、导出能力

`StochasticWrapper` 不负责：

- 定义特征提取结构
- 直接操作环境观测格式
- 生成训练算法专用损失

## 2. 对外暴露面

`rsl_rl.models.__init__` 当前导出：

- 抽象/基础类型：`BaseModel`
- 具体 backbone：`BackboneMLP`、`BackboneCNN`、`BackboneRNN`、`BackboneFSQ`、`BackboneMoE`
- 向后兼容别名：`MLPModel`、`CNNModel`、`RNNModel`、`FSQModel`、`MoEModel`
- 随机策略包装器：`StochasticWrapper`
- 向后兼容别名：`ActorModel`

边界约束如下：

- 上游若依赖“统一接口”，应面向 `BaseModel` 和 `StochasticWrapper` 编程，而不是依赖某个具体 backbone 的内部字段。
- 具体 `Backbone*` 可以替换，但必须满足 `BaseModel` 契约。
- 训练流程如果需要随机策略能力，应依赖 `StochasticWrapper`，而不是直接拼接 `Distribution` 逻辑。

## 3. Backbone 契约

### 3.1 构造参数契约

所有 backbone 构造函数至少接收以下公共参数：

- `obs: TensorDict`
  用于推导观测维度与初始化归一化器。
- `obs_groups: dict[str, list[str]]`
  定义观测集合到观测字段列表的映射。
- `obs_set: str`
  指定当前模型使用哪一个观测集合，例如 `actor` 或 `critic`。
- `output_dim: int`
  backbone 主输出维度，通常对应动作均值头或价值头维度。
- `**backbone_cfg`
  具体 backbone 自己消费的配置。

backbone 必须只读取 `obs_groups[obs_set]` 中声明的观测键；不得隐式依赖未声明观测。

### 3.2 `forward` 输入契约

统一签名：

```python
forward(
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    train_mode: bool = False,
) -> dict[str, torch.Tensor]
```

输入语义：

- `obs`
  以观测组名为 key 的 `TensorDict` / `dict[str, Tensor]`。
- `masks`
  仅在序列模型或需要 unpad 时使用。
- `hidden_state`
  仅在循环模型中生效。
- `train_mode`
  用于声明是否需要返回训练态附加信息。

### 3.3 `forward` 输出契约

所有 backbone 都必须返回 `dict`，并满足：

- 必含键：`"actions"`
- 可选键：训练或调试所需附加字段

`"actions"` 的含义不是“最终采样动作”，而是：

- 对确定性模型：最终动作输出
- 对随机策略模型：分布头输入，通常是 action mean / logits

可选附加字段约束：

- 字段名必须稳定，避免同义多名
- 不得覆盖 `"actions"`
- 训练流程只能把这些字段视为“可选增强信息”，不能把它们当成所有 backbone 都必须返回的公共契约

当前代码中的附加字段示例：

- `aux_losses`
- `fsq_z_q`
- `fsq_level_indices`
- `moe_router_logits`
- `moe_gates`

## 4. BaseModel 公共职责

`BaseModel` 定义的是 backbone 公共基类契约，而不是完整实现。

### 4.1 BaseModel 必须维护的公共语义

- `self.obs_groups`
  当前模型真正使用的观测组列表。
- `self.obs_dim`
  每个观测组对应的最后一维大小。
- `self.output_dim`
  主输出维度。
- `self.obs_normalization`
  是否启用观测归一化。
- `self.obs_normalizers`
  若启用归一化，则按观测组维护的归一化器集合。
- `is_recurrent: bool`
  标识该 backbone 是否含时序隐状态。

### 4.2 BaseModel 默认提供的能力

- 根据 `obs_groups` / `obs_set` 解析活动观测
- 在 `forward` 前执行可选归一化
- 在 `update_normalization(obs)` 中更新归一化统计量
- 为循环模型预留 `reset` / `get_hidden_state` / `detach_hidden_state`
- 为导出预留 `as_jit` / `as_onnx`

### 4.3 子类扩展约束

任意新增 backbone 必须：

1. 继承 `BaseModel`
2. 保持公共 `forward` 签名不变
3. 返回包含 `"actions"` 的字典
4. 如果支持循环状态，正确实现 `is_recurrent`、`reset`、`get_hidden_state`、`detach_hidden_state`
5. 如果声明支持部署导出，提供 `as_jit` / `as_onnx`
6. 不把算法层对象直接耦合进模型层，例如 PPO batch、optimizer、advantage、returns

## 5. StochasticWrapper 契约

### 5.1 构造契约

```python
StochasticWrapper(
    backbone: BaseModel,
    output_dim: int,
    distribution_cfg: dict,
)
```

约束：

- `backbone` 必须满足 `BaseModel` 契约
- `distribution_cfg` 不能为空，且必须能解析出 `Distribution` 子类
- `output_dim` 必须与分布头输入维度兼容

### 5.2 `forward` 契约

```python
forward(
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
    train_mode: bool = False,
) -> dict[str, torch.Tensor | dict]
```

返回结构固定为：

- `actions`
- `extra`

语义：

- `actions`
  当 `stochastic_output=True` 时为采样结果，否则为确定性输出。
- `extra`
  来自 backbone 除 `"actions"` 以外的全部附加字段。

边界约束：

- wrapper 不解释 `extra` 的业务语义，只负责透传
- wrapper 不应生成新的算法专用字段
- wrapper 对外暴露的 `output_mean`、`output_std`、`output_entropy`、`output_distribution_params`、`get_output_log_prob()`、`get_kl_divergence()` 都基于“最近一次 `forward` 更新后的分布状态”

## 6. 生命周期与状态边界

`models` 模块允许维护的内部状态仅限于：

- 网络参数
- 观测归一化统计量
- 循环隐状态
- 随机策略分布当前参数

`models` 模块不应维护：

- rollout 级缓存
- PPO/AMP 更新步状态
- 环境实例引用
- replay buffer / dataset 游标

状态同步边界：

- `reset()` 只处理模型内部状态，尤其是 RNN 隐状态
- `detach_hidden_state()` 只服务于截断反传
- `update_normalization()` 只更新归一化统计量，不改变训练语义

## 7. 与上下游模块的依赖方向

依赖方向必须保持单向：

- `algorithms` / `runners` / `storage` 可以依赖 `models`
- `models` 可以依赖 `rsl_rl.modules` 中的基础网络块、分布、归一化与工具函数
- `models` 不应反向依赖具体算法实现、runner 流程或 storage 结构

这意味着：

- `models` 可以返回训练态附加张量
- 但这些张量如何进入 loss，由上游算法决定
- `models` 不能在内部直接计算 PPO 总损失或操作 rollout 数据结构

## 8. 当前具体实现的归类

当前各文件应被理解为：

- `wrapper_stochastic.py`
  随机策略包装层，是 actor 类模型的统一外壳。
- `backbone_base.py`
  公共 backbone 基类与最小契约定义。
- `backbone_mlp.py`
  最基础的 1D 观测前馈实现。
- `backbone_rnn.py`
  带隐状态的时序 backbone，实现循环契约。
- `backbone_cnn.py`
  混合 1D/2D 观测的卷积 backbone，并负责导出兼容封装。
- `backbone_fsq.py`
  带编码器/解码器和 FSQ 训练态附加输出的专用 backbone。
- `backbone_moe.py`
  带 router / experts / 辅助损失的专用 backbone。

其中 `FSQ` 和 `MoE` 属于“专用 backbone 扩展”，它们可以返回更多训练态字段，但这些字段不提升为所有模型共享的基础契约。

## 9. 一句话原则

`models` 模块的稳定边界是：
backbone 负责把观测映射成动作头输入与可选附加特征，wrapper 负责把这些输出解释成随机策略分布行为，其余训练流程全部留在上游模块。

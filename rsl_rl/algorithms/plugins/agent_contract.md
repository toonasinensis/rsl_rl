# Plugins 模块边界定义

`rsl_rl.algorithms.plugins` 的职责是为 `PPO` 提供可插拔扩展点。
它允许在不改写 PPO 主体流程的前提下，注入额外 reward 逻辑、额外 loss、额外 metric、附加网络参数、以及 checkpoint 状态。

## 1. 模块定位

`plugins` 是 `algorithms` 的扩展层，而不是独立训练框架。

它的核心目标是：

1. 保持 `PPO` 主流程稳定
2. 通过 hook 在固定时机插入附加逻辑
3. 把算法特化能力做成可组合组件，而不是持续膨胀 `PPO` 主类

一句话说：
plugin 可以扩展 PPO，但不应接管 PPO。

## 2. 依赖方向

依赖方向应保持单向：

- `plugins` 可以依赖 `PPO`、`OnPolicyRunner`、`VecEnv`、`models`、插件私有子模块
- `PPO` 通过 hook 调度 `plugins`
- `runner` 只在少量 rollout hook 上直接调用插件

边界约束：

- 插件不应反向要求 `runner` 或 `PPO` 理解它的内部实现
- 插件应尽量通过公开 hook 和公开对象接口工作
- 插件不应把主训练循环重写到自己内部

## 3. 当前对外暴露面

`rsl_rl.algorithms.plugins.__init__` 当前导出：

- `PPOPlugin`
- `AMPPlugin`
- `TeacherKLPlugin`

其中：

- `PPOPlugin` 是当前唯一公共插件基类契约
- `AMPPlugin` 和 `TeacherKLPlugin` 是基于该契约的具体实现

当前插件体系默认只服务于 `PPO`，并未为 `Distillation` 定义独立插件协议。

## 4. 插件层的核心职责

插件当前允许负责：

- 在算法构建完成后初始化附加对象
- 在 rollout 期间缓存跨步信息
- 在 `env.step()` 后重写 reward
- 在 update 阶段添加额外 loss
- 在 update 结束后上报额外 metric
- 在 train/eval mode 切换时同步自己的模块状态
- 在 save/load 时写入和恢复插件状态

插件不负责：

- 直接调用 `env.step()`
- 决定 rollout 外层循环
- 直接替代 PPO 主损失
- 重写 storage 的 batch 生成机制
- 重新定义 runner / algorithm 的公共接口

## 5. `PPOPlugin` 基础契约

`PPOPlugin` 是所有当前插件的公共基类。
它定义的不是功能实现，而是一组“可在固定时机被调用的 hook”。

### 5.1 最小原则

任意插件子类都必须满足：

1. 可以只覆盖自己需要的 hook
2. 未覆盖的 hook 保持空操作或返回默认值
3. 不破坏 PPO 对其他插件和主流程的时序假设
4. 所有副作用都应局限在当前 hook 允许的范围内

### 5.2 公共 hook 列表

当前 `PPOPlugin` 定义了这些 hook：

- `on_init(ppo, env)`
- `on_after_act(runner, obs)`
- `on_after_step(runner, obs, rewards, dones, extras) -> rewards`
- `on_update_start(ppo)`
- `on_per_batch_extra_loss(ppo, batch) -> dict[str, torch.Tensor]`
- `on_post_backward(ppo)`
- `on_post_update(ppo) -> dict[str, float]`
- `on_train_mode(ppo)`
- `on_eval_mode(ppo)`
- `on_save(ppo, saved_dict)`
- `on_load(ppo, loaded_dict)`

## 6. Hook 调用顺序契约

当前代码里的真实调用顺序如下。

### 6.1 初始化阶段

在 `PPO.construct_algorithm()` 中：

1. `PPO` 构建完成
2. 所有 plugin 实例已经创建
3. 逐个调用 `plugin.on_init(ppo, env)`

此时：

- `ppo.actor` / `ppo.critic` / `ppo.optimizer` / `ppo.storage` 都已可用
- `env` 已可用
- 适合创建附加网络、回放缓冲、normalizer，或向 optimizer 添加参数组

### 6.2 rollout 阶段

在 `OnPolicyRunner.learn()` 的每个环境 step 中：

1. `actions = alg.act(obs)`
2. `plugin.on_after_act(runner, obs)`
3. `obs, rewards, dones, extras = env.step(actions)`
4. `rewards = plugin.on_after_step(runner, obs, rewards, dones, extras)`
5. `alg.process_env_step(obs, rewards, dones, extras)`

语义上：

- `on_after_act()` 适合缓存“当前步但 step 后还会用到”的观测
- `on_after_step()` 是 reward 改写和 step 级统计注入的唯一正式入口

### 6.3 update 阶段

在 `PPO.update()` 中：

1. `plugin.on_update_start(ppo)`
2. 对每个 mini-batch：
   - 先算 PPO 主损失与 backbone 附加损失
   - 再调 `plugin.on_per_batch_extra_loss(ppo, batch)`
   - 其返回值会并入总 loss 和日志
   - `loss.backward()`
   - 若多 GPU，则 `reduce_parameters()`
   - 主模型梯度裁剪
   - TODO 确认这里是否也应加入 hook 做 plugins 独有模块的梯度裁剪
   - `optimizer.step()`
   - `plugin.on_post_backward(ppo)`
3. batch 全部结束后：
   - `plugin.on_post_update(ppo)`

### 6.4 模式切换与 checkpoint 阶段

在 `PPO.train_mode()` / `eval_mode()` 中：

- `plugin.on_train_mode(ppo)`
- `plugin.on_eval_mode(ppo)`

在 `PPO.save()` / `load()` 中：

- `plugin.on_save(ppo, saved_dict)`
- `plugin.on_load(ppo, loaded_dict)`

## 7. 各 hook 的可修改范围

### 7.1 `on_init(ppo, env)`

允许：

- 创建插件私有模块
- 从 `env` 读取环境结构信息
- 向 `ppo.optimizer` 添加参数组
- 初始化 replay buffer / normalizer / 统计量

不应：

- 重新构造 `ppo.actor` / `ppo.critic`
- 重写 `ppo.storage` 的类型
- 修改 runner 级配置流程

### 7.2 `on_after_act(runner, obs)`

允许：

- 读取当前观测
- 缓存下一步要用的中间状态

不应：

- 修改 `actions`
- 直接推进环境
- 写 storage

### 7.3 `on_after_step(runner, obs, rewards, dones, extras)`

允许：

- 基于 step 结果重写 rewards
- 向 `extras` 添加 step 级统计信息
- 使用 `dones` 处理 terminal 边界

强契约：

- 返回值必须是“新的 rewards tensor”
- shape 必须与输入 `rewards` 兼容

不应：

- 返回别的结构代替 rewards
- 改变 `obs` 的公共契约
- 直接写入 algorithm storage

### 7.4 `on_update_start(ppo)`

允许：

- 初始化跨 batch generator
- 重置本轮 update 的统计计数器
- 切换插件模块到合适模式

### 7.5 `on_per_batch_extra_loss(ppo, batch)`

允许：

- 读取 mini-batch
- 运行插件私有网络
- 计算附加 loss
- 返回一个或多个命名 loss

强契约：

- 返回值必须是 `dict[str, torch.Tensor]`
- 所有返回的 tensor 都会被求和并加到总 loss 上
- 这些 key 也会进入日志系统

不应：

- 直接执行 `optimizer.step()`
- 直接清空主优化器梯度
- 修改 batch 结构的基础语义

### 7.6 `on_post_backward(ppo)`

允许：

- 做 step 后的插件收尾逻辑
- 在当前实现下处理不影响本次 step 的状态同步或约束逻辑

### 7.7 `on_post_update(ppo)`

允许：

- 返回额外标量 metric
- 推进插件内部 step 计数
- 汇总本轮 update 的统计量

强契约：

- 返回 `dict[str, float]`
- 这些值会合并进最终 `loss_dict`

### 7.8 `on_train_mode` / `on_eval_mode`

允许：

- 切换插件内部模块的 `train()/eval()`
- 冻结或保持某些模块只在 eval

### 7.9 `on_save` / `on_load`

允许：

- 向 checkpoint 附加插件状态
- 从 checkpoint 恢复插件状态

强契约：

- 不应覆盖主模型已有 key，除非有明确约定
- 恢复逻辑应能容忍 checkpoint 中缺少插件状态

## 8. AMPPlugin 的边界

`AMPPlugin` 是当前最重型的插件实现，它说明了插件体系允许做到的扩展上限。

### 8.1 它当前做了什么

`AMPPlugin` 当前通过 hook 完成：

- `on_init()`
  创建 `Discriminator`、`ReplayBuffer`、`AMPLoader`、`Normalizer`
  并把判别器参数组接入 `ppo.optimizer`
- `on_after_act()`
  缓存当前 AMP 观测
- `on_after_step()`
  计算 AMP reward，覆盖原 reward，并向 `extras["step_metrics"]` 注入分解指标
- `on_update_start()`
  初始化 policy/expert generator
- `on_per_batch_extra_loss()`
  计算判别器损失与梯度惩罚
- `on_post_backward()`
  做判别器梯度裁剪，并可约束 policy std
- `on_post_update()`
  输出判别器预测统计
- `on_save()` / `on_load()`
  保存和恢复判别器权重

### 8.2 体现出的插件边界

这说明当前插件体系允许：

- 挂载附加网络
- 共享主优化器
- 改写 reward
- 使用私有 replay buffer
- 输出 step 级和 update 级 metric

但即使如此，AMPPlugin 仍然没有：

- 接管 PPO 主损失
- 接管 runner 主循环
- 重写 storage 基础语义

## 9. TeacherKLPlugin 的边界

`TeacherKLPlugin` 展示的是“轻量 loss 正则型插件”模式。

### 9.1 它当前做了什么

- `on_init()`
  构建并加载 frozen teacher actor
- `on_update_start()`
  更新当前 KL loss 系数并重置统计
- `on_per_batch_extra_loss()`
  计算 teacher-student KL 附加 loss
- `on_post_update()`
  输出 KL 相关 metric
- `on_train_mode()` / `on_eval_mode()`
  保持 teacher 冻结
- `on_save()` / `on_load()`
  保存和恢复插件步数与系数状态

### 9.2 体现出的插件边界

这说明当前插件体系也支持：

- 引入只读 teacher 模型
- 使用和 student 不同的 observation 映射
- 只在 update 阶段生效，不参与 reward 改写
- 通过 schedule 控制附加 loss 权重

当前限制也很明确：

- 只支持 feed-forward actor
- 依赖 PPO 的分布参数和 KL 接口

## 10. 命名与日志约束

插件返回的命名字段应保持稳定。

### 10.1 loss key

`on_per_batch_extra_loss()` 返回的 key 会：

- 直接进入 `loss_results`
- 直接进入 logger 的 `Loss/...` 命名空间

因此：

- key 应简洁且稳定
- 不应频繁变更
- 不应和 PPO 主损失字段含义混淆

### 10.2 metric key

`on_post_update()` 返回的 key 会并入最终日志字典。

因此：

- metric 名称应稳定
- 尽量避免和主算法已有字段重名

### 10.3 step metrics

若插件向 `extras["step_metrics"]` 写入指标：

- 值应按环境维对齐
- logger 会按环境聚合再记平均

## 11. 新增插件的实现要求

任何新增插件都应满足：

1. 继承 `PPOPlugin`
2. 只覆盖必要 hook
3. 明确每个 hook 的输入输出和副作用范围
4. 若添加 loss，返回命名字典而不是直接篡改主 loss 结构
5. 若添加参数，优先在 `on_init()` 中接入 optimizer
6. 若写 checkpoint，使用稳定 key，并兼容缺失状态恢复
7. 不接管 PPO 主循环和 runner 主循环

推荐做法：

- reward 类扩展优先用 `on_after_step()`
- regularization 类扩展优先用 `on_per_batch_extra_loss()`
- 统计类扩展优先用 `on_post_update()`
- 模块状态切换统一放在 `on_train_mode()` / `on_eval_mode()`

## 12. 当前实现状态说明

就当前仓库而言：

- 插件体系当前是 `PPO` 专属扩展机制
- `PPOPlugin` 是唯一公共插件基类
- `AMPPlugin` 是重型 reward + loss 混合插件
- `TeacherKLPlugin` 是轻量 teacher regularization 插件

因此现阶段 `plugins` 模块的稳定边界，应以 `PPOPlugin` 的 hook 契约和 `PPO.update()/runner.learn()` 的真实调用顺序为准。

## 13. 一句话原则

`plugins` 模块的稳定边界是：
在 PPO 预留的固定 hook 上附加 reward、loss、metric 和 checkpoint 扩展能力，但不接管 PPO 主训练流程，也不破坏 runner、model、storage 的基础契约。

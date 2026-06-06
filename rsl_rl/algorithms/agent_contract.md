# Algorithms 模块边界定义

`rsl_rl.algorithms` 的职责是定义“如何利用模型和 rollout 数据完成训练更新”。
它位于 `runner` 与 `models/storage/plugins` 之间，既要承接 rollout 期间的状态记录，也要负责 loss 计算、参数更新、checkpoint 语义和分布式梯度同步。

## 1. 模块定位

`algorithms` 是训练系统中的“学习语义层”。
它的核心任务不是编排时序，也不是实现网络结构，而是回答下面这些问题：

1. rollout 期间哪些信息需要记录
2. 记录后的数据如何转化为 update 所需 batch
3. loss 如何计算与聚合
4. 参数如何更新、同步、保存和恢复

一句话说：
runner 决定“什么时候训练”，algorithm 决定“训练时算什么、存什么、怎么更新”。

## 2. 依赖方向

依赖方向应保持单向：

- `algorithms` 可以依赖 `env`、`models`、`storage`、`utils`、`plugins`
- `runners` 可以依赖 `algorithms`
- `models`、`storage`、`env` 不应反向依赖具体算法实现

边界约束：

- 环境推进留在 `env`
- 模型结构与分布语义留在 `models`
- batch 缓冲张量布局留在 `storage`
- 插件扩展点定义留在 `plugins`
- 训练时序编排留在 `runner`

## 3. 当前对外暴露面

`rsl_rl.algorithms.__init__` 当前导出：

- `PPO`
- `Distillation`
- `AMPPlugin`
- `PPOPlugin`
- `TeacherKLPlugin`

其中：

- `PPO` 是当前主算法实现，也是最完整的稳定契约来源
- `Distillation` 是 student-teacher 训练的特化算法
- `plugins` 是 `PPO` 侧的扩展机制，不是独立训练入口

## 4. 算法层的核心职责

算法层当前负责：

- 持有可训练模型与优化器
- 持有 rollout storage 与单步 transition 容器
- 定义 `act()` 时要缓存哪些 rollout 信息
- 定义 `process_env_step()` 如何把环境返回写入 storage
- 定义 `compute_returns()` / `update()` 的训练语义
- 负责 train/eval mode 切换
- 负责 checkpoint save/load 语义
- 在多 GPU 训练下负责参数广播与梯度规约
- 在 PPO 中负责插件 hook 的调用点

算法层不负责：

- 驱动整个训练时序循环
- 直接调用 `env.step()` 或决定 rollout 长度
- 管理日志后端与指标打印
- 定义神经网络结构细节

## 5. Runner 依赖的公共算法契约

当前 `OnPolicyRunner` 实际依赖算法对象提供以下接口：

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

若启用分布式训练，还依赖：

- `broadcast_parameters()`

这组接口应视为当前算法层对 runner 的强契约。

## 6. PPO 契约

`PPO` 是当前算法层最完整、最稳定的实现。

### 6.1 初始化职责

`PPO.__init__()` 当前负责：

- 保存 `device` 与 multi-GPU 相关状态
- 持有 `actor`、`critic`
- 创建联合优化器
- 持有 `storage`
- 创建 `transition = RolloutStorage.Transition()`
- 保存 PPO 超参数
- 注册插件列表

边界约束：

- `PPO` 不自己实例化 env
- `PPO` 不自己构造 logger
- `PPO` 不负责 runner 层时间循环

### 6.2 `act()` 契约

`act(obs)` 的职责是：

1. 记录 actor/critic 当前隐状态
2. 通过 actor 采样动作
3. 通过 critic 计算 value
4. 记录动作 log-prob 与分布参数
5. 记录当前观测到 `transition`
6. 返回给 runner 一个可发往环境的 action tensor

这意味着 `act()` 是“rollout 采样 + transition 暂存”的统一入口，而不只是单纯的策略前向。

### 6.3 `process_env_step()` 契约

`process_env_step(obs, rewards, dones, extras)` 的职责是：

- 更新 actor/critic 的观测归一化统计
- 把 reward / done 写入 `transition`
- 基于 `extras["time_outs"]` 做 timeout bootstrap
- 将完整 transition 写入 storage
- 清空单步 transition
- 根据 `dones` reset actor/critic 的隐状态

边界约束：

- `env` 只负责返回 `extras["time_outs"]`
- timeout 奖励修正的训练语义由算法层负责

### 6.4 `compute_returns()` 契约

`compute_returns(obs)` 负责：

- 使用 critic 对最后一个观测求 value bootstrap
- 依据 `gamma` 与 `lam` 计算 GAE/returns
- 生成 `storage.returns` 与 `storage.advantages`
- 按配置决定是否全局归一化 advantage

因此：

- return / advantage 的数学语义属于算法层
- storage 只负责保存结果，不负责推导规则

### 6.5 `update()` 契约

`update()` 是 PPO 的核心优化入口，当前负责：

1. 根据模型是否 recurrent 选择 generator
2. 调用所有 plugin 的 `on_update_start()`
3. 遍历 mini-batch
4. 按需做 mini-batch advantage 归一化
5. 运行 actor/critic 前向
6. 计算 PPO 主损失
7. 合并 backbone 自带 `aux_losses`
8. 合并 plugin 额外 loss
9. 反向传播
10. 在多 GPU 下规约梯度
11. 做梯度裁剪和 `optimizer.step()`
12. 累积日志指标
13. 调用 plugin 的 `on_post_update()`
14. 清空 storage
15. 返回平均 loss dict

这说明 `update()` 的边界不仅是“算 loss”，还包括：

- 选择 batch 生成策略
- 协调插件额外 loss
- 执行优化器 step
- 输出 logger 可消费的标量字典

## 7. PPO 的内部损失分层

当前 PPO 内部大致分三层：

### 7.1 `_forward_model()`

职责：

- 运行 actor / critic 前向
- 取出 `actions_log_prob`、`values`、`distribution_params`、`entropy`
- 从 actor `extra` 中解析 `aux_losses`

它是“模型输出 -> 算法所需张量”的适配层。

### 7.2 `_compute_ppo_loss()`

职责：

- 基于旧 log-prob、advantage、returns 计算 PPO surrogate/value/entropy 项
- 处理自适应 KL 学习率调整

它只负责 PPO 主目标，不处理插件 loss。

### 7.3 `_compute_loss()`

职责：

- 调用 `_compute_ppo_loss()`
- 把 backbone 自带 `aux_losses` 合并到总 loss
- 返回嵌套 loss dict 供日志与扩展使用

因此，当前总 loss 语义是：

- PPO 主损失
- 模型 backbone 附加损失
- 插件附加损失

三者都由算法层统一汇总。

## 8. `construct_algorithm()` 契约

当前 `PPO.construct_algorithm(obs, env, cfg, device)` 负责：

- 解析算法类和 critic 类
- 解析和补全 `obs_groups`
- 构建 actor / critic
- 可选共享 CNN encoder
- 构建 `RolloutStorage`
- 从配置实例化 plugins
- 初始化 `PPO`
- 对每个 plugin 调用 `on_init(alg, env)`

当前 `Distillation.construct_algorithm(...)` 负责：

- 解析 student / teacher 类
- 解析 `obs_groups`
- 检查与 RND / symmetry 扩展的不兼容性
- 构建 student / teacher
- 构建 distillation storage
- 初始化 `Distillation`

边界约束：

- 算法层负责“把配置解析成可训练对象图”
- runner 只负责调用入口，不负责知道 actor/critic/storage 的组装细节

## 9. Distillation 契约

`Distillation` 是当前另一类算法实现，其核心目标不是 PPO 更新，而是行为克隆式 student-teacher 蒸馏。

### 9.1 主要职责

`Distillation` 当前负责：

- 持有 `student` / `teacher`
- 仅对 `student` 建立优化器
- 在 rollout 中记录 student action 与 teacher privileged action
- 在 update 中按 `gradient_length` 聚合行为克隆损失
- 支持从 RL checkpoint 自动加载 teacher 权重

### 9.2 与 PPO 的差异

`Distillation` 与 `PPO` 的关键差异有：

- 不计算 returns / advantages
- 不使用 PPO ratio / clipped value loss
- 不支持 PPO 插件机制
- `compute_returns()` 是 no-op
- `update()` 语义是 behavior cloning 而不是 policy optimization

### 9.3 当前实现状态

`Distillation` 本身实现是闭合的，但与 runner 的组合当前并未完全闭合，因为 `DistillationRunner.learn()` 在主流程后仍会抛出 `NotImplementedError`。

因此文档上应区分：

- 算法实现本身已具备主要训练语义
- 端到端训练入口当前仍不算完全稳定

## 10. Plugins 与算法层的边界

当前只有 `PPO` 明确集成插件机制。

`PPO` 对 plugin 的公共承诺包括：

- `construct_algorithm()` 后调用 `plugin.on_init(alg, env)`
- `update()` 前调用 `plugin.on_update_start(alg)`
- 每个 mini-batch 后调用 `plugin.on_per_batch_extra_loss(alg, batch)`
- 每个 mini-batch 的 backward 后、step 前调用 `plugin.on_per_batch_post_backward(alg)`
- 每个 mini-batch 的 step 后调用 `plugin.on_per_batch_post_step(alg)`
- `update()` 结束后调用 `plugin.on_post_update(alg)`
- train/eval/save/load 时调用对应 hook

边界约束：

- 插件可以扩展 PPO loss、metrics、checkpoint 内容
- 插件不应接管 PPO 主循环
- 插件依赖的是 hook 契约，而不是 runner 或 model 的私有实现

更细的插件边界以 `algorithms/plugins/agent_contract.md` 为准。

## 11. Storage 与算法层的边界

算法层当前依赖 `RolloutStorage` 提供：

- `Transition`
- `add_transition()`
- `clear()`
- `mini_batch_generator()`
- `recurrent_mini_batch_generator()`
- `generator()` for distillation

边界划分如下：

- 算法层定义“要存什么、什么时候存、什么时候读”
- storage 定义“这些数据在内存里怎么排布和产出 batch”

算法层不应直接把 storage 降级成“任意 tensor 容器”来随意读写内部结构。

## 12. Models 与算法层的边界

算法层当前依赖模型提供：

- forward
- `get_output_log_prob()`
- `output_distribution_params`
- `output_entropy`
- `get_kl_divergence()`
- `update_normalization()`
- `reset()`
- `detach_hidden_state()`
- `get_hidden_state()`
- `as_jit()` / `as_onnx()` through runner-export path

边界划分如下：

- 模型负责产生动作分布与 value 输出
- 算法负责解释这些输出如何进入 rollout 和记 loss

因此，算法层不应把网络结构细节写死到自己内部，只应依赖模型契约。

## 13. Checkpoint 契约

算法层负责定义 checkpoint 中“训练语义相关”的内容。

### 13.1 PPO

`save()` 当前至少保存：

- `actor_state_dict`
- `critic_state_dict`
- `optimizer_state_dict`

插件还可以通过 `on_save()` 追加状态。

`load()` 当前支持：

- 选择性加载 actor / critic / optimizer
- 调用 plugin `on_load()`
- 返回是否恢复 iteration 的布尔值给 runner

### 13.2 Distillation

`save()` 当前保存：

- `student_state_dict`
- `teacher_state_dict`
- `optimizer_state_dict`

`load()` 支持：

- 从 distillation checkpoint 恢复 student/teacher/optimizer
- 从 RL checkpoint 自动把 `actor_state_dict` 当成 teacher 加载

## 14. 多 GPU 边界

算法层当前负责的多 GPU 语义包括：

- `broadcast_parameters()`
- `reduce_parameters()`
- 在 PPO 自适应 KL 学习率中做跨卡均值同步

边界约束：

- runner 负责配置进程和调用广播入口
- 算法负责参数和梯度如何同步

当前实现中：

- `PPO.reduce_parameters()` 同步 actor + critic 梯度
- `Distillation.reduce_parameters()` 只同步 student 梯度

这属于算法语义的一部分，不应上推到 runner。

## 15. 新增算法的实现要求

任何新增算法若要接入当前 runner 体系，至少应满足：

1. 提供与 runner 兼容的公共接口
2. 明确 rollout 阶段要缓存哪些 transition 字段
3. 明确 update 阶段如何从 storage 生成 batch 并完成优化
4. 明确 save/load 的 checkpoint 语义
5. 若支持多 GPU，提供参数广播与梯度同步逻辑
6. 若声明支持插件，明确 hook 调用顺序与允许的扩展范围
7. 不把训练时序循环重新塞回算法层

推荐做法：

- 把模型构建集中在 `construct_algorithm()`
- 把 loss 分层成“主损失 + 模型附加损失 + 插件附加损失”
- 把日志输出保留为纯标量字典，交给 logger 处理

## 16. 当前实现状态说明

就当前仓库而言：

- `PPO` 是最主要、最稳定的算法实现
- `Distillation` 是功能特化算法
- 插件体系当前主要服务于 PPO
- `algorithms` 模块没有单独抽象出统一基类

因此现阶段 `algorithms` 模块的稳定边界，应主要以 `PPO` 的公共接口和 `runner` 的实际依赖为准。

## 17. 一句话原则

`algorithms` 模块的稳定边界是：
负责定义 rollout 数据语义、loss 与 update 规则、checkpoint 与多 GPU 同步机制，但不吸收训练总流程编排、模型结构实现或环境执行职责。

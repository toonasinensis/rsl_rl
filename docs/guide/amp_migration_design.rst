AMP Migration Design
====================

This document defines the target architecture for migrating AMP in ``rsl_rl`` from the
current algorithm-specific implementation to a layered plugin-based design.

Background
----------

``rsl_rl`` currently contains a lightweight AMP implementation centered around
``AMPPPO``. The algorithm assumes that expert AMP observations are provided by the
environment or runner through ``expert_data`` or ``expert_sampler`` style inputs.

``rsl_rl_ref`` contains a heavier AMP implementation that owns a larger part of the
training pipeline, including motion-file loading, motion preprocessing, replay storage,
reward decomposition, and discriminator-side normalization.

Both designs are useful, but they solve different problems:

- The lightweight design is a good fit for a general RL library where AMP can act as an
  optional regularizer for locomotion or imitation-adjacent tasks.
- The heavyweight design is a good fit for complete motion-imitation workflows where AMP
  should be usable out of the box from motion files alone.

The goal of this migration is not to choose one and delete the other. The goal is to
make the lightweight AMP core the canonical library abstraction while preserving the
heavyweight pipeline as an optional backend.

Design Goals
------------

The new AMP design should satisfy the following goals:

- Keep AMP usable as a standalone library component that can regularize arbitrary PPO
  training jobs.
- Preserve a complete motion-file-driven AMP workflow for humanoid and motion imitation
  tasks.
- Separate PPO lifecycle logic from expert-data sourcing and motion preprocessing.
- Avoid maintaining multiple unrelated AMP training loops with duplicated reward, loss,
  save/load, and logging logic.
- Make AMP align with the new ``PPOPlugin`` lifecycle already introduced into ``rsl_rl``.
- Keep existing short-term users of ``AMPPPO`` working during the migration.

Non-Goals
---------

The following are explicitly out of scope for this migration phase:

- Rewriting all motion-processing math utilities at the same time.
- Unifying AMP, SMP, MoE, and Deparse in a single large refactor.
- Removing ``AMPPPO`` immediately.
- Forcing all users to adopt motion-file AMP even when they already provide expert data
  from the environment side.

Current State
-------------

Current ``rsl_rl`` structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current AMP path in ``rsl_rl`` has the following shape:

- ``AMPPPO`` owns discriminator reward computation, discriminator loss, save/load, and
  construction.
- ``AMPOnPolicyRunner`` injects expert data or a sampler from config or environment.
- PPO lifecycle hooks now exist in the shared ``PPOPlugin`` interface, but AMP does not
  yet use them for its own internal behavior.

Strengths:

- Clear algorithm boundary.
- Easy to provide expert samples from the environment.
- Reasonable fit for using AMP as an auxiliary regularizer.

Weaknesses:

- AMP logic is embedded inside ``AMPPPO`` rather than modeled as a reusable extension.
- The heavy pipeline from ``rsl_rl_ref`` cannot be plugged into the current AMP path
  without reintroducing algorithm-specific branching.
- Reward decomposition and logging are not yet standardized as AMP-specific plugin
  outputs.

Current migration status in ``rsl_rl``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The migration has now reached the point where the lightweight plugin path is usable for
real integrations.

Already implemented in ``rsl_rl``:

- ``AMPPlugin`` under ``rsl_rl.algorithms.plugins``
- ``AMPExpertProvider`` and ``ExternalAMPProvider``
- reward decomposition into ``extras["step_metrics"]``
- discriminator-side loss optimization through plugin hooks
- gradient-phase plugin hook support via ``on_after_backward()``
- policy std clamping and discriminator grad clipping inside the plugin path
- ``algorithm.amp_cfg`` translation into a standard ``AMPPlugin`` config for
  plugin-driven algorithms such as ``PPO`` and ``SMPPPO``
- shared runner-level AMP expert-source injection for both ``AMPPPO`` and
  plugin-driven AMP

Not yet implemented:

- ``MotionFileAMPProvider``
- full motion-loader, replay-buffer, and normalizer migration from ``rsl_rl_ref``
- full heavyweight-provider parity and final cleanup of the remaining ``AMPPPO`` compatibility shell

Current ``rsl_rl_ref`` structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The reference implementation contains:

- A richer AMP plugin lifecycle integration.
- A motion-file-based pipeline with body-name and anchor configuration.
- Replay-buffer and expert-data generator logic internal to AMP.
- Reward decomposition suitable for step-level logging.

Strengths:

- Complete out-of-the-box motion imitation pipeline.
- Stronger closed-loop ownership over AMP data flow.

Weaknesses:

- Tighter coupling between the algorithm library and motion-data backend.
- Harder to reuse AMP as a small library component for unrelated locomotion tasks.
- More difficult to maintain clean boundaries between PPO core, feature extraction,
  dataset loading, and environment conventions.

Target Architecture
-------------------

Overview
^^^^^^^^

The target design introduces a single AMP training core with multiple expert-data
backends:

- ``AMPPlugin`` becomes the canonical training-time AMP unit.
- ``AMPExpertProvider`` becomes the abstraction for expert and policy AMP data sourcing.
- The heavyweight motion-file implementation becomes one provider implementation rather
  than a separate algorithm family.

The intended layering is:

1. ``PPO`` and ``OnPolicyRunner`` provide the shared training lifecycle.
2. ``AMPPlugin`` attaches to that lifecycle and owns AMP-specific optimization logic.
3. ``AMPExpertProvider`` implementations provide expert samples, optional policy-side
   replay storage, and optional motion preprocessing.
4. Runners or config select which provider implementation to use.

This yields two standard operating modes:

- Lightweight AMP:
  ``PPO + AMPPlugin + ExternalAMPProvider``
- Heavyweight AMP:
  ``PPO + AMPPlugin + MotionFileAMPProvider``

Core AMP Responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^

``AMPPlugin`` should own:

- AMP discriminator construction and parameter registration.
- Per-step reward augmentation or replacement.
- Policy-side AMP transition capture.
- AMP mini-batch loss computation.
- AMP metrics and reward decomposition for logging.
- AMP-specific checkpoint save/load.
- AMP-specific train/eval mode propagation.

``AMPPlugin`` should not own:

- Motion-file I/O details.
- Body-name conventions tied to a specific simulator.
- Environment-specific feature extraction APIs unless wrapped by a provider.

Provider Responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^

``AMPExpertProvider`` should own:

- How expert AMP observations are sourced.
- Optional replay storage for policy AMP state transitions.
- Motion-file loading or environment callback integration.
- Optional normalizers that are semantically tied to the expert-data backend.
- Any adapter logic needed to convert dataset- or simulator-specific structures into the
  canonical AMP tensors used by ``AMPPlugin``.

Provider implementations
^^^^^^^^^^^^^^^^^^^^^^^^

At minimum, the design should support two provider classes.

``ExternalAMPProvider``
"""""""""""""""""""""""

Purpose:

- Preserve current ``rsl_rl`` behavior.
- Accept expert data from config, environment helpers, or custom samplers.

Typical sources:

- ``amp_runner.expert_data``
- ``amp_runner.expert_data_path``
- ``env.get_amp_expert_observations()``
- ``env.sample_amp_expert_observations()``

``MotionFileAMPProvider``
"""""""""""""""""""""""""

Purpose:

- Preserve the complete heavy AMP workflow from ``rsl_rl_ref``.
- Load motion files and produce expert AMP transitions without requiring the environment
  to prebuild expert buffers.

Typical owned features:

- motion-file loading
- body selection
- anchor alignment
- optional provider-local normalizer
- expert transition generators

Proposed Interfaces
-------------------

AMPPlugin
^^^^^^^^^

The plugin should fit the existing ``PPOPlugin`` lifecycle:

.. code-block:: python

   class AMPPlugin(PPOPlugin):
       def on_init(self, ppo, env) -> None:
           ...

       def on_after_act(self, runner, obs) -> None:
           ...

       def on_after_step(self, runner, obs, rewards, dones, extras) -> torch.Tensor:
           ...

       def on_update_start(self, ppo) -> None:
           ...

       def on_per_batch_extra_loss(self, ppo, batch, forward_results=None) -> dict[str, torch.Tensor]:
           ...

       def on_post_update(self, ppo) -> dict[str, float]:
           ...

       def on_save(self, ppo, saved_dict) -> None:
           ...

       def on_load(self, ppo, loaded_dict) -> None:
           ...

AMPExpertProvider
^^^^^^^^^^^^^^^^^

The provider interface should remain small and stable:

.. code-block:: python

   class AMPExpertProvider(Protocol):
       def setup(self, env, device: str) -> None:
           ...

       def record_policy_transition(
           self,
           amp_obs: torch.Tensor,
           next_amp_obs: torch.Tensor,
           dones: torch.Tensor,
       ) -> None:
           ...

       def sample_expert_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
           ...

       def sample_policy_pairs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
           ...

       def state_dict(self) -> dict:
           ...

       def load_state_dict(self, state_dict: dict) -> None:
           ...

Notes:

- ``sample_policy_pairs`` should be optional so purely external providers can skip local
  replay storage if the plugin does not require it.
- A provider may expose additional helper methods internally, but the plugin should rely
  on a narrow common contract.

Canonical Data Contract
-----------------------

The plugin-facing AMP contract should be defined in terms of canonical tensors, not file
formats or simulator names.

Required runtime concepts:

- ``amp_obs``: current AMP observation tensor.
- ``next_amp_obs``: next-step AMP observation tensor.
- ``task_reward``: base environment reward before AMP mixing.
- ``dones``: reset mask for terminal handling.

Recommended conventions:

- The environment or provider should expose AMP data under a consistent group name such as
  ``"amp"``.
- Reward decomposition should write into ``extras["step_metrics"]`` using stable names
  such as ``task_reward``, ``amp_reward``, and ``mixed_reward``.
- The plugin should treat feature extraction as already-resolved input unless a provider
  explicitly owns that conversion.

Configuration Model
-------------------

The new design should move AMP configuration toward a plugin-plus-provider pattern.

Recommended plugin config
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   {
       "algorithm": {
           "class_name": "PPO",
           "plugins": [
               {
                   "class_name": "rsl_rl.algorithms.plugins:AMPPlugin",
                   "reward_scale": 1.0,
                   "loss_coef": 1.0,
                   "gradient_penalty_coef": 10.0,
                   "provider": {
                       "class_name": "rsl_rl.algorithms.plugins.amp:ExternalAMPProvider"
                   },
                   "discriminator": {
                       "class_name": "AMPDiscriminator",
                       "hidden_dims": [256, 256],
                   },
               }
           ],
       }
   }

Recommended standard config entrypoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For new plugin-driven integrations, the preferred entrypoint is now
``algorithm.amp_cfg`` rather than manually spelling out an ``AMPPlugin`` block.

.. code-block:: python

   {
       "algorithm": {
           "class_name": "PPO",
           "amp_cfg": {
               "reward_scale": 1.0,
               "loss_coef": 1.0,
               "gradient_penalty_coef": 10.0,
               "min_normalized_std": 0.2,
               "enable_policy_replay": True,
               "discriminator": {
                   "class_name": "AMPDiscriminator",
                   "hidden_dims": [256, 256],
                   "activation": "elu",
               },
               "expert_data": ...,          # optional
               "expert_sampler": ...,       # optional
           },
       }
   }

This form is internally translated into an ``AMPPlugin`` plus an
``ExternalAMPProvider``.

Notes:

- If you need full manual control, you can still define ``algorithm.plugins``
  explicitly.
- Do not configure both ``algorithm.amp_cfg`` and an explicit ``AMPPlugin`` in
  ``algorithm.plugins`` at the same time. The constructor will raise an error to avoid
  double AMP reward/loss application.

Recommended heavyweight provider config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   {
       "provider": {
           "class_name": "rsl_rl.algorithms.plugins.amp:MotionFileAMPProvider",
           "motion_files": "...",
           "body_names": [...],
           "anchor_name": "pelvis",
           "replay_buffer_size": 100000,
       }
   }

Backward compatibility
^^^^^^^^^^^^^^^^^^^^^^

During migration, the following compatibility paths should remain valid:

- Existing ``AMPPPO`` configs continue to work.
- Existing ``amp_runner`` resource injection continues to work.
- Existing direct expert-data and sampler APIs continue to work.
- Legacy ``algorithm.amp_cfg`` can be internally translated into plugin config where
  practical.

Runner injection behavior
^^^^^^^^^^^^^^^^^^^^^^^^^

Runner-side AMP source discovery is now shared between legacy ``AMPPPO`` and the new
plugin-driven AMP path.

The runner resolves expert sources in the following order:

1. ``amp_runner.expert_data``
2. ``amp_runner.expert_data_path``
3. ``env.get_amp_expert_observations()``

If the environment also exposes ``env.sample_amp_expert_observations()``, that sampler
is attached in addition to the fixed expert-data source.

Implications:

- For legacy ``AMPPPO``, the resolved data/sampler is injected into the algorithm via
  ``set_amp_expert_data`` and ``set_amp_expert_sampler``.
- For plugin-driven AMP, the resolved data/sampler is injected into the configured
  ``AMPPlugin`` provider.
- Environment samplers remain higher-priority at sample time than fixed expert pools, as
  they were in the legacy runner path.

Migration Strategy
------------------

Phase 1: Completed foundation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Already completed:

- ``PPOPlugin`` lifecycle introduced into ``rsl_rl``.
- ``OnPolicyRunner`` rollout hooks introduced.
- ``Logger.step_metrics`` support introduced.

Phase 2: Introduce AMPPlugin without removing AMPPPO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tasks:

- Add ``AMPPlugin`` with the same discriminator, reward, metric, and checkpoint logic as
  the current ``AMPPPO`` implementation.
- Implement ``ExternalAMPProvider`` first, using the current expert-data and sampler
  path.
- Keep ``AMPPPO`` as a compatibility wrapper around the same internal AMP core when
  possible.

Current status:

- The plugin path is now implemented and tested.
- ``AMPPPO`` now reuses the shared AMP plugin core internally for reward, loss,
  lifecycle metrics, provider state, and checkpoint plumbing.
- ``AMPPPO`` is still retained as a compatibility entrypoint rather than removed.

Success criteria:

- ``PPO + AMPPlugin + ExternalAMPProvider`` matches current ``AMPPPO`` behavior for the
  existing test suite.

Phase 3: Port heavy pipeline behind a provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tasks:

- Port motion-loader and replay-buffer logic from ``rsl_rl_ref`` into a provider module.
- Implement ``MotionFileAMPProvider`` using those components.
- Move motion-file-specific config out of the main AMP plugin and into the provider
  configuration.

Success criteria:

- Motion-file-driven AMP works without requiring external expert buffers.
- The PPO and AMP plugin interfaces remain unchanged between lightweight and heavyweight
  modes.

Phase 4: Shrink AMPPPO into compatibility mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tasks:

- Reuse ``AMPPlugin`` internally from ``AMPPPO``.
- Mark ``AMPPPO`` as a compatibility or convenience entrypoint rather than the canonical
  long-term API.

Success criteria:

- AMP logic no longer lives in two unrelated training implementations.

Testing Strategy
----------------

The migration should be guarded by three layers of tests:

1. Core AMP plugin tests

   - reward decomposition
   - checkpoint save/load
   - discriminator loss
   - train/eval propagation

2. Provider tests

   - external sampler provider
   - motion-file provider
   - policy replay behavior
   - terminal-step handling

3. End-to-end PPO integration tests

   - ``PPO + AMPPlugin`` short training
   - ``AMPPPO`` backward compatibility
   - logger ``step_metrics`` output

Risks and Mitigations
---------------------

Risk: duplicate AMP logic survives in both ``AMPPPO`` and ``AMPPlugin``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mitigation:

- Treat ``AMPPlugin`` as the canonical location for new logic.
- Convert ``AMPPPO`` into a wrapper as early as practical after behavior parity is proven.

Risk: provider interface becomes too wide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mitigation:

- Keep the provider contract focused on sample acquisition and state persistence.
- Avoid pushing discriminator or reward semantics into the provider.

Risk: motion-file backend leaks simulator-specific assumptions into PPO core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mitigation:

- Confine simulator naming, body mapping, and anchor semantics to the heavyweight provider.
- Keep the plugin-facing contract tensor-based and simulator-agnostic.

Recommended Decision
--------------------

The recommended long-term design is:

- AMP remains a first-class reusable component in ``rsl_rl``.
- The standard reusable form is a lightweight ``AMPPlugin`` attached to PPO.
- The heavyweight motion-file implementation is preserved as an optional provider backend.
- ``AMPPPO`` is retained for compatibility during migration, but the architecture should
  converge toward one AMP training core with multiple expert-data backends.

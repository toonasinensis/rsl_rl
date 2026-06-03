AMP Implementation Plan
=======================

This document translates the AMP migration design into an executable implementation
plan for ``rsl_rl``.

Scope
-----

This plan focuses on the first implementation batch needed to establish the new AMP
plugin architecture while preserving current behavior.

Out of scope for the first batch:

- Removing ``AMPPPO``
- Porting the full heavy motion-file pipeline from ``rsl_rl_ref``
- Migrating SMP, MoE, or Deparse onto the same provider abstraction
- Rewriting all AMP tests at once

Batch-1 Status
--------------

The first batch is now substantially implemented in ``rsl_rl``.

Implemented:

- ``AMPPlugin``
- ``AMPExpertProvider`` and ``ExternalAMPProvider``
- step-level AMP reward decomposition through ``step_metrics``
- plugin-side discriminator optimization through ``on_per_batch_extra_loss()``
- plugin-side gradient-phase logic through ``on_after_backward()``
- ``algorithm.amp_cfg`` to ``AMPPlugin`` translation for plugin-driven algorithms
- shared runner-level expert-data injection for both ``AMPPPO`` and plugin-driven AMP

Still pending from the broader migration:

- heavy motion-file provider
- final slimming of ``AMPPPO`` as a compatibility shell
- broader cross-algorithm parity and provider reuse beyond AMP

Implementation Strategy
-----------------------

The first batch should optimize for the following:

- Introduce the new AMP abstraction without breaking existing ``AMPPPO`` users.
- Reuse as much current ``AMPPPO`` logic as possible.
- Land a lightweight provider path first.
- Keep plugin-facing interfaces stable enough that the heavy provider can be added later.

The implementation should proceed in two nested tracks:

1. Establish the new AMP plugin and provider interfaces.
2. Make the current ``AMPPPO`` path reuse or mirror those interfaces closely enough that
   later convergence becomes mechanical rather than conceptual.

Target Output of Batch 1
------------------------

At the end of the first batch, the repository should support:

- ``PPO + AMPPlugin + ExternalAMPProvider``
- Existing ``AMPPPO`` continuing to work
- Shared AMP metric names and checkpoint semantics
- Shared runner-level expert-data injection semantics
- Step-level AMP reward decomposition written into ``extras["step_metrics"]``

The first batch does **not** require:

- Motion-file loading in the new plugin path
- Immediate deletion of discriminator logic from ``AMPPPO``
- A full wrapper-based rewrite of ``AMPPPO``

Architecture to Implement in Batch 1
------------------------------------

New modules
^^^^^^^^^^^

Batch 1 should introduce the following new modules under the AMP plugin namespace:

- ``rsl_rl/algorithms/plugins/amp_plugin.py``
- ``rsl_rl/algorithms/plugins/amp_provider.py``

Optional structure if you prefer a package split:

- ``rsl_rl/algorithms/plugins/amp/__init__.py``
- ``rsl_rl/algorithms/plugins/amp/plugin.py``
- ``rsl_rl/algorithms/plugins/amp/provider.py``

Recommended Batch-1 class set:

- ``AMPPlugin``
- ``AMPExpertProvider`` or ``BaseAMPProvider``
- ``ExternalAMPProvider``

Keep the batch-1 interface surface intentionally small.

Reuse candidates from current code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following existing code should be reused rather than rewritten:

- ``AMPDiscriminator`` from ``rsl_rl/algorithms/amp_ppo.py``
- AMP reward computation logic from ``AMPPPO.compute_amp_rewards``
- AMP loss computation logic from ``AMPPPO._compute_amp_loss``
- Existing expert-data normalization conventions already used by the discriminator
- Existing ``AMPOnPolicyRunner`` expert-data discovery path

Recommended refactor style:

- Extract reusable logic into helper methods or shared classes.
- Avoid copying large blocks of AMP logic into both ``AMPPPO`` and ``AMPPlugin``.
- If full extraction is too risky in batch 1, allow temporary duplication only behind
  explicit TODO markers and follow-up tasks.

First Batch Task List
---------------------

Task 1: Add AMP plugin module skeleton
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Create a new ``AMPPlugin`` that fits the current ``PPOPlugin`` lifecycle.

Files:

- New: ``rsl_rl/algorithms/plugins/amp_plugin.py``
- Update: ``rsl_rl/algorithms/plugins/__init__.py``
- Update: ``rsl_rl/algorithms/__init__.py``

Implementation requirements:

- ``AMPPlugin`` inherits from ``PPOPlugin``.
- Constructor accepts a discriminator config and a provider config or provider instance.
- Plugin stores AMP coefficients such as reward scale, loss coefficient, and gradient
  penalty coefficient.
- Plugin exposes clear internal slots for:

  - discriminator
  - provider
  - cached current AMP observations
  - running AMP metrics

Acceptance criteria:

- ``AMPPlugin`` is importable from the public plugin namespace.
- A config can instantiate ``AMPPlugin`` through ``algorithm.plugins``.

Task 2: Add provider abstraction and lightweight provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Define the provider interface and implement the external-data path first.

Files:

- New: ``rsl_rl/algorithms/plugins/amp_provider.py``
- Potential new tests under ``tests/algorithms`` or ``tests/plugins``

Implementation requirements:

- Introduce ``AMPExpertProvider`` or ``BaseAMPProvider``.
- Add ``ExternalAMPProvider`` with support for:

  - fixed expert data
  - callable expert sampler
  - optional local replay storage for policy AMP transitions

- Provider API should minimally support:

  - setup
  - record policy transition
  - sample expert pairs
  - sample policy pairs
  - save/load state

Acceptance criteria:

- Provider can be instantiated from config.
- Provider can accept current ``expert_data`` and ``expert_sampler`` style inputs.

Task 3: Move step-level AMP reward decomposition into the plugin path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Standardize AMP reward reporting so the logger can consume it uniformly.

Files:

- ``rsl_rl/algorithms/plugins/amp_plugin.py``
- Potential reuse from ``rsl_rl_ref/rsl_rl/algorithms/plugins/amp_plugins.py``
- Tests alongside plugin tests

Implementation requirements:

- ``on_after_act`` captures current AMP observations when needed.
- ``on_after_step``:

  - reads next AMP observations
  - records policy transition through the provider
  - computes AMP reward components
  - writes

    - ``task_reward``
    - ``amp_reward``
    - ``mixed_reward``

    into ``extras["step_metrics"]``
  - returns the reward tensor that should be fed into PPO storage

- Terminal-step handling should mirror current AMP semantics.

Acceptance criteria:

- Logger can report AMP reward metrics without AMP-specific logger code.
- Plugin reward behavior is test-covered.

Task 4: Move AMP loss computation into the plugin path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Make discriminator optimization a plugin responsibility for the new path.

Files:

- ``rsl_rl/algorithms/plugins/amp_plugin.py``
- Possibly shared helper extraction from ``rsl_rl/algorithms/amp_ppo.py``

Implementation requirements:

- ``on_update_start`` prepares any provider-side generators or replay handles.
- ``on_per_batch_extra_loss`` returns AMP loss terms.
- ``on_post_update`` returns aggregate AMP metrics such as:

  - discriminator loss
  - policy prediction score
  - expert prediction score

- If discriminator parameters are optimizer-owned through plugin initialization, ensure
  the plugin registers them exactly once.

Acceptance criteria:

- ``PPO + AMPPlugin`` can optimize the discriminator without custom PPO code.
- Plugin save/load includes discriminator state.

Task 5: Wire AMP plugin into runner-level expert-data discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Reuse the current practical convenience path from ``AMPOnPolicyRunner``.

Files:

- ``rsl_rl/runners/amp_on_policy_runner.py``
- Possibly ``rsl_rl/runners/on_policy_runner.py`` only if helper extraction improves reuse

Implementation requirements:

- Detect whether the algorithm is plugin-driven AMP or legacy ``AMPPPO``.
- If plugin-driven:

  - find the configured ``AMPPlugin``
  - pass in expert data or sampler to its provider

- Preserve current config behavior:

  - ``amp_runner.expert_data``
  - ``amp_runner.expert_data_path``
  - ``env.get_amp_expert_observations()``
  - ``env.sample_amp_expert_observations()``

Acceptance criteria:

- Existing convenience behavior remains available in the plugin path.
- No regression for current ``AMPOnPolicyRunner`` tests.

Status:

- Completed.
- ``OnPolicyRunner`` now injects AMP expert data/samplers for both legacy ``AMPPPO``
  and plugin-driven ``AMPPlugin`` paths.

Task 6: Keep AMPPPO behavior intact while aligning interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Avoid a destabilizing rewrite while narrowing the gap between legacy and target paths.

Files:

- ``rsl_rl/algorithms/amp_ppo.py``

Implementation requirements:

- Leave ``AMPPPO`` available and passing tests.
- Where practical, extract shared helper functions that can be used by both
  ``AMPPPO`` and ``AMPPlugin``.
- Add TODO comments marking which pieces are temporary legacy ownership:

  - reward computation
  - provider logic
  - discriminator loss
  - save/load

Acceptance criteria:

- Existing ``AMPPPO`` tests remain green.
- The new plugin path does not require deleting the old path.

Status:

- Partially completed.
- ``AMPPPO`` now reuses the shared ``AMPPlugin`` core for reward computation,
  discriminator loss, update-phase plugin lifecycle, provider-backed expert-state
  handling, and checkpoint plumbing.
- Remaining work is mostly reducing the amount of wrapper code that still lives in
  ``amp_ppo.py`` and eventually deciding how thin the legacy entrypoint should become.

Task 7: Add focused tests for the new AMP plugin path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Lock down the new plugin path before the heavy provider lands.

Files:

- New: ``tests/algorithms/test_amp_plugin.py`` or similar
- New: ``tests/runners/test_amp_plugin_runner.py`` if needed

Minimum test coverage:

- plugin instantiation from config
- reward component insertion into ``extras["step_metrics"]``
- plugin save/load
- discriminator loss returns gradients
- provider receives expert data and sampler
- ``PPO + AMPPlugin`` short update loop executes

Acceptance criteria:

- Plugin tests cover the new path independently of legacy ``AMPPPO`` tests.

Task 8: Add batch-1 compatibility notes to docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal:

- Keep the migration understandable while both paths coexist.

Files:

- ``docs/guide/amp_migration_design.rst``
- Potential release note or changelog entry if the repo uses one

Implementation requirements:

- Mark ``AMPPPO`` as supported legacy path.
- Mark ``AMPPlugin`` as the preferred path for new plugin-based integrations.
- Document that the heavy motion-file provider is planned but not yet the default plugin
  backend.

Acceptance criteria:

- Developers can tell which AMP entrypoint is experimental, preferred, or legacy.

Status:

- In progress.
- The migration documents now describe the standard plugin-driven config path and the
  shared runner injection semantics.

Recommended File-by-File Execution Order
----------------------------------------

To reduce risk, implement the first batch in this order:

1. Add ``AMPPlugin`` and provider module skeletons.
2. Export them publicly.
3. Add minimal plugin tests with fake provider/discriminator behavior.
4. Implement reward decomposition through plugin hooks.
5. Implement loss computation through plugin hooks.
6. Update ``AMPOnPolicyRunner`` to discover plugin-driven AMP.
7. Run and fix tests for both plugin and legacy ``AMPPPO`` paths.
8. Only then extract shared helpers from ``AMPPPO`` if duplication is too high.

Suggested First PR Slice
------------------------

If you want to keep the first code review manageable, the first PR should contain only:

- ``AMPPlugin`` skeleton
- provider abstraction
- ``ExternalAMPProvider``
- reward decomposition with ``step_metrics``
- focused tests
- no heavy motion-file support yet

This first PR should explicitly avoid:

- motion-loader porting
- checkpoint format redesign
- AMPPPO deprecation
- cross-algorithm refactors

Suggested Second PR Slice
-------------------------

The second PR can then add:

- plugin-side discriminator optimization
- runner expert-data injection compatibility
- legacy/helper extraction from ``AMPPPO``
- broader parity tests

Current actual status:

- The first three items in this slice are now implemented.
- The remaining work is mostly legacy/helper convergence and heavyweight provider
  migration.

Suggested Third PR Slice
------------------------

The third PR can focus on the heavy backend:

- motion-file provider
- body-name and anchor config
- replay-buffer and normalizer port
- heavy AMP parity tests against ``rsl_rl_ref`` behavior

Acceptance Checklist for Batch 1
--------------------------------

The batch is complete when all of the following are true:

- ``PPO`` can construct an ``AMPPlugin`` from config.
- ``AMPPlugin`` can consume externally provided expert data or samplers.
- AMP reward components are emitted into ``step_metrics``.
- AMP discriminator loss is optimized through plugin hooks.
- ``AMPOnPolicyRunner`` can feed expert data into the plugin path.
- Existing ``AMPPPO`` behavior still passes tests.
- New plugin-path tests pass.
- Documentation reflects the coexistence of plugin and legacy AMP modes.

Current checklist status:

- ``PPO`` can construct an ``AMPPlugin`` from config: done
- ``AMPPlugin`` can consume externally provided expert data or samplers: done
- AMP reward components are emitted into ``step_metrics``: done
- AMP discriminator loss is optimized through plugin hooks: done
- ``AMPOnPolicyRunner`` can feed expert data into the plugin path: done
- Existing ``AMPPPO`` behavior still passes tests: done
- New plugin-path tests pass: done
- Documentation reflects coexistence of plugin and legacy AMP modes: now updated, with
  heavyweight provider documentation still intentionally forward-looking

Open Questions to Resolve During Implementation
-----------------------------------------------

These questions should be answered while implementing batch 1:

- Should policy replay storage live fully inside the provider, or partially inside the
  plugin?
- Should the provider return already-normalized tensors, or should normalization remain a
  plugin concern?
- Should ``AMPDiscriminator`` stay in ``amp_ppo.py`` temporarily, or be moved into a
  shared module immediately?
- Should plugin metric names mirror current ``AMPPPO`` tags exactly, or use the richer
  reward decomposition names from the reference implementation?

Recommended answer direction:

- Prefer provider-owned policy replay storage.
- Prefer provider-owned backend-specific normalization.
- Move discriminator only if doing so does not destabilize the current tests.
- Prefer richer step-level reward names if they can be added without breaking existing
  dashboards.

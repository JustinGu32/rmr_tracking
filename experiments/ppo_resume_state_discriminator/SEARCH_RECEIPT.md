# Search receipt: resumed PPO state discriminator

- Hypothesis: the first resumed PPO minibatch destroys the retained short gait
  because the loaded Adam state is combined with an unsaved fresh adaptive-KL
  scheduler scalar; synchronizing the scheduler, resetting Adam, or both will
  preserve the categorical 125-action gate.
- Acceptance signal: on one shared 98,304-transition rollout and identical first
  permutation partition, at least one intervention package completes all three
  strict short-reference episodes while the native branch reproduces E010's
  43-action failure.
- Local sources checked: installed RSL-RL `PPO.update` and
  `OnPolicyRunner.load`; E008-E010 runners, packages, rollout tensors, and
  independent audits; the stable RMR trainer and strict source evaluator.
- Baseline result: E010 step zero is 125/125/125; native Adam step one is 43;
  the stored optimizer rate is 2.25e-5, the fresh scheduler scalar is 1e-3,
  and adaptive KL applies 1.5e-3 before that step.
- Exact prior art selected: reuse the installed native PPO update and E010's
  parity-tested permutation instrumentation. No downloaded implementation is
  needed because the exact running dependency already exposes the decisive
  seam.
- Minimal delta: branch only the scheduler scalar and presence of Adam state,
  then execute one native first minibatch per arm from byte-identical model,
  optimizer, rollout, and RNG snapshots. Task, losses, clipping, normalization,
  and evaluator remain unchanged.
- Verdict: adapt the existing E010 experimental seam; do not invent an optimizer
  or change the PPO/AHAC architecture for this test.

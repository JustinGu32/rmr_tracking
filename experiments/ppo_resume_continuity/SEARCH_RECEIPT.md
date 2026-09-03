# Search receipt: synchronized resumed-PPO continuity

- Hypothesis: E011's scheduler synchronization prevents catastrophic forgetting
  beyond the first minibatch and keeps the retained short gait through one
  complete native PPO update; it may or may not acquire any exact-long suffix.
- Acceptance signal: the E012 step-one package is nested-tensor identical to
  E011's verified restored-Adam/synchronized-scheduler package, and every
  preregistered short checkpoint through step 20 categorically completes all
  three strict episodes. Exact-long step 20 is reported independently.
- Local sources checked: E010's parity-tested full-update runner, checkpoint
  sweep, and strict evaluator; E011's factorial runner, independent audit, and
  synchronized package; installed RSL-RL 2.3.1 adaptive-KL update.
- Baseline result: E010's stale 1e-3 scalar applies 1.5e-3 at step one and loses
  short competence immediately. E011's synchronized 2.25e-5 scalar applies
  3.375e-5 and preserves 125/125/125 with either restored or reset Adam.
- External candidates: none are needed. The exact installed implementation and
  two passed local audits expose the cheapest decisive seam more faithfully
  than reimplementing PPO from a paper or another release.
- Minimal delta: subclass E010's runner, change only `PPO.learning_rate` to the
  exact serialized restored optimizer-group rate, then delegate its exact
  rollout, permutation, native 20-step update, package saves, and evaluator.
  Preserve Adam and task state. E012 showed that assigning the nominal Python
  literal instead produces the same model update but fails exact optimizer-
  metadata parity by one floating-point representation quantum; the successor
  copies the group value directly.
- Verdict: adapt the proven E010 seam for one final bounded PPO control; retain
  no checkpoint and transfer only the verified continuity constraint to AHAC.

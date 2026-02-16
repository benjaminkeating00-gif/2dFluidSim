# STOP — Φ-Flow Dye Separation / Jet Control (2026-02-16)

## Goal (what this was)
2D incompressible velocity + two dye inflows (PhiFlow). Learn time-series jet controls (uL/uR) via gradients to separate dye A → right, dye B → left. After inflow shuts off, dye mass should behave predictably.

## What exists
- Differentiable rollout (Colab): 32×40 grid, CenteredGrid dyes, StaggeredGrid velocity
- Two dye inflows, early timesteps only (mask)
- Two wall jets as controllable forcing (time-masked, tanh-bounded, A_MAX up to 1000)
- Objective: purity = (A in REGION_A + B in REGION_B) / total; optimized via math.gradient(...) w.r.t. uL,uR
- Projection each step via fluid.make_incompressible(...)
- Instrumentation: total mass trace, min/max dye values, predictor/corrector breakdown, trajectory visualization

## Stop reason (core technical failure)
After inflow shutoff, total dye mass became untrustworthy under MacCormack/BFECC dye advection: mass increased systematically over time. This makes optimization “success” physically meaningless.

## Key evidence
MacCormack dye mass trace (inflow off after early mask):
t=10 124.67  
t=20 226.22  
t=30 342.18  
t=40 449.30  
t=80 556.81  

Correction-term logging showed persistent positive bias:
delta = corrected − predictor had Σ(delta) > 0 across many timesteps (acts like a weak source term).

Sanity checks:
- Zero inflow for all timesteps → total dye mass stayed exactly 0.
- Pure semi-Lagrangian dye advection → no runaway mass increase (bounded + diffusive).

## Technical conclusion (what I learned)
In this setup, MacCormack/BFECC correction is not conservative. Predictor/backward reconstruction are not symmetric (interpolation + boundaries + backtrace asymmetry), yielding a correction with nonzero global mean and cumulative mass inflation. Semi-Lagrangian is stable but diffusive; MacCormack adds instability/bias.

## Tooling / workflow failure (secondary stop reason)
Trigger: my engineering laptop (Zephyrus) broke, forcing a switch from a normal VS Code + git workflow to Colab/Drive mid-project.

What broke operationally:
- No single source of truth: code split across Drive + notebooks + partial repo; unclear which version is “real”.
- Statefulness: notebook execution order + hidden state made runs non-reproducible and regressions hard to isolate.
- Poor diffability: reviewing changes across cells/notebooks is slow; small edits are hard to track and revert cleanly.
- High friction iteration loop: reconnecting runtimes/GPU, re-mounting Drive, re-running cells, and dealing with session resets added constant overhead.
- Debugging limitations: stepping through code, breakpoints, modular tests, and structured logging were all worse than in a normal file-based project.
- Tool mismatch: relying on Gemini in Colab degraded code quality/accuracy compared to my normal workflow.

Net effect:
Even when the underlying issue was “just one operator,” the environment prevented tight experiment cycles (single-variable changes + clean reruns), so time spent did not reliably produce understanding.

## Artifacts
- Notebook: phi_flow_colab.ipynb (and backup)
- Archive snapshot: /archive/fluid_archive_2026-02-16.zip
- Notes/paper draft: /paper/*
- Instrumentation: mass traces + correction-term sum logs

## If future-me revisits (non-negotiable)
Do not resume from notebook state. Restart with a minimal script harness:
fixed velocity → single dye blob → 1–5 steps → log sum/min/max → add components one at a time.
If MacCormack cannot be made conservative under chosen interpolation/boundaries, accept semi-Lagrangian and move on.

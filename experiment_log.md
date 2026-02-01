## EXP-001 — Mass drift under symmetric inflow (semi-Lagrangian)

### Setup
Two tracers: A and B.  
Inflow mass per step: A = 9.6, B = 9.6.  
Advection: MacCormack (dye), semi-Lagrangian (velocity).  
Projection: incompressible.  
Inflow active for t < 5, disabled for t ≥ 5.  
Jet control activates at t = 10.  
Time horizon: t = 0 → 80.  
Metric: total mass M(t) = ΣA + ΣB.

### Change tested
Baseline configuration. No normalization, clipping, or corrective mass constraint.

### Observation
Inflow shuts off cleanly at t = 5 (added mass = 0 thereafter).  
Despite this, total mass increases:

`t=10 → 124.67 | t=20 → 226.22 | t=30 → 342.18 | t=40 → 449.30 | t=50 → 483.58 | t=60 → 510.00 | t=70 → 540.06 | t=80 → 556.81`

Mass growth continues well after inflow cutoff.

### Notes
Mass is not conserved under this configuration.  
Drift begins before the loss window (70–80).  
Symmetry between tracers degrades over time.  
No explicit sinks or clipping applied.


## EXP-002 — Zero inflow control (sanity check)

### Setup
Two tracers: A and B.  
Inflow mass per step: A = 0.0, B = 0.0 (all timesteps).  
Advection: MacCormack (dye), semi-Lagrangian (velocity).  
Projection: incompressible.  
Jet control activates at t = 10.  
Time horizon: t = 0 → 80.  
Metric: total mass M(t) = ΣA + ΣB.

### Change tested
Inflow disabled for all timesteps (global zero inflow).

### Observation
Total mass remains identically zero for the full rollout:

`t=0 → 0.0 | t=20 → 0.0 | t=40 → 0.0 | t=60 → 0.0 | t=80 → 0.0`

No spontaneous mass generation observed.

### Notes
Confirms no hidden sources in advection, projection, or jet forcing alone.  
Mass drift in EXP-001 requires nonzero inflow history.  
Serves as a baseline sanity check for the simulation pipeline.



## EXP-003 — MacCormack correction bias (mass creation diagnosis)

### Setup
Two tracers: A and B.
Inflow mass per step: A = 9.6, B = 9.6 for t < 5; A = 0.0, B = 0.0 for t >= 5.
Advection: MacCormack (dye), semi-Lagrangian (velocity).
Projection: incompressible.
Jet control activates at t = 10.
Time horizon: t = 0 -> 80.
Metric: total mass M(t) = ΣA + ΣB.
Instrumentation:
  pred = advect.semi_lagrangian(dye, vel, dt=1)
  corr = advect.mac_cormack(dye, vel, dt=1)
  delta = corr - pred
Logged: sum(delta), mean(delta) for A and B each timestep.

### Change tested
Add MacCormack instrumentation to measure whether the correction term (delta)
has a nonzero mean / nonzero spatial sum (i.e., creates net dye mass).

### Observation
After inflow shuts off (t >= 5), the MacCormack correction term is persistently
positive (sum(delta) > 0) for most timesteps, implying net mass creation.

Representative samples (delta = corr - pred):
  t=5:  sum(delta_A)=+0.813357, sum(delta_B)=+0.813352
  t=8:  sum(delta_A)=+1.457604, sum(delta_B)=+1.457606
  t=10: sum(delta_A)=+3.145359, sum(delta_B)=+3.145351
  t=20: sum(delta_A)=+3.338366, sum(delta_B)=+5.681999
  t=30: sum(delta_A)=+9.564965, sum(delta_B)=+7.838605
  t=40: sum(delta_A)=+8.384270, sum(delta_B)=+3.971051
  t=60: sum(delta_A)=+2.149047, sum(delta_B)=+1.658776
  t=79: sum(delta_A)=+0.405100, sum(delta_B)=+0.359195

Total mass growth (M(t) = ΣA + ΣB) despite zero inflow for t >= 5:
  t=10 -> 124.66841
  t=20 -> 226.21545
  t=30 -> 342.18396
  t=40 -> 449.29860
  t=50 -> 483.57660
  t=60 -> 510.00195
  t=70 -> 540.06690
  t=80 -> 556.79100

### Notes
This indicates the MacCormack correction step introduces a positive global bias
(nonzero mean correction), so Σdye is not conserved and can inflate over time.


## EXP-004 — Semi-Lagrangian dye advection (confirmation test)

### Setup
Two tracers: A and B.
Inflow: A = 9.6, B = 9.6 for t < 5; zero for t >= 5.
Advection: semi-Lagrangian (dye), semi-Lagrangian (velocity).
Projection: incompressible.
Jet control starts at t = 10.
Horizon: t = 0 -> 80.
Metric: M(t) = ΣA + ΣB.

### Change tested
Replaced dye advection:
  advect.mac_cormack(...)  →  advect.semi_lagrangian(...)

### Observation
No runaway mass growth observed.
Mass increases slightly early (from prior inflow + stretching),
then gradually diffuses and trends downward.

M(t):
  t=10  -> 106.57815
  t=20  -> 120.74455
  t=30  -> 129.07123
  t=40  -> 121.63263
  t=50  -> 113.14740
  t=60  -> 106.80898
  t=70  -> 103.59132
  t=80  -> 103.00247

### Notes
Mass inflation seen in EXP-003 disappears when MacCormack correction
is removed. Behavior now consistent with diffusive, non-conservative
semi-Lagrangian transport (mild decay, no systematic growth).
Confirms MacCormack correction term was the source of positive bias.


# -*- coding: utf-8 -*-
"""Φ-Flow fluid sim (script version).

Runs a 2D smoke simulation with a bottom-center box inflow,
then computes a simple gradient objective toward a fixed target plume.
"""
import matplotlib
matplotlib.use("TkAgg")
from phi.torch.flow import *
import matplotlib.pyplot as plt

plt.close("all")

# Build two inflow sources: left-middle and right-middle of the domain.
def make_inflow_A():
  return 0.6 * CenteredGrid(
    Box(vec(x=6, y=0), vec(x=10, y=4)),
    extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
  )

def make_inflow_B():
  return 0.6 * CenteredGrid(
    Box(vec(x=22, y=0), vec(x=26, y=4)),
    extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
  )


INFLOW_A = make_inflow_A()
INFLOW_B = make_inflow_B()

# Region masks: right half for dye_A, left half for dye_B.
REGION_A = CenteredGrid(
  Box(vec(x=16, y=0), vec(x=32, y=40)),
  extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
)
REGION_B = CenteredGrid(
  Box(vec(x=0, y=0), vec(x=16, y=40)),
  extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
)

# Define two jet masks as thin rectangles near left/right walls at mid-height.
JET_L = CenteredGrid(
  Box(vec(x=1, y=16), vec(x=3, y=26)),
  extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
)
JET_R = CenteredGrid(
  Box(vec(x=29, y=16), vec(x=31, y=26)),
  extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
)

# Jet action tensors: differentiable time-series controls.
# Left jet blows +x (right), right jet blows -x (left).
NUM_STEPS = 80
CONTROL_START = 40  # timestep when jet control kicks in

# Precomputed control mask: 0 before CONTROL_START, 1 after
# IMPORTANT: Recreate this whenever you change CONTROL_START!
control_mask = math.tensor(
    [0.0 if t < CONTROL_START else 1.0 for t in range(NUM_STEPS)],
    batch('time')
)

# Unconstrained parameters for optimization (will be mapped through tanh)
uL = math.zeros(batch(time=NUM_STEPS))
uR = math.zeros(batch(time=NUM_STEPS))

# Maximum jet strength (bounded via tanh)
A_MAX = 1.0   # keep small for gradient stability; raise to 2, 3, 5 once stable

# Diagnostic: check inflow masses for discretization differences.
print("inflow_A_mass_per_step:", math.sum(INFLOW_A.values))
print("inflow_B_mass_per_step:", math.sum(INFLOW_B.values))

# Perform one physics timestep including advection, buoyancy, and projection.
def step(dye_A: CenteredGrid, dye_B: CenteredGrid, velocity: StaggeredGrid, aL_t: float, aR_t: float):
  dye_A = advect.mac_cormack(dye_A, velocity, dt=1) + INFLOW_A
  dye_B = advect.mac_cormack(dye_B, velocity, dt=1) + INFLOW_B
  total = dye_A + dye_B
  buoyancy_force = (total * vec(x=0, y=0.5)) @ velocity
  jet_force_centered = (aL_t * JET_L - aR_t * JET_R) * vec(x=1, y=0)
  velocity = velocity + (buoyancy_force + jet_force_centered @ velocity)
  velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
  velocity, _ = fluid.make_incompressible(velocity, (), Solve(rel_tol=1e-3, abs_tol=1e-3, max_iterations=1000))

  return dye_A, dye_B, velocity

# Roll out a trajectory and stack dye frames over time for visualization.
def rollout(dye_A0: CenteredGrid, dye_B0: CenteredGrid, vel0: StaggeredGrid, actions_L, actions_R, print_mass=False, debug=False):
  steps = actions_L.shape.get_size('time')
  frames_A = [dye_A0]
  frames_B = [dye_B0]
  dye_A, dye_B, vel = dye_A0, dye_B0, vel0
  for t in range(steps):
    # Use precomputed mask: 0 before CONTROL_START, 1 after
    mask_t = control_mask.time[t]
    aL_t = mask_t * A_MAX * math.tanh(actions_L.time[t])
    aR_t = mask_t * A_MAX * math.tanh(actions_R.time[t])
    
    if debug and t == CONTROL_START:
      print(f"ASSERT t={t}: uL.time[t]={actions_L.time[t]}, tanh(uL.time[t])={math.tanh(actions_L.time[t])}, aL_t={aL_t}")
    
    dye_A, dye_B, vel = step(dye_A, dye_B, vel, aL_t, aR_t)
    
    # Detect NaNs early (only in debug mode to avoid tensor->float during grad)
    if debug:
      vmax = math.max(math.abs(vel.values))
      vmax_f = float(vmax.native().detach())
      if vmax_f != vmax_f:  # NaN check
        print(f"NaN vel at t={t}, aL_t={aL_t}, aR_t={aR_t}")
        break
    
    frames_A.append(dye_A)
    frames_B.append(dye_B)
    
    # Diagnostic print for timesteps around CONTROL_START (after step)
    if debug and CONTROL_START - 2 <= t <= CONTROL_START + 10:
      vel_comp_max = math.max(math.abs(vel.values))
      mass_A = math.sum(dye_A.values)
      mass_B = math.sum(dye_B.values)
      print(f"t={t}: aL_t={aL_t}, aR_t={aR_t}, vel_comp_max={vel_comp_max:.4f}, mass_A={mass_A:.4f}, mass_B={mass_B:.4f}")
    
    # Print mass every 10 timesteps
    if print_mass and (t + 1) % 10 == 0:
      mass_A = math.sum(dye_A.values)
      mass_B = math.sum(dye_B.values)
      total_mass = mass_A + mass_B
      print(f"Timestep {t + 1}: mass_A = {mass_A:.4f}, mass_B = {mass_B:.4f}, total = {total_mass:.4f}")
  traj_A = field.stack(frames_A, batch('traj_time'))
  traj_B = field.stack(frames_B, batch('traj_time'))
  return traj_A, traj_B, dye_A, dye_B

# Differentiable loss function: takes actions, returns negative purity.
def compute_loss(actions_L, actions_R):
  """Run simulation and compute loss = -purity (to maximize separation)."""
  _traj_A, _traj_B, final_A, final_B = rollout(
    initial_dye_A, initial_dye_B, initial_velocity,
    actions_L, actions_R,
    print_mass=False,
    debug=False
  )
  
  # Compute mass in correct region vs wrong region.
  A_in_A = math.sum((final_A * REGION_A).values)
  B_in_B = math.sum((final_B * REGION_B).values)
  A_in_B = math.sum((final_A * REGION_B).values)
  B_in_A = math.sum((final_B * REGION_A).values)
  
  correct = A_in_A + B_in_B
  wrong = A_in_B + B_in_A
  purity = correct / (correct + wrong + 1e-8)
  
  loss = -purity  # minimize negative purity = maximize purity
  return loss

# Objective wrapper for gradient computation.
def objective(uL, uR):
  return compute_loss(uL, uR)

initial_dye_A = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_dye_B = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))

# Gradient descent on actions.
lr = 1e-2
grad_fn = math.gradient(objective, wrt='uL,uR', get_output=True)

print("Initial loss:", compute_loss(uL, uR))

print("Sanity forward loss (no grad):", compute_loss(uL, uR))
(loss, (d_uL, d_uR)) = grad_fn(uL, uR)

# Visualize the gradients over time (2x2: linear on top, log magnitude on bottom)
import numpy as np
fig_grad, ((ax1, ax2), (ax1b, ax2b)) = plt.subplots(2, 2, figsize=(12, 8))
timesteps = list(range(NUM_STEPS))

# Top row: linear gradients
ax1.plot(timesteps, d_uL.native('time'), label='d_uL')
ax1.axvline(x=CONTROL_START, color='r', linestyle='--', label='control start')
ax1.set_xlabel('timestep')
ax1.set_ylabel('gradient')
ax1.set_title('Gradient w.r.t. left jet (uL) - Linear')
ax1.legend()

ax2.plot(timesteps, d_uR.native('time'), label='d_uR')
ax2.axvline(x=CONTROL_START, color='r', linestyle='--', label='control start')
ax2.set_xlabel('timestep')
ax2.set_ylabel('gradient')
ax2.set_title('Gradient w.r.t. right jet (uR) - Linear')
ax2.legend()

# Bottom row: log magnitude (convert to numpy explicitly to avoid deprecation warnings)
gL_native = d_uL.native('time').detach().cpu().numpy()
gR_native = d_uR.native('time').detach().cpu().numpy()
gL = np.abs(gL_native)
gR = np.abs(gR_native)

ax1b.plot(timesteps, np.log10(gL + 1e-12), label='log10(|d_uL|)')
ax1b.axvline(x=CONTROL_START, color='r', linestyle='--', label='control start')
ax1b.set_xlabel('timestep')
ax1b.set_ylabel('log10(|gradient|)')
ax1b.set_title('Gradient w.r.t. left jet (uL) - Log Magnitude')
ax1b.legend()

ax2b.plot(timesteps, np.log10(gR + 1e-12), label='log10(|d_uR|)')
ax2b.axvline(x=CONTROL_START, color='r', linestyle='--', label='control start')
ax2b.set_xlabel('timestep')
ax2b.set_ylabel('log10(|gradient|)')
ax2b.set_title('Gradient w.r.t. right jet (uR) - Log Magnitude')
ax2b.legend()

fig_grad.tight_layout()

# Clip gradients before update
CLIP = 100.0  # start here; adjust
d_uL = math.clip(d_uL, -CLIP, CLIP)
d_uR = math.clip(d_uR, -CLIP, CLIP)

uL = uL - lr * d_uL
uR = uR - lr * d_uR

print("After 1 step, loss:", compute_loss(uL, uR))

# Run simulation with updated actions for visualization.
print("\n--- Mass tracking every 10 timesteps ---")
traj_A, traj_B, final_A, final_B = rollout(initial_dye_A, initial_dye_B, initial_velocity, uL, uR, print_mass=True, debug=True)

# Stack both dye trajectories along a batch dimension so they appear on the same figure.
traj_combined = field.stack([traj_A, traj_B], batch(dye=['A', 'B']))
anim = vis.plot(traj_combined, animate='traj_time')

# --------------------------
# Display any generated plots or animations.
# --------------------------
plt.show(block=True)


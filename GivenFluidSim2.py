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

# Region masks: left half for dye_A, right half for dye_B.
REGION_A = CenteredGrid(
  Box(vec(x=0, y=0), vec(x=16, y=40)),
  extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40),
)
REGION_B = CenteredGrid(
  Box(vec(x=16, y=0), vec(x=32, y=40)),
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

# Jet action arrays: left jet blows +x (right), right jet blows -x (left).
# Each array has one value per timestep.
NUM_STEPS = 80
CONTROL_START = 30  # timestep when jet control kicks in
aL = [10.0] * NUM_STEPS  # strength of left jet at each timestep
aR = [10.0] * NUM_STEPS  # strength of right jet at each timestep

type(INFLOW_A.values.native(INFLOW_A.shape))

# Diagnostic: check inflow masses for discretization differences.
print("inflow_A_mass_per_step:", math.sum(INFLOW_A.values))
print("inflow_B_mass_per_step:", math.sum(INFLOW_B.values))

# Perform one physics timestep including advection, buoyancy, and projection.
def step(dye_A: CenteredGrid, dye_B: CenteredGrid, velocity: StaggeredGrid, aL_t: float, aR_t: float):
  dye_A = advect.mac_cormack(dye_A, velocity, dt=1) + INFLOW_A
  dye_B = advect.mac_cormack(dye_B, velocity, dt=1) + INFLOW_B
  total = dye_A + dye_B
  buoyancy_force = total * (0, 0.5) @ velocity
  jet_force_centered = (aL_t * JET_L - aR_t * JET_R) * vec(x=1, y=0)
  velocity = velocity + (buoyancy_force + jet_force_centered @ velocity)
  velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
  velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))

  return dye_A, dye_B, velocity

# Roll out a trajectory and stack dye frames over time for visualization.
def rollout(dye_A0: CenteredGrid, dye_B0: CenteredGrid, vel0: StaggeredGrid, actions_L: list, actions_R: list):
  steps = len(actions_L)
  frames_A = [dye_A0]
  frames_B = [dye_B0]
  dye_A, dye_B, vel = dye_A0, dye_B0, vel0
  for t in range(steps):
    # No jet control until CONTROL_START
    aL_t = 0.0 if t < CONTROL_START else actions_L[t]
    aR_t = 0.0 if t < CONTROL_START else actions_R[t]
    dye_A, dye_B, vel = step(dye_A, dye_B, vel, aL_t, aR_t)
    frames_A.append(dye_A)
    frames_B.append(dye_B)
  traj_A = field.stack(frames_A, batch('time'))
  traj_B = field.stack(frames_B, batch('time'))
  return traj_A, traj_B

initial_dye_A = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_dye_B = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))

traj_A, traj_B = rollout(initial_dye_A, initial_dye_B, initial_velocity, aL, aR)

# Grab the final frame from each trajectory.
final_A = traj_A.time[-1]
final_B = traj_B.time[-1]

# Compute mass in correct region vs wrong region.
A_in_A = math.sum((final_A * REGION_A).values)  # dye A in left half (correct)
B_in_B = math.sum((final_B * REGION_B).values)  # dye B in right half (correct)
A_in_B = math.sum((final_A * REGION_B).values)  # dye A in right half (wrong)
B_in_A = math.sum((final_B * REGION_A).values)  # dye B in left half (wrong)

# Overlap diagnostic: how much dye_A and dye_B occupy the same cells.
overlap = math.sum((final_A * final_B).values)

# Aggregate metrics.
correct = A_in_A + B_in_B
wrong   = A_in_B + B_in_A
total   = correct + wrong

purity = correct / (total + 1e-8)                    # 0..1, higher = better
normalized_sep = (correct - wrong) / (total + 1e-8) # -1..1, higher = better
wrong_frac = wrong / (total + 1e-8)                 # 0..1, lower = better

print("correct:", correct, "wrong:", wrong, "total:", total)
print("purity:", purity)
print("normalized_sep:", normalized_sep)
print("wrong_frac:", wrong_frac)

# Per-dye purity.
A_total = A_in_A + A_in_B
B_total = B_in_A + B_in_B

A_purity = A_in_A / (A_total + 1e-8)  # fraction of A on correct side
B_purity = B_in_B / (B_total + 1e-8)  # fraction of B on correct side

print("A_purity:", A_purity, "B_purity:", B_purity)

# Normalized overlap (scale-free).
A_mass = math.sum(final_A.values)
B_mass = math.sum(final_B.values)
overlap_norm = overlap / (A_mass * B_mass + 1e-8)

print("A_mass:", A_mass, "B_mass:", B_mass)
print("overlap:", overlap, "overlap_norm:", overlap_norm)

# Stack both dye trajectories along a batch dimension so they appear on the same figure.
traj_combined = field.stack([traj_A, traj_B], batch(dye=['A', 'B']))
anim = vis.plot(traj_combined, animate='time')

# --------------------------
# Display any generated plots or animations.
# --------------------------
plt.show(block=True)


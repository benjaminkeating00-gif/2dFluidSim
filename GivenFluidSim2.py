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

INFLOW_XS = [6, 12, 20, 26]  # left -> right
animations = []

def make_inflow(xpos):
  loc = vec(x=xpos, y=2)
  return 0.6 * CenteredGrid(
    Box(loc - vec(x=3, y=2), loc + vec(x=3, y=2)),
    extrapolation.BOUNDARY,
    x=32, y=40,
    bounds=Box(x=32, y=40),
  )

INFLOW = make_inflow(16)
TARGET = CenteredGrid(
    Box(vec(x=16, y=20), vec(x=32, y=40)),
    extrapolation.ZERO,
    x=32, y=40,
    bounds=Box(x=32, y=40),
)

type(INFLOW.values.native(INFLOW.shape))

# --------------------------
# Gradient objective
# --------------------------

def simulate(smoke: CenteredGrid, velocity: StaggeredGrid):
  for _ in range(20):
    smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
    buoyancy_force = smoke * (0, 0.5) @ velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
    velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
  loss = -math.sum((smoke * TARGET).values)

  return loss, smoke, velocity

initial_smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))

for x in INFLOW_XS:
  INFLOW = make_inflow(x)  # uses same variable name simulate() expects
  loss, final_smoke, _ = simulate(initial_smoke, initial_velocity)
  print(f"inflow x={x}  loss={loss}")
  frames = [initial_smoke]
  smoke_t = initial_smoke
  vel_t = initial_velocity
  for _ in range(20):
    smoke_t = advect.mac_cormack(smoke_t, vel_t, dt=1) + INFLOW
    buoyancy_force = smoke_t * (0, 0.5) @ vel_t
    vel_t = advect.semi_lagrangian(vel_t, vel_t, dt=1) + buoyancy_force
    vel_t, _ = fluid.make_incompressible(vel_t, (), Solve(rank_deficiency=0))
    frames.append(smoke_t)
  traj = field.stack(frames, batch('time'))
  animations.append(vis.plot(traj, animate='time', title=f"x={x} animation"))
# sim_grad = field.gradient(simulate, wrt='velocity', get_output=False)
# velocity_grad = sim_grad(initial_smoke, initial_velocity)
#
# vis.plot(velocity_grad)
#
# print(f"Initial loss: {simulate(initial_smoke, initial_velocity)[0]}")
# print(f"Initial loss: {simulate(initial_smoke, initial_velocity)[0]}")
# initial_velocity -= 0.01 * velocity_grad
# print(f"Next loss: {simulate(initial_smoke, initial_velocity)[0]}")
#
# sim_grad = field.gradient(simulate, wrt='velocity', get_output=True)
#
# for opt_step in range(5):
#   (loss, final_smoke, _v), velocity_grad = sim_grad(initial_smoke, initial_velocity)
#   print(f"Step {opt_step}, loss: {loss}")
#   initial_velocity -= 0.01 * velocity_grad

# --------------------------
# Show all plots/animations
# --------------------------
vis.show()
plt.show()


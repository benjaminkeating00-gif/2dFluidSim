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

# Build a reusable inflow source centered near the bottom of the domain.
def make_inflow(xpos):
  loc = vec(x=xpos, y=2)
  return 0.6 * CenteredGrid(
    Box(loc - vec(x=3, y=2), loc + vec(x=3, y=2)),
    extrapolation.BOUNDARY,
    x=32, y=40,
    bounds=Box(x=32, y=40),
  )

INFLOW = make_inflow(16)

type(INFLOW.values.native(INFLOW.shape))

# Perform one physics timestep including advection, buoyancy, and projection.
def step(smoke: CenteredGrid, velocity: StaggeredGrid):
  smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
  buoyancy_force = smoke * (0, 0.5) @ velocity
  velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
  velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))

  return smoke, velocity

# Roll out a trajectory and stack smoke frames over time for visualization.
def rollout(smoke0: CenteredGrid, vel0: StaggeredGrid, steps: int):
  frames = [smoke0]
  smoke, vel = smoke0, vel0
  for _ in range(steps):
    smoke, vel = step(smoke, vel)
    frames.append(smoke)
  return field.stack(frames, batch('time'))

initial_smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))

traj = rollout(initial_smoke, initial_velocity, steps=80)
vis.plot(traj, animate='time')

# --------------------------
# Display any generated plots or animations.
# --------------------------
vis.show()
plt.show()


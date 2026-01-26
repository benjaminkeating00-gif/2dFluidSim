# -*- coding: utf-8 -*-
"""Φ-Flow fluid sim (script version).

Runs a 2D smoke simulation with a bottom-center box inflow,
then computes a simple gradient objective toward a fixed target plume.
"""
# !pip install --quiet phiflow
from phi.flow import *
import matplotlib.pyplot as plt

# --- Simulation setup ---
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))  # sampled at cell centers
velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))  # sampled in staggered form at face centers
INFLOW_LOCATION = vec(x=16, y=2)
INFLOW = 0.6 * CenteredGrid(
  Box(INFLOW_LOCATION - vec(x=3, y=2), INFLOW_LOCATION + vec(x=3, y=2)),
  extrapolation.BOUNDARY,
  x=32,
  y=40,
  bounds=Box(x=32, y=40),
)
print(f"Smoke: {smoke.shape}")
print(f"Velocity: {velocity.shape}")
print(f"Inflow: {INFLOW.shape}")
print(f"Inflow, spatial only: {INFLOW.shape.spatial}")
print(smoke.values)
print(velocity.values)
print(INFLOW.values)

# --- Single-step update ---
smoke += INFLOW
buoyancy_force = smoke * (0, 0.5) @ velocity
velocity += buoyancy_force
velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))

vis.plot(smoke)

# --- Time integration ---
trajectory = [smoke]
for i in range(20):
  print(i, end=' ')
  smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
  buoyancy_force = smoke * (0, 0.5) @ velocity
  velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
  velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
  trajectory.append(smoke)
trajectory = field.stack(trajectory, batch('time'))
anim = vis.plot(trajectory, animate='time')

# --- Differentiable section (PyTorch backend) ---
# from phi.jax.flow import *
from phi.torch.flow import *
# from phi.tf.flow import *
INFLOW_LOCATION = vec(x=16, y=2)
INFLOW = 0.6 * CenteredGrid(
  Box(INFLOW_LOCATION - vec(x=3, y=2), INFLOW_LOCATION + vec(x=3, y=2)),
  extrapolation.BOUNDARY,
  x=32,
  y=40,
  bounds=Box(x=32, y=40),
)
TARGET = CenteredGrid(
  Sphere(center=vec(x=16, y=34), radius=4),
  extrapolation.BOUNDARY,
  x=32,
  y=40,
  bounds=Box(x=32, y=40),
)
type(INFLOW.values.native(INFLOW.shape))

def simulate(smoke: CenteredGrid, velocity: StaggeredGrid):
  for _ in range(20):
    smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
    buoyancy_force = smoke * (0, 0.5) @ velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
    velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
  loss = math.sum(field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(TARGET), 1, 1, 10)))
  return loss, smoke, velocity
initial_smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))
sim_grad = field.gradient(simulate, wrt='velocity', get_output=False)
velocity_grad = sim_grad(initial_smoke, initial_velocity)

vis.plot(velocity_grad)
print(f"Initial loss: {simulate(initial_smoke, initial_velocity)[0]}")
initial_velocity -= 0.01 * velocity_grad
print(f"Next loss: {simulate(initial_smoke, initial_velocity)[0]}")

sim_grad = field.gradient(simulate, wrt='velocity', get_output=True)

for opt_step in range(4):
  (loss, final_smoke, _v), velocity_grad = sim_grad(initial_smoke, initial_velocity)
  print(f"Step {opt_step}, loss: {loss}")
  initial_velocity -= 0.01 * velocity_grad
# Show all plots/animations when running as a script
vis.show()
plt.show()


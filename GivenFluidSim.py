# -*- coding: utf-8 -*-
"""PhiFlow fluid simulation demo adapted from the official tutorial."""
print("STARTING", __file__)

# !pip install --quiet phiflow
from phi.flow import *
import matplotlib.pyplot as plt

"""## Setting up a Simulation

Create smoke and velocity grids for the simulation.
"""
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))  # sampled at cell centers
velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))  # sampled in staggered form at face centers
BUOY = 1

"""Define a box-shaped inflow and batch over multiple inflow locations."""
INFLOW_LOCATION = tensor([(4, 5), (8, 5), (12, 5), (16, 5)], batch('inflow_loc'), channel(vector='x,y'))
INFLOW = 0.6 * CenteredGrid(Box(INFLOW_LOCATION - vec(x=3, y=3), INFLOW_LOCATION + vec(x=3, y=3)), extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))

"""Print grid shapes to verify dimensions."""
print(f"Smoke: {smoke.shape}")
print(f"Velocity: {velocity.shape}")
print(f"Inflow: {INFLOW.shape}")
print(f"Inflow, spatial only: {INFLOW.shape.spatial}")

"""Print grid values for inspection."""
print(smoke.values)
print(velocity.values)
print(INFLOW.values)

"""## Running the Simulation

Add inflow, apply buoyancy, and project to incompressible flow.
"""
smoke += INFLOW
buoyancy_force = smoke * (0, BUOY) @ velocity
velocity += buoyancy_force
velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))

plt.close('all')
vis.plot(smoke)
plt.show()

"""Run a longer simulation with advection and buoyancy."""
trajectory = [smoke]
for i in range(80):
  print(i, end=' ')
  smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
  buoyancy_force = smoke * (0, BUOY) @ velocity
  velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
  velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
  trajectory.append(smoke)
trajectory = field.stack(trajectory, batch('time'))
plt.close('all')
vis.plot(trajectory, animate='time')
plt.show()

"""## Obtaining Gradients

Switch to a differentiable backend to compute gradients.
"""
# from phi.jax.flow import *
from phi.torch.flow import *
# from phi.tf.flow import *

"""Recreate the inflow with the differentiable backend."""
INFLOW_LOCATION = tensor([(4, 5), (8, 5), (12, 5), (16, 5)], batch('inflow_loc'), channel(vector='x,y'))
INFLOW = 0.6 * CenteredGrid(Box(INFLOW_LOCATION - vec(x=3, y=3), INFLOW_LOCATION + vec(x=3, y=3)), extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))

"""Verify backend type for the inflow tensor."""
type(INFLOW.values.native(INFLOW.shape))

"""Define a differentiable loss and simulate to get gradients."""
def simulate(smoke: CenteredGrid, velocity: StaggeredGrid):
  for _ in range(20):
    smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
    buoyancy_force = smoke * (0, 0.5) @ velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
    velocity, _ = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
  loss = math.sum(field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(smoke.inflow_loc[-1]), 1, 1, 10)))
  return loss, smoke, velocity

"""Initialize velocity with the batch dimension for gradients."""
initial_smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
initial_velocity = StaggeredGrid(math.zeros(batch(inflow_loc=4)), extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))

"""Create a gradient function with respect to velocity."""
sim_grad = field.gradient(simulate, wrt='velocity', get_output=False)

"""Evaluate the gradient by calling the gradient function."""
velocity_grad = sim_grad(initial_smoke, initial_velocity)

plt.close('all')
vis.plot(velocity_grad)
plt.show()

"""Run a few simple gradient-descent steps."""
print(f"Initial loss: {simulate(initial_smoke, initial_velocity)[0]}")
initial_velocity -= 0.01 * velocity_grad
print(f"Next loss: {simulate(initial_smoke, initial_velocity)[0]}")

sim_grad = field.gradient(simulate, wrt='velocity', get_output=True)

for opt_step in range(4):
  (loss, final_smoke, _v), velocity_grad = sim_grad(initial_smoke, initial_velocity)
  print(f"Step {opt_step}, loss: {loss}")
  initial_velocity -= 0.01 * velocity_grad

"""End of demo; see the PhiFlow docs for details."""


import firedrake
from firedrake import inner
import specfabpy.firedrake.CPO

nx, ny = 16, 16
mesh = firedrake.UnitSquareMesh(nx, ny, diagonal="crossed")
x = firedrake.SpatialCoordinate(mesh)
V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
expr = firedrake.as_vector((x[1], 0))
u = firedrake.Function(V).interpolate(expr)

cpo = specfabpy.firedrake.CPO.CPO(mesh, (1, 2, 3, 4), 6)
cpo.initialize()
for bc_id in (1, 2, 3, 4):
    cpo.set_isotropic_BCs(bc_id)

timestep = 0.01
final_time = 1.0
num_steps = int(final_time / timestep)
for step in range(num_steps):
    cpo.evolve(u, firedrake.as_tensor([[0, 1], [1, 0]]), timestep)

Q = firedrake.FunctionSpace(mesh, "CG", 1)
m = firedrake.Function(Q).interpolate(inner(cpo.w, cpo.w))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_aspect("equal")
colors = firedrake.tripcolor(m, axes=ax)
fig.colorbar(colors)
plt.show()

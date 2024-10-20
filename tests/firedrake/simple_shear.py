import firedrake
import specfabpy.firedrake

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
cpo.evolve(u, firedrake.as_tensor([[0, 1], [1, 0]]), 0.01)

from dolfinx import default_scalar_type, fem, log, mesh, plot
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
import pyvista
import ufl

# -------------------------------
# Define a mesh
# -------------------------------
lower_x, lower_y = 0.0, 0.0
upper_x, upper_y = 20.0, 1.0
nx, ny = 160, 8
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[lower_x, lower_y], [upper_x, upper_y]],
    [nx, ny],
    cell_type=mesh.CellType.triangle)

# -------------------------------
# Define a function space over the domain
# -------------------------------
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

# -------------------------------
# Hyperelastic Constitutive Model
# -------------------------------
E = 1e5
nu = 0.3
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

d = domain.geometry.dim
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))
J = ufl.det(F)
C = F.T * F
I1 = ufl.tr(C)

# Strain energy for compressible Neo-Hookean material
psi = (mu / 2) * (I1 - 3) - mu * ufl.ln(J) + (lmbda / 2) * ufl.ln(J)**2

# Stress
P = ufl.diff(psi, F)


# -------------------------------
# Mark locations where boundary conditions will be applied
# -------------------------------
def left(x): return np.isclose(x[0], lower_x)


def top(x): return np.isclose(x[1], upper_y)


fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)

marked_facets = np.hstack([left_facets, top_facets])
marked_values = np.hstack([np.full_like(left_facets, 1),
                          np.full_like(top_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets],
                          marked_values[sorted_facets])

# -------------------------------
# Dirichlet boundary conditions
# -------------------------------
u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

# -------------------------------
# Define Body force + traction vectors for Neumann boundary conditions
# Note: value of T is assigned during load stepping
# -------------------------------
B = fem.Constant(domain, default_scalar_type((0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0)))

# -------------------------------
# Weak form
# -------------------------------
metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag,
                 metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
F_form = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

# -------------------------------
# Solver details
# -------------------------------
problem = NonlinearProblem(F_form, u, bcs)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# -------------------------------
# Set up visualizer (optional)
# -------------------------------
pyvista.start_xvfb()
plotter = pyvista.Plotter(window_size=(800, 600), off_screen=True, 
                          notebook=False, shape=(1,1), title="2D FEniCSx Plot", 
                          lighting=None)
plotter.open_gif("intro_to_fenicsx.gif", fps=3)

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# Prepare 3D vectors for PyVista by appending a zero z-component
values = np.zeros((geometry.shape[0], 3))
u_vals = u.x.array.reshape(geometry.shape[0], -1)
values[:, :u_vals.shape[1]] = u_vals  # Typically, u_vals.shape[1] == 2
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Warp only in-plane (avoid any artificial out-of-plane deformation)
warped = function_grid.warp_by_vector("u", factor=1.0)
warped.points[:, 2] = 0.0  # Ensure all points lie in the 2D plane (z=0)

warped.set_active_vectors("u")
actor = plotter.add_mesh(warped, show_edges=False, lighting=False, clim=[0, 10])


Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), 
                    Vs.element.interpolation_points())
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array

# -------------------------------
# "Time" stepping loop to incrementally apply load
# -------------------------------
log.set_log_level(log.LogLevel.INFO)
tval0 = -1.0
for n in range(1, 10):
    T.value[1] = n * tval0
    num_its, converged = solver.solve(u)
    assert converged
    u.x.scatter_forward()
    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")
    function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    magnitude.interpolate(us)
    warped.set_active_scalars("mag")
    warped_n = function_grid.warp_by_vector(factor=1)
    warped.points[:, :] = warped_n.points
    warped.point_data["mag"][:] = magnitude.x.array
    plotter.update_scalar_bar_range([0, 12])
    plotter.view_xy()
    if n == 1:
        plotter.reset_camera()
        plotter.camera.Zoom(0.7)
        saved_camera = plotter.camera_position
    else:
        plotter.camera_position = saved_camera
        plotter.camera.Zoom(0.7)
    plotter.write_frame()

plotter.close()

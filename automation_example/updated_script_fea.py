from dolfinx import default_scalar_type, fem, log, mesh
from dolfinx.io import gmshio, VTKFile
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
import os
import sys
import ufl
import utils

# -------------------------------
# Command line arguments
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python updated_script_fea.py <config_path>")
    sys.exit(1)

config_path = sys.argv[1]
print(f"Using config file: {config_path}")

# -------------------------------
# Key simulation parameters
# -------------------------------
config = utils.load_simulation_config(config_path)

mesh_fname = os.path.join(config["files"]["mesh_dir"], config["files"]["mesh_file"])
output_fname = os.path.join(config["files"]["output_dir"], config["files"]["output_file"])

# geometric properties
length = config["geometry"]["length"]
width = config["geometry"]["width"]

# material properties
E = config["material"]["E"]
nu = config["material"]["nu"]

mesh_order = config["simulation"]["mesh_order"]
num_load_steps = config["simulation"]["num_load_steps"]
max_load = config["simulation"]["max_load"]

# -------------------------------
# Upload the mesh
# -------------------------------

domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_fname, comm=MPI.COMM_WORLD, gdim=2)

# -------------------------------
# Define function space for vector displacement
# -------------------------------
V = fem.functionspace(domain, ("Lagrange", mesh_order, (domain.geometry.dim,)))
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

# -------------------------------
# Material model (Neo-Hookean)
# -------------------------------
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

d = domain.geometry.dim
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))
J = ufl.det(F)
C = F.T * F
I1 = ufl.tr(C)
psi = (mu / 2) * (I1 - 3) - mu * ufl.ln(J) + (lmbda / 2) * ufl.ln(J)**2
P = ufl.diff(psi, F)


# -------------------------------
# Boundary conditions
# -------------------------------
def left(x): return np.isclose(x[0], 0.0, atol=1e-5)


def right(x): return np.isclose(x[0], length, atol=1e-5)


fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1),
                          np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets],
                          marked_values[sorted_facets])

u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

# -------------------------------
# Loads
# -------------------------------
B = fem.Constant(domain, default_scalar_type((0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0)))

# -------------------------------
# Variational form
# -------------------------------
metadata = {"quadrature_degree": 4}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
F_form = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

# -------------------------------
# Solver setup
# -------------------------------
problem = NonlinearProblem(F_form, u, bcs)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

vtkfile = VTKFile(domain.comm, output_fname, "w")
# -------------------------------
# "Time" stepping loop to incrementally apply load
# -------------------------------
log.set_log_level(log.LogLevel.INFO)
tval0 = max_load / num_load_steps
for n in range(1, num_load_steps + 1):
    T.value[0] = n * tval0
    num_its, converged = solver.solve(u)
    assert converged
    u.x.scatter_forward()
    # Directly write vector-valued displacement for visualization
    u.name = "Displacement"
    vtkfile.write_function(u, t=n)

# Numerical and symbolic tools
import numpy as np
import ufl

# MPI for parallelism
from mpi4py import MPI

# FEniCSx core
from dolfinx import mesh, fem, io, default_scalar_type

# PETSc backend (for solving systems)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)

from petsc4py import PETSc

# Required for post processing
from dolfinx.io import VTKFile
from pathlib import Path


# Create a unit square domain with quadrilateral elements
domain_size = 1.0
mesh_resolution = 64
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([domain_size, domain_size])],  # Domain corners
    [mesh_resolution, mesh_resolution],                            # Number of cells in each direction
    cell_type=mesh.CellType.quadrilateral                          # Use quadrilateral elements
)

# Create directory and file for output
output_dir = Path("./results")  # Customize the folder name
output_dir.mkdir(parents=True, exist_ok=True)
vtk_file_d = VTKFile(domain.comm, str(output_dir / "phase_field.pvd"), "w")
vtk_file_u = VTKFile(domain.comm, str(output_dir / "displacement.pvd"), "w")

# Function spaces
# V: scalar field for the phase field variable p
# W: vector field for the displacement u
# VV: discontinuous scalar field for the history field H
V = fem.functionspace(domain, ("Lagrange", 1,))
W = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
VV = fem.functionspace(domain, ("DG", 0,))

# Trial and test functions for variational forms
u, v = ufl.TrialFunction(W), ufl.TestFunction(W)  # Displacement
d, w = ufl.TrialFunction(V), ufl.TestFunction(V)  # Phase field

# FEM solution functions (updated every timestep)
u_new = fem.Function(W)     # Current displacement
u_old = fem.Function(W)     # Displacement from previous iteration
d_new = fem.Function(V)     # Current phase field
d_old = fem.Function(V)     # Phase field from previous iteration
H_old = fem.Function(VV)    # History field storing maximum tensile energy
H_init_ = fem.Function(V)   # Optional initial history field

# Material parameters (as UFL Constants so they can vary in space if needed)
E = fem.Constant(domain, 1e3)        # Young's modulus
nu = fem.Constant(domain, 0.3)       # Poisson's ratio
g_c = fem.Constant(domain, 1.0)      # Fracture toughness (critical energy release rate)
l_0 = fem.Constant(domain, 0.02)     # Regularization length scale (controls crack width)

# Derived material constants
mu = E / (2 * (1 + nu))                                   # Shear modulus
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))                # Lamé's first parameter

# Time stepping parameters
delta_t = 1e-3            # Time increment
num_steps = 100           # Total number of time steps

# Time variable used to define boundary loading over time
t = fem.Constant(domain, 0.0)

if domain.comm.rank == 0:
    print("Setup complete. Mesh and function spaces initialized.")


# --- Geometry dimensions ---
tdim = domain.topology.dim  # Topological dimension of mesh (2D here)
fdim = tdim - 1             # Facet dimension (1D edges in 2D)


# --- Define boundary detection functions ---
def top_boundary(x):
    return np.isclose(x[1], domain_size)


def bottom_boundary(x):
    return np.isclose(x[1], 0.0)


def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], domain_size)


# --- Locate boundary facets ---
top_facet = mesh.locate_entities_boundary(domain, fdim, top_boundary)
bot_facet = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
right_facet = mesh.locate_entities_boundary(domain, fdim, right_boundary)
left_facet = mesh.locate_entities_boundary(domain, fdim, left_boundary)

# --- Assign marker values for each boundary ---
top_marker = 1
bot_marker = 2
right_marker = 3
left_marker = 4

# --- Mark facets with integer tags ---
marked_facets = np.hstack([top_facet, bot_facet, right_facet, left_facet])
marked_values = np.hstack([
    np.full_like(top_facet, top_marker),
    np.full_like(bot_facet, bot_marker),
    np.full_like(right_facet, right_marker),
    np.full_like(left_facet, left_marker)
])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim,
                          marked_facets[sorted_facets],
                          marked_values[sorted_facets])

# --- Locate degrees of freedom on each boundary for u and d ---
top_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, top_facet)
top_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, top_facet)
top_phi_dofs = fem.locate_dofs_topological(V, fdim, top_facet)

bot_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, bot_facet)
bot_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, bot_facet)
bot_phi_dofs = fem.locate_dofs_topological(V, fdim, bot_facet)

right_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, right_facet)
right_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, right_facet)
right_phi_dofs = fem.locate_dofs_topological(V, fdim, right_facet)

left_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, left_facet)
left_y_dofs = fem.locate_dofs_topological(W.sub(1), fdim, left_facet)
left_phi_dofs = fem.locate_dofs_topological(V, fdim, left_facet)

# --- Define time-varying vertical displacement at the top boundary ---
u_bc_top = fem.Constant(domain, default_scalar_type(0.0))

# --- Define Dirichlet boundary conditions for displacement ---
bc_bot_y = fem.dirichletbc(default_scalar_type(0.0), bot_y_dofs, W.sub(1))  # Fix bottom in y
bc_left_x = fem.dirichletbc(default_scalar_type(0.0), left_x_dofs, W.sub(0))  # Fix left in x
bc_top_y = fem.dirichletbc(u_bc_top, top_y_dofs, W.sub(1))  # Pull top in y

# --- Combine into list for use in variational problem ---
bcs_u = [bc_bot_y, bc_left_x, bc_top_y]

if domain.comm.rank == 0:
    print("Boundary conditions and facet tags set up.")

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})

# --- Strain and stress definitions ---
def epsilon(u):
    """Small-strain tensor: symmetric gradient of displacement"""
    return ufl.sym(ufl.grad(u))

def sigma(u):
    """Linear elastic stress tensor (Hooke's law for isotropic materials)"""
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2.0 * mu * epsilon(u)

# --- Bracket functions for splitting tensile and compressive parts ---
def bracket_pos(x):
    """Positive part of a scalar quantity"""
    return 0.5 * (x + np.abs(x))

def bracket_neg(x):
    """Negative part of a scalar quantity"""
    return 0.5 * (x - np.abs(x))

# --- Spectral decomposition of strain tensor ---
# Used in Miehe model to separate tension and compression energy modes

A = ufl.variable(epsilon(u_new))        # Strain tensor (symbolic variable for differentiation)
I1 = ufl.tr(A)                           # First invariant (trace)
delta = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0] + 3.0e-16**2  # Avoid zero sqrt

# Eigenvalues (2D)
eigval_1 = (I1 - ufl.sqrt(delta)) / 2
eigval_2 = (I1 + ufl.sqrt(delta)) / 2

# Eigenvectors (differentiation wrt A)
eigvec_1 = ufl.diff(eigval_1, A).T
eigvec_2 = ufl.diff(eigval_2, A).T

# Positive and negative parts of the strain tensor (Miehe)
epsilon_p = 0.5 * (eigval_1 + abs(eigval_1)) * eigvec_1 + \
            0.5 * (eigval_2 + abs(eigval_2)) * eigvec_2
epsilon_n = 0.5 * (eigval_1 - abs(eigval_1)) * eigvec_1 + \
            0.5 * (eigval_2 - abs(eigval_2)) * eigvec_2


# --- Positive (tensile) and negative (compressive) energy densities ---
def psi_pos(u):
    """Tensile part of elastic energy (drives fracture)"""
    return 0.5 * lmbda * bracket_pos(ufl.tr(epsilon(u)))**2 + mu * ufl.inner(epsilon_p, epsilon_p)


def psi_neg(u):
    """Compressive part (does not contribute to fracture evolution)"""
    return 0.5 * lmbda * bracket_neg(ufl.tr(epsilon(u)))**2 + mu * ufl.inner(epsilon_n, epsilon_n)


# --- History field update ---
def H(u_new, H_old):
    """Update history variable with max positive energy so far (irreversibility)"""
    return ufl.conditional(ufl.gt(psi_pos(u_new), H_old), psi_pos(u_new), H_old)


# --- Define initial notch geometry ---
# Horizontal notch 1/4 domain wide, starting at right edge, halfway up
notch_start = np.array([domain_size, 0.5 * domain_size])                        # Right edge midpoint
notch_end = np.array([domain_size - 0.25 * domain_size, 0.5 * domain_size])     # 1/4 width to the left


# --- Distance from each mesh point to the notch segment ---
def point_line_distance(P, A, B):
    """Compute the distance from points P to line segment AB"""
    AB = B - A
    AP = P - A
    AB_norm_sq = np.dot(AB, AB)
    t = np.clip(np.einsum("ij,j->i", AP, AB) / AB_norm_sq, 0.0, 1.0)
    proj = A + np.outer(t, AB)
    return np.linalg.norm(P - proj, axis=1)


# Get mesh points (2D only)
points = domain.geometry.x[:, :2]
distances = point_line_distance(points, notch_start, notch_end)

# --- Convert distances to initial fracture energy field ---
phi_c = 0.999  # Controls sharpness of initial damage
width = float(l_0.value) / 2
H_array = np.zeros_like(distances)

mask = distances <= width
H_array[mask] = ((phi_c / (1 - phi_c)) * float(g_c.value) / (2 * float(l_0.value))) \
                * (1 - 2 * distances[mask] / float(l_0.value))

# Assign to initial function and interpolate into DG space
H_init_.x.array[:] = H_array
H_old.interpolate(H_init_)

if domain.comm.rank == 0:
    print("Initial notch crack initialized.")

# --- External traction load (optional, here set to zero) ---
T = fem.Constant(domain, default_scalar_type((0.0, 0.0)))  # External traction force (can be updated)

# --- Elasticity problem: displacement u ---
# Energy functional (weak form) for the displacement problem
# Degraded elasticity: (1 - d)^2 * sigma(u)
E_du = ((1.0 - d_new)**2) * ufl.inner(ufl.grad(v), sigma(u)) * ufl.dx + ufl.dot(T, v) * ufl.ds

# Left-hand side (Jacobian) and right-hand side (residual) of linear system
a_u = fem.form(ufl.lhs(E_du))
L_u = fem.form(ufl.rhs(E_du))

# Allocate PETSc matrix and vector for solving displacement
A_u = create_matrix(a_u)
b_u = create_vector(L_u)

# --- PETSc solver for displacement ---
solver_u = PETSc.KSP().create(domain.comm)
solver_u.setOperators(A_u)
solver_u.setType(PETSc.KSP.Type.GMRES)           # Iterative solver
solver_u.getPC().setType(PETSc.PC.Type.HYPRE)    # Multigrid preconditioner
solver_u.setTolerances(rtol=1e-8, max_it=1000)
solver_u.setFromOptions()

# --- Phase-field problem: d ---
# Variational form from phase-field fracture theory
# Minimizes gradient energy + damage energy weighted by the history field H
E_phi = (l_0**2 * ufl.dot(ufl.grad(d), ufl.grad(w)) +
         ((2 * l_0 / g_c) * H(u_new, H_old) + 1.0) * d * w) * ufl.dx \
        - (2 * l_0 / g_c) * H(u_new, H_old) * w * ufl.dx

# Left-hand and right-hand side
a_phi = fem.form(ufl.lhs(E_phi))
L_phi = fem.form(ufl.rhs(E_phi))

# PETSc matrix and vector for phase-field system
A_phi = create_matrix(a_phi)
b_phi = create_vector(L_phi)

# --- PETSc solver for phase-field ---
solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A_phi)
solver_phi.setType(PETSc.KSP.Type.GMRES)
solver_phi.getPC().setType(PETSc.PC.Type.HYPRE)
solver_phi.setTolerances(rtol=1e-8, max_it=1000)
solver_phi.setFromOptions()

# --- Error forms (for staggered convergence checks) ---
u_l2_error = fem.form(ufl.dot(u_new - u_old, u_new - u_old) * ufl.dx)
d_l2_error = fem.form(ufl.dot(d_new - d_old, d_new - d_old) * ufl.dx)

if domain.comm.rank == 0:
    print("Problem definitions and solvers initialized.")

# --- Lists to store reaction force vs. time for plotting ---
reaction_bot = []  # Bottom boundary vertical reaction force

# --- Residual of the displacement equation ---
# Used for virtual work calculation (reaction force via duality with virtual displacement)
residual = ufl.action(ufl.lhs(E_du), u_new) - ufl.rhs(E_du)

# --- Virtual test function for reaction force evaluation ---
v_reac = fem.Function(W)

# --- Form for computing virtual work over fixed DOFs ---
virtual_work_form = fem.form(ufl.action(residual, v_reac))

# --- Identify degrees of freedom for constrained boundaries ---
bot_dofs = fem.locate_dofs_geometrical(W, bottom_boundary)

# --- Set up virtual displacement with unit value in y at bottom (for computing reaction) ---
u_bc_bot = fem.Function(W)
bc_bot_rxn = fem.dirichletbc(u_bc_bot, bot_dofs)


# --- Interpolate unit vertical virtual displacement on bottom ---
def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values


u_bc_bot.sub(1).interpolate(one)

# --- Expression for updating history variable ---
# Ensures irreversibility: H = max(H, psi_pos)
H_expr = fem.Expression(
    ufl.conditional(ufl.gt(psi_pos(u_new), H_old), psi_pos(u_new), H_old),
    VV.element.interpolation_points()
)

if domain.comm.rank == 0:
    print("Reaction force setup complete.")


# --- Simulation parameters ---
error_tol = fem.Constant(domain, 1e-4)
error_total = fem.Constant(domain, 1.0)

# For storing reaction force data
reaction_bot = []

# Time-stepping loop
for i in range(num_steps + 1):
    staggered_iter = 0
    t.value = delta_t * i  # Update time
    u_bc_top.value = t.value  # Apply vertical loading at top
    error_total.value = 1.0

    # --- Staggered iteration: alternate solving u and p until convergence ---
    while error_total.value > error_tol.value:
        staggered_iter += 1

        # --- Displacement solve ---
        A_u.zeroEntries()
        fem.petsc.assemble_matrix(A_u, a_u, bcs=bcs_u)
        A_u.assemble()
        with b_u.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(b_u, L_u)
        fem.petsc.apply_lifting(b_u, [a_u], [bcs_u])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_u, bcs_u)
        solver_u.solve(b_u, u_new.x.petsc_vec)
        u_new.x.scatter_forward()

        # --- Phase field solve ---
        A_phi.zeroEntries()
        fem.petsc.assemble_matrix(A_phi, a_phi)
        A_phi.assemble()
        with b_phi.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(b_phi, L_phi)
        b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver_phi.solve(b_phi, d_new.x.petsc_vec)
        d_new.x.scatter_forward()

        # --- Compute L2 error for convergence check ---
        u_err = fem.assemble_scalar(u_l2_error)
        d_err = fem.assemble_scalar(d_l2_error)
        error_total.value = np.sqrt(domain.comm.allreduce(u_err, op=MPI.SUM)) + \
                            np.sqrt(domain.comm.allreduce(d_err, op=MPI.SUM))

        # --- Update fields for next staggered iteration ---
        u_old.x.array[:] = u_new.x.array
        d_old.x.array[:] = d_new.x.array
        H_old.interpolate(H_expr)

        if domain.comm.rank == 0:
            print(f"Step {i}, iter {staggered_iter}, error = {error_total.value:.3e}")
        
    # --- Save output info ---
    vtk_file_d.write_function(d_new, float(t.value))
    vtk_file_u.write_function(u_new, float(t.value))

    # --- Reaction force evaluation at bottom boundary ---
    v_reac.x.array[:] = 0.0
    v_reac.x.scatter_forward()
    fem.set_bc(v_reac.x.array, [bc_bot_rxn])
    R_bot = fem.assemble_scalar(virtual_work_form)
    total_R_bot = domain.comm.reduce(R_bot, op=MPI.SUM, root=0)

    if domain.comm.rank == 0:
        reaction_bot.append((float(t.value), total_R_bot))


# --- Close files from saving process
vtk_file_d.close()
vtk_file_u.close()

# --- Print final output to screen ---
if domain.comm.rank == 0:
    print("Simulation complete.")
    print("Final reaction forces (bottom):")
    for t_val, R in reaction_bot:
        print(f"  t = {t_val:.6f}, R = {R:.4e}")

# --- Save reaction force ---
if domain.comm.rank == 0:
    np.savetxt(output_dir / "reaction_bottom.txt", np.array(reaction_bot))

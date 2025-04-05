from dolfinx import mesh, fem
from mpi4py import MPI
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
# Try printing things out
# -------------------------------
print("psi -- type:", type(psi))
print("psi:", psi)
print("psi -- repr print:", repr(psi))

print("P -- type:", type(P))
print("P:", P)
print("P -- repr print:", repr(P))

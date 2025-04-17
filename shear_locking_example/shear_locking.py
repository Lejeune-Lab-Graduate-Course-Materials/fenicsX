from dolfinx import default_scalar_type, fem, log, mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import VTKFile
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pyvista as pv
from typing import Optional, List, Tuple
import ufl


######################################################################################
# simulation functions
######################################################################################

def run_simulation(nx, ny, element_order, is_quad_element, output_fname, E, nu, traction_val, H, L):
    # -------------------------------
    # Define a mesh
    # -------------------------------
    lower_x, lower_y = 0.0, 0.0
    upper_x, upper_y = L, H
    if is_quad_element:
        select_cell_type = mesh.CellType.quadrilateral
    else:
        select_cell_type = mesh.CellType.triangle
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[lower_x, lower_y], [upper_x, upper_y]],
        [nx, ny],
        cell_type=select_cell_type)

    # -------------------------------
    # Define a function space over the domain
    # -------------------------------
    V = fem.functionspace(domain, ("Lagrange", element_order, (domain.geometry.dim,)))
    v = ufl.TestFunction(V)
    u = fem.Function(V, name="Displacement")

    # -------------------------------
    # Hyperelastic Constitutive Model
    # -------------------------------
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
    if is_quad_element:
        if element_order == 2:
            metadata = {"quadrature_degree": 9}
        elif element_order == 1:
            metadata = {"quadrature_degree": 4}
    else:
        if element_order == 2:
            metadata = {"quadrature_degree": 4}
        elif element_order == 1:
            metadata = {"quadrature_degree": 1}
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
    # Save output as .pvd file
    # -------------------------------
    vtkfile = VTKFile(domain.comm, output_fname, "w")

    # -------------------------------
    # "Time" stepping loop to incrementally apply load
    # -------------------------------
    log.set_log_level(log.LogLevel.INFO)
    T.value[1] = traction_val
    num_its, converged = solver.solve(u)
    assert converged
    # save pvd file
    u.x.scatter_forward()
    # Directly write vector-valued displacement for visualization
    u.name = "Displacement"
    vtkfile.write_function(u, t=1)

    return converged


######################################################################################
# post-processing functions
######################################################################################
def get_centerline_displacement_from_pvd(
    pvd_path: str,
    centerline_points: List[Tuple[float, float, float]],
    time_index: int = -1
) -> List[Optional[Tuple[float, float, float]]]:
    """
    Evaluate the displacement field from a .pvd file at a list of (x, y, z) points.

    Parameters
    ----------
    pvd_path : str
        Path to the .pvd file containing a displacement time series.
    centerline_points : list of tuple[float, float, float]
        3D points (x, y, z), where z is often 0 in 2D problems.
    time_index : int, default=-1
        Time step to evaluate (default is last step).

    Returns
    -------
    list of tuple[float, float, float] or None
        List of displacement vectors at each centerline point.
        If a point falls outside the mesh, its entry is None.
    """
    # Load the time series mesh
    reader = pv.get_reader(pvd_path)
    reader.set_active_time_value(reader.time_values[time_index])
    mesh = reader.read()

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh[0]

    # Detect displacement field
    vector_field_name = next(
        (name for name in mesh.array_names
         if mesh[name].ndim == 2 and mesh[name].shape[1] in (2, 3)),
        None
    )
    if vector_field_name is None:
        raise ValueError(f"No vector field found in the .pvd file. Available fields: {mesh.array_names}")

    mesh.set_active_vectors(vector_field_name)

    # Convert points to Nx3 array
    points = np.array(centerline_points, dtype=np.float64)

    # Explicitly construct PolyData with one vertex per point
    # old syntax -- struggling with using older version of PyVista here
    # n_points = len(points)
    # verts = np.hstack([[1, i] for i in range(n_points)]).reshape(n_points, 2)
    # verts = np.array(verts, dtype=np.int64)

    # point_cloud = pv.PolyData(points, verts)
    point_cloud = pv.PolyData(points)
    n_points = len(points)
    point_cloud["id"] = np.arange(len(points))  # Dummy scalar to preserve all points

    # Sample displacement field at the given points
    result = point_cloud.interpolate(mesh)

    # Extract displacements
    displacements = []
    for i in range(n_points):
        if result[vector_field_name] is None or result[vector_field_name][i] is None:
            displacements.append(None)
        else:
            disp_vec = result[vector_field_name][i]
            displacements.append(tuple(float(x) for x in disp_vec))

    return displacements


case = 1

if case == 1:
    #  assigned simulation parameters
    E = 1e4
    nu = 0.3
    traction_val = -0.001
    H = 1.0
    L = 40.0
    #  computed analytical solution for comparison
    E_eff = E / (1 - nu**2)
    Izz = H ** 3 / 12
    w_analytical = traction_val * L ** 4 / (8.0 * E_eff * Izz)
    #  centerline points for assessing analytical solution
    centerline_points = []
    num_pts = 20
    x = np.linspace(0, L, num_pts)
    y = H / 2.0
    z = 0
    for kk in range(0, num_pts):
        centerline_points.append((x[kk], y, z))


ele_size_list = [[1, 40], [2, 80], [4, 160], [8, 320], [16, 640]]

ele_order_list = [1, 2]

is_quad_element_list = [True, False]

run_simulations = True
run_post_process = True

if run_simulations:
    for kk in range(0, len(ele_size_list)):
        ny = ele_size_list[kk][0]
        nx = ele_size_list[kk][1]
        for element_order in ele_order_list:
            for is_quad_element in is_quad_element_list:
                output_fname = "nx%i_ny%i_eo%i_isquad%i_case%i.pvd" % (nx, ny, element_order, is_quad_element, case)
                converged = run_simulation(nx, ny, element_order, is_quad_element, output_fname, E, nu, traction_val, H, L)
                print("-----------------")
                print(output_fname)
                print("completed!")


if run_post_process:
    all_centerlines = []
    for element_order in ele_order_list:
        for is_quad_element in is_quad_element_list:
            centerline_list = []
            for kk in range(0, len(ele_size_list)):
                ny = ele_size_list[kk][0]
                nx = ele_size_list[kk][1]
                output_fname = "nx%i_ny%i_eo%i_isquad%i_case%i.pvd" % (nx, ny, element_order, is_quad_element, case)
                centerline_disp = get_centerline_displacement_from_pvd(output_fname, centerline_points)
                centerline_list.append(centerline_disp)
            all_centerlines.append(centerline_list)
    
    # centerlines plot
    ix = 0
    title_list = ["Quad order 1 centerline", "Tri order 1 centerline", "Quad order 2 centerline", "Tri order 2 centerline"]
    save_list = ["Quad_order_1_centerline", "Tri_order_1_centerline", "Quad_order_2_centerline", "Tri_order_2_centerline"]
    for element_order in ele_order_list:
        for is_quad_element in is_quad_element_list:
            title = title_list[ix]
            centerline_list = all_centerlines[ix]
            plt.figure()
            cp = np.asarray(centerline_points)
            for kk in range(0, len(ele_size_list)):
                cd = np.asarray(centerline_list[kk])
                ny = ele_size_list[kk][0]
                nx = ele_size_list[kk][1]
                plt.plot(cp[:, 0] + cd[:, 0], cp[:, 1] + cd[:, 1], label="nx%i_ny%i" % (nx, ny))
            
            plt.legend()
            plt.title(title)
            plt.savefig(save_list[ix])
            ix += 1

    # comparison to w_analytical plot
    ix = 0
    title_list = ["Quad order 1 tip", "Tri order 1 tip", "Quad order 2 tip", "Tri order 2 tip"]
    save_list = ["Quad_order_1_tip", "Tri_order_1_tip", "Quad_order_2_tip", "Tri_order_2_tip"]
    for element_order in ele_order_list:
        for is_quad_element in is_quad_element_list:
            title = title_list[ix]
            centerline_list = all_centerlines[ix]
            tip_list = []
            num_ele_list = []
            for kk in range(0, len(ele_size_list)):
                tip_list.append(centerline_list[kk][-1][1])
                if is_quad_element:
                    num_ele_list.append(ele_size_list[kk][0] * ele_size_list[kk][1])
                else:
                    num_ele_list.append(ele_size_list[kk][0] * ele_size_list[kk][1] * 2)
            plt.figure()
            plt.plot(num_ele_list, tip_list, "k-", label="FEA solution")
            plt.plot(num_ele_list, [w_analytical] * len(ele_size_list), "r--", label="analytical solution")
            plt.title(title)
            plt.legend()
            plt.savefig(save_list[ix])
            ix += 1
            
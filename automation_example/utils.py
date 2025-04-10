import glob
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import solve as solve_linear
import os
import pyvista as pv
from typing import Optional, List, Tuple
import yaml


def build_and_save_simulation_config(
    length: float,
    width: float,
    amplitude: float,
    frequency: float,
    n_points: int,
    mesh_size: float,
    E: float,
    nu: float,
    mesh_order: int,
    num_load_steps: int,
    max_load: float,
    mesh_fname: str,
    mesh_dir: str,
    output_dir: str,
    output_fname: str,
    config_path: str,
):
    """
    Build and save a complete simulation configuration to a YAML file.

    Parameters
    ----------
    length : float
        Length of the rectangular domain (x-direction extent).
    width : float
        Width of the rectangular domain (y-direction extent).
    amplitude : float
        Amplitude of the sine wave applied to the centerline.
    frequency : float
        Frequency of the sine wave distortion.
    n_points : int
        Number of points to sample along the sine-distorted centerline.
    mesh_size : float
        Target mesh size used during mesh generation.
    E : float
        Young's modulus (elastic stiffness).
    nu : float
        Poisson's ratio.
    mesh_order : int
        Order of the mesh elements (e.g., 1 for linear, 2 for quadratic).
    num_load_steps : int
        Number of time/load steps in the simulation.
    max_load : float
        Maximum applied load in the simulation.
    mesh_fname : str
        Name of the mesh file (e.g., 'mesh.msh').
    mesh_dir : str
        Directory where the mesh file will be stored.
    output_dir : str
        Directory where output files (e.g., results, logs) will be stored.
    output_fname : str
        Output filename (e.g., 'displacement.pvd').
    config_path : str
        Full path where the YAML config file will be saved (e.g., 'inputs/config.yaml').
    """
    mesh_path = os.path.join(mesh_dir, mesh_fname)

    config = {
        "geometry": {
            "length": length,
            "width": width,
            "amplitude": amplitude,
            "frequency": frequency,
            "n_points": n_points,
            "mesh_size": mesh_size,
        },
        "material": {
            "E": E,
            "nu": nu,
        },
        "simulation": {
            "mesh_order": mesh_order,
            "num_load_steps": num_load_steps,
            "max_load": max_load,
        },
        "files": {
            "mesh_dir": mesh_dir,
            "mesh_file": mesh_fname,
            "mesh_path": mesh_path,
            "output_dir": output_dir,
            "output_file": output_fname,
        },
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Saved config to: {config_path}")


def load_simulation_config(config_path: str) -> dict:
    """
    Load a simulation configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Dictionary containing simulation parameters, matching the format
        from build_and_save_simulation_config().
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def mesh_size_to_filename_tag(mesh_size: float) -> str:
    """
    Convert a float mesh size (e.g., 0.05) to a safe filename tag like 'h0p05'.
    This avoids having decimal points in the file name.

    Parameters
    ----------
    mesh_size : float

    Returns
    -------
    str : filename-safe string like 'h0p05'
    """
    return f"h{mesh_size:.5f}".replace(".", "p")


def get_displacement_at_point(u, point, domain, tol=1e-10):
    """
    This code is really slow.
    Works on older dolfinx versions without geometry acceleration.
    With a more up to date dolfinx there are better methods for geometry.
    Recovering displacement from the pvd file is much faster.

    Parameters
    ----------
    u : dolfinx.fem.Function
        Displacement function.
    point : tuple[float, float]
        Point where displacement is evaluated.
    domain : dolfinx.mesh.Mesh
        Mesh corresponding to u.
    tol : float, optional
        Barycentric tolerance for point-in-triangle test.
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    Returns
    -------
    np.ndarray or None
        Displacement at the point, or None if not found.
    """
    geometry = domain.geometry.x
    dofmap = domain.geometry.dofmap
    num_cells = domain.topology.index_map(domain.topology.dim).size_local

    px, py, pz = point

    for cell_index in range(num_cells):
        node_ids = dofmap[cell_index]
        coords = geometry[node_ids]

        try:
            # Barycentric coordinate test
            A = np.array([
                [coords[0, 0], coords[1, 0], coords[2, 0]],
                [coords[0, 1], coords[1, 1], coords[2, 1]],
                [1.0,          1.0,          1.0]
            ])
            b = np.array([px, py, 1.0])
            bary = solve_linear(A, b)

            if np.all(bary >= -tol) and np.all(bary <= 1.0 + tol):
                return u.eval(point, cell_index)

        except LinAlgError:
            continue

    return None


def get_centerline_displacement(u, centerline_points, domain, tol=1e-10):
    """
    Evaluate the displacement field `u` at each point along a given centerline.

    Parameters
    ----------
    u : dolfinx.fem.Function
        The displacement function to evaluate (vector-valued).
    centerline_points : list[tuple[float, float]]
        A list of (x, y) points along the centerline where displacement is sampled.
    domain : dolfinx.mesh.Mesh
        The mesh associated with the function `u`.
    tol : float, optional
        Tolerance used for determining whether a point is inside a cell via barycentric coordinates.

    Returns
    -------
    list[tuple[float, float]]
        A list of displacement vectors (u_x, u_y) at each centerline point.
        If a point is not found inside the mesh, its value will be `None`.
    """
    centerline_displacement = []
    for pt in centerline_points:
        disp = get_displacement_at_point(u, pt, domain, tol=tol)
        if disp is not None:
            centerline_displacement.append(tuple(disp))
        else:
            centerline_displacement.append(None)
    return centerline_displacement


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


def generate_and_save_linear_mesh(
    outline_points: list[tuple[float, float]],
    mesh_name: str,
    mesh_size: float = 0.05,
    msh_filename: str = "mesh.msh",
):
    """
    Generate a 2D linear triangle mesh from a closed outline and save it to a .msh file
    for loading in FEniCSx. Uses explicit line closure to avoid Gmsh curve loop bugs.

    Parameters
    ----------
    outline_points : list of (float, float)
        Polygon points defining the closed shape in XY.
    mesh_name : str
        A name for the Gmsh model.
    mesh_size : float
        Target mesh size at the outline points.
    msh_filename : str
        Output path for the Gmsh .msh file (must end in .msh).
    """
    assert msh_filename.endswith(".msh"), "Output mesh file must end with .msh"

    gmsh.initialize()
    gmsh.model.add(mesh_name)

    # --- Fix for curve loop bug: manually ensure closure ---
    if outline_points[0] != outline_points[-1]:
        outline_points.append(outline_points[0])

    # Create gmsh points (skip duplicated last point when creating points)
    point_tags = []
    for kk in range(len(outline_points) - 1):
        x, y = outline_points[kk]
        pt_tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size)
        point_tags.append(pt_tag)

    gmsh.model.geo.synchronize()

    # Create lines (manually close the loop)
    curve_tags = []
    for i in range(len(point_tags) - 1):
        start_pt = point_tags[i]
        end_pt = point_tags[i + 1]
        line_tag = gmsh.model.geo.addLine(start_pt, end_pt)
        curve_tags.append(line_tag)

    # Close the loop with one final line
    start_pt = point_tags[-1]
    end_pt = point_tags[0]
    line_tag = gmsh.model.geo.addLine(start_pt, end_pt)
    curve_tags.append(line_tag)

    gmsh.model.geo.synchronize()

    # Create surface
    loop = gmsh.model.geo.addCurveLoop(curve_tags)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    
    # Add physical groups (useful for subdomains/BCs in FEniCSx)
    gmsh.model.addPhysicalGroup(2, [surface])
    for tag in curve_tags:
        gmsh.model.addPhysicalGroup(1, [tag])

    gmsh.model.geo.synchronize()

    # Mesh settings: linear mesh
    gmsh.model.mesh.setOrder(1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # make sure 2D
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    # Save mesh to file
    gmsh.write(msh_filename)
    gmsh.finalize()

    print(f"Linear mesh saved to: {os.path.abspath(msh_filename)}")
    return


def generate_sine_distorted_rectangle(
    length: float = 20.0,
    width: float = 1.0,
    amplitude: float = 1.0,
    frequency: float = 2 * np.pi,
    n_points: int = 100,
) -> list[tuple[float, float, float]]:
    """
    Generate outline points for a distorted rectangle with sinusoidal top and bottom edges.

    Parameters
    ----------
    length : float
        Length of the domain along the x-axis.
    width : float
        Vertical distance between the bottom and top sine waves.
    amplitude : float
        Amplitude of the sine distortion applied to top and bottom edges.
    frequency : float
        Frequency (angular) of the sine wave along the length.
    n_points : int
        Number of sample points per curve (top and bottom).

    Returns
    -------
    outline_points : list of (float, float)
        List of 2D points forming the closed outline.
    """
    x = np.linspace(0, length, n_points)

    # Bottom sine edge: y = -amplitude * sin(...) + y0
    y_bottom = amplitude * np.sin(frequency * x) + 0.0
    bottom_edge = list(zip(x, y_bottom))

    # Top sine edge: y = +amplitude * sin(...) + y0
    y_top = amplitude * np.sin(frequency * x) + width
    top_edge = list(zip(x[::-1], y_top[::-1]))  # reverse to close loop

    # Combine edges bottom → right side → top → left side
    outline = bottom_edge + top_edge + [bottom_edge[0]]  # close the loop

    return outline


def generate_sine_distorted_rectangle_centerline(
    length: float = 20.0,
    width: float = 1.0,
    amplitude: float = 1.0,
    frequency: float = 2 * np.pi,
    n_points_sample: int = 25,
) -> list[tuple[float, float]]:
    """
    Generate a centerline with a sinusoidal distortion along the length of a rectangle.

    The function returns a list of (x, y) points representing the centerline of a
    rectangle with a sine wave applied in the vertical direction. The sine wave is
    defined by its amplitude and frequency, and the centerline is offset vertically
    to lie within the rectangle's height.

    Parameters
    ----------
    length : float, default=20.0
        Length of the rectangle (extent in the x-direction).
    width : float, default=1.0
        Total width of the rectangle; the sine wave is centered vertically at width / 2.
    amplitude : float, default=1.0
        Amplitude of the sine distortion applied to the centerline.
    frequency : float, default=2 * np.pi
        Frequency of the sine wave (radians per unit length).
    n_points_sample : int, default=25
        Number of sample points to generate along the centerline.

    Returns
    -------
    list of tuple[float, float]
        List of (x, y) coordinates defining the sine-distorted centerline.
    """
    x_sample = np.linspace(0, length, n_points_sample)
    y_sample = amplitude * np.sin(frequency * x_sample) + width / 2.0
    z_sample = [0] * x_sample.shape[0]
    centerline_points = list(zip(x_sample, y_sample, z_sample))  # false z dimension 
    return centerline_points


def plot_outline(fname, outline):
    """
    Plot and save a 2D outline as a closed curve.

    Parameters
    ----------
    fname : str
        Path to save the figure (e.g., 'results/outline.png').
    outline : list[tuple[float, float]]
        List of (x, y) points defining the outline (should be ordered and closed).
    """
    outline_arr = np.array(outline)

    plt.figure(figsize=(10, 2))
    plt.plot(outline_arr[:, 0], outline_arr[:, 1], 'k-o', markersize=4)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Visualization of Outline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return


def plot_deformed_centerlines(centerline_points, directory=".", fname=None):
    """
    Plot and optionally save the deformed centerlines over time from saved displacement files.

    Parameters
    ----------
    centerline_points : list[tuple[float, float]]
        Original centerline point coordinates as (x, y).
    directory : str, default="."
        Directory containing `centerline_*.npy` files.
    fname : str or None, default=None
        If provided, the plot will be saved to this filepath (e.g., 'results/centerlines.png').
        If None, the plot will not be shown or saved (silent mode).
    """
    # Load saved displacement files
    files = sorted(glob.glob(os.path.join(directory, "centerline_*.npy")))
    if not files:
        raise FileNotFoundError("No centerline_*.npy files found in directory.")

    centerline_array = np.array(centerline_points)
    centerline_array = centerline_array[:, 0:2]  # neglect the all 0s 3rd dimension

    plt.figure(figsize=(8, 4))
    for f in files:
        displacements = np.load(f, allow_pickle=True)
        disp_array = np.array([
            disp if disp is not None else (0.0, 0.0)
            for disp in displacements
        ])

        # Compute new positions
        new_positions = centerline_array + disp_array

        # Label by time step (from filename)
        step = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        plt.plot(
            new_positions[:, 0],
            new_positions[:, 1],
            marker="o",
            label=f"Step {step}"
        )

    plt.xlabel("x (deformed)")
    plt.ylabel("y (deformed)")
    plt.title("Deformed Centerlines Over Time")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        print(f"Plot saved to {fname}")
    plt.close()
    return


def generate_displacement_gif(
    vtu_pattern="results/displacement_*.vtu",
    gif_path="results/displacement_postproc.gif",
    warp_factor=1.0,
    clim=(0, 12),
    fps=3
):
    """
    Generate an animated GIF of displacement from a series of .vtu files.

    Parameters
    ----------
    vtu_pattern : str
        Glob pattern to find .vtu files (default: 'results/displacement_*.vtu').
    gif_path : str
        Path where the output GIF will be saved.
    warp_factor : float
        Factor by which to scale displacement for mesh warping.
    clim : tuple
        Color limits for displacement magnitude visualization.
    fps : int
        Frames per second for the GIF.
    """
    pv.start_xvfb()
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    files = sorted(glob.glob(vtu_pattern))
    if not files:
        raise FileNotFoundError(f"No VTU files found matching pattern: {vtu_pattern}")
    print(f"Found {len(files)} files.")

    plotter = pv.Plotter(off_screen=True, window_size=(800, 600), title="Displacement Viewer")
    plotter.open_gif(gif_path, fps=fps)

    saved_camera = None

    for i, file in enumerate(files):
        mesh = pv.read(file)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh[0]

        print(f"Reading {file} | Available arrays: {mesh.array_names}")

        # Detect the vector field
        vector_name = next(
            (name for name in mesh.array_names
             if mesh[name].ndim == 2 and mesh[name].shape[1] in (2, 3)),
            None
        )
        if vector_name is None:
            raise ValueError(f"No vector field found in {file}")

        disp = mesh[vector_name]
        padded_disp = np.zeros((disp.shape[0], 3))
        padded_disp[:, :disp.shape[1]] = disp
        mesh["Displacement3D"] = padded_disp
        mesh.set_active_vectors("Displacement3D")

        # Warp and compute magnitude
        warped = mesh.warp_by_vector("Displacement3D", factor=warp_factor)
        warped.points[:, 2] = 0.0
        mag = np.linalg.norm(padded_disp, axis=1)
        warped["mag"] = mag
        warped.set_active_scalars("mag")

        # Render frame
        plotter.clear()
        plotter.add_mesh(warped, scalars="mag", cmap="viridis", clim=clim,
                         show_edges=False, lighting=False)
        plotter.view_xy()

        if i == 0:
            plotter.reset_camera()
            plotter.camera.zoom(0.7)
            saved_camera = plotter.camera_position
        else:
            plotter.camera_position = saved_camera
            plotter.camera.zoom(0.7)

        plotter.write_frame()

    plotter.close()
    print(f"GIF saved to {gif_path}")
    return

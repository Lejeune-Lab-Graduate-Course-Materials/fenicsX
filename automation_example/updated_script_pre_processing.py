import os
import numpy as np
import utils
import sys


if len(sys.argv) < 5:
    print("Usage: python updated_script_pre_processing.py <mesh_dir> <config_dir> <output_dir> <mesh_sizes_string>")
    sys.exit(1)

# -------------------------------------
# Define output directories
# -------------------------------------
mesh_dir = sys.argv[1]       # Where to save all generated mesh files
print(f"Using mesh directory: {mesh_dir}")
config_dir = sys.argv[2]     # Where to save YAML config files
print(f"Using configuration directory: {config_dir}")
output_dir = sys.argv[3]    # Where to save simulation output (e.g., .pvd files)
print(f"Using output directory: {output_dir}")

# Create directories if they don't already exist
for path in [mesh_dir, config_dir, output_dir]:
    os.makedirs(path, exist_ok=True)

mesh_sizes_string = sys.argv[4]
mesh_size_all = [float(x) for x in mesh_sizes_string.split(',')]
print("Parsed mesh sizes:", mesh_size_all)

# -------------------------------------
# Define geometry and material parameters
# -------------------------------------
length = 20.0
width = 1.0
amplitude = 1.0
frequency = 10 * np.pi / 20  # One full sine wave over the domain
E = 1e5                      # Young's modulus
nu = 0.3                     # Poisson's ratio
n_points = 100             # Number of horizontal points for outline sampling
mesh_name = "rectangular_mesh"  # Name for Gmsh internal model
mesh_order = 1
num_load_steps = 10
max_load = 5000

# -------------------------------------
# Generate distorted rectangular outline (sine-wave centerline)
# -------------------------------------
outline_points = utils.generate_sine_distorted_rectangle(
    length,
    width,
    amplitude,
    frequency,
    n_points,
)

# -------------------------------------
# Loop over a list of mesh sizes to create separate inputs
# -------------------------------------

for mesh_size in mesh_size_all:
    print("mesh size", mesh_size, "started.")

    # Generate a clean filename tag like h0p01 from the mesh size
    tag = utils.mesh_size_to_filename_tag(mesh_size)

    # Define mesh filename and full path
    mesh_fname = f"{tag}.msh"
    mesh_fname_full = mesh_dir + "/" + mesh_fname

    # Define output file and config file path
    output_fname = f"displacement_{tag}.pvd"
    config_path = config_dir + f"/{tag}.yaml"

    # -------------------------------------
    # Save YAML config with simulation parameters
    # -------------------------------------
    utils.build_and_save_simulation_config(
        length=length,
        width=width,
        amplitude=amplitude,
        frequency=frequency,
        n_points=n_points,
        mesh_size=mesh_size,
        E=E,
        nu=nu,
        mesh_order=mesh_order,
        num_load_steps=num_load_steps,
        max_load=max_load,
        mesh_fname=mesh_fname,
        mesh_dir=mesh_dir,
        output_dir=output_dir,
        output_fname=output_fname,
        config_path=config_path
    )

    # -------------------------------------
    # Generate and save the 2D mesh using Gmsh
    # -------------------------------------
    utils.generate_and_save_linear_mesh(
        outline_points=outline_points,
        mesh_name=mesh_name,
        mesh_size=mesh_size,
        msh_filename=mesh_fname_full
    )

    print("mesh size", mesh_size, "completed!")

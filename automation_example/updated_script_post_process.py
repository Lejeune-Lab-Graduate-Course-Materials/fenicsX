import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import utils


if len(sys.argv) < 5:
    print("Usage: python update_script_post_process.py <config_dir> <output_dir> <n_points_sample> <mesh_sizes_string>")
    sys.exit(1)

# -------------------------------------
# Define output directories
# -------------------------------------
config_dir = sys.argv[1]     # Where to save YAML config files
output_dir = sys.argv[2]    # Where to save simulation output (e.g., .pvd files)
n_points_sample = int(sys.argv[3])

mesh_sizes_string = sys.argv[4]
mesh_size_all = [float(x) for x in mesh_sizes_string.split(',')]
print("Parsed mesh sizes:", mesh_size_all)

centerline_disp_all = []

for mesh_size in mesh_size_all:
    # Generate a clean filename tag like h0p01 from the mesh size
    tag = utils.mesh_size_to_filename_tag(mesh_size)

    # Define output file and config file path
    output_fname = os.path.join(output_dir, f"displacement_{tag}.pvd")
    config_path = config_dir + f"/{tag}.yaml"

    # load the configuration file and extract key info
    config = utils.load_simulation_config(config_path)

    # geometric properties
    length = config["geometry"]["length"]
    width = config["geometry"]["width"]
    amplitude = config["geometry"]["amplitude"]
    frequency = config["geometry"]["frequency"]

    # simulation info
    num_load_steps = config["simulation"]["num_load_steps"]
    max_load = config["simulation"]["max_load"]

    # extract the centerline displacements for mesh refinement
    centerline_points = utils.generate_sine_distorted_rectangle_centerline(
        length,
        width,
        amplitude,
        frequency,
        n_points_sample,
    )

    # compute the centerline displacement at the end of the simulation
    displacements = utils.get_centerline_displacement_from_pvd(output_fname, centerline_points, num_load_steps - 1)
    centerline_disp_all.append(displacements)


save_file_name = output_dir + "/centerline_displacements.npy"
np.save(save_file_name, centerline_disp_all)

# plot the centerline displacements
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

for kk in range(0, len(mesh_size_all)):
    mesh_size = mesh_size_all[kk]
    displacements = centerline_disp_all[kk]
    # this relies on centerlines being the same for all meshes -- this should be true
    locs = np.array(centerline_points) + np.array(displacements)
    # Left subplot (equal aspect)
    ax1.plot(locs[:, 0], locs[:, 1], ".-", label=f"mesh size {mesh_size:.4f}")
    # Right subplot (no forced aspect)
    ax2.plot(locs[:, 0], locs[:, 1], ".-", label=f"mesh size {mesh_size:.4f}")


# Subplot 1 settings: axis ratio = equal
ax1.grid(True)
ax1.set_aspect("equal", adjustable="datalim")
ax1.set_xlabel("x-position")
ax1.set_ylabel("y-position")
ax1.set_title("Centerline (aspect = 'equal')")
ax1.legend()

# Subplot 2 settings: default aspect
ax2.grid(True)
ax2.set_xlabel("x-position")
ax2.set_ylabel("y-position")
ax2.set_title("Centerline (zoomed in)")
ax2.legend()

# Zoom in around the middle
ax2.set_xlim(10.0, 15.0)
ax2.set_ylim(0.0, 2.0)

plt.tight_layout()
save_plot_name = os.path.join(output_dir, "centerline_displacements_visualize.png")
plt.savefig(save_plot_name, dpi=500)
plt.close()

# plot the error in centerline_points with respect to the finest mesh size
finest_disp = np.array(centerline_disp_all[-1])

error_all = []
for kk in range(0, len(mesh_size_all) - 1):
    vec = np.array(centerline_disp_all[kk])[:, 0:2] - finest_disp[:, 0:2]  # ignore z
    error = np.mean(np.abs(vec))
    error_all.append(error)

plt.figure()
plt.plot(mesh_size_all[0:-1], error_all, ".-")   
plt.grid(True)
plt.xlabel("mesh size")
plt.ylabel("mean absolute error")
plt.title("Centerline displacement error wrt finest mesh")
save_plot_name_err = os.path.join(output_dir, "centerline_displacements_error.png")
plt.savefig(save_plot_name_err, dpi=150)
plt.close()


aa = 44
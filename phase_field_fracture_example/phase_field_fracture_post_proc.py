import glob
import pyvista as pv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def generate_phase_field_gif(
    vtu_pattern="results/phase_field_p*_*.vtu",
    gif_path="results/phase_field_postproc.gif",
    clim=(0.0, 1.0),
    fps=5,
    cmap="viridis"
):
    """
    Generate an animated GIF from a series of parallel .vtu files (written by MPI ranks).

    Parameters
    ----------
    vtu_pattern : str
        Glob pattern to find all per-rank .vtu files.
    gif_path : str
        Path where the output GIF will be saved.
    clim : tuple
        Color limits for visualization.
    fps : int
        Frames per second for the GIF.
    cmap : str
        Colormap to use for scalar field.
    """
    pv.start_xvfb()
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    all_files = sorted(glob.glob(vtu_pattern))
    if not all_files:
        raise FileNotFoundError(f"No VTU files found matching pattern: {vtu_pattern}")

    # Group files by timestep (based on suffix)
    timestep_groups = defaultdict(list)
    for f in all_files:
        suffix = f.split("_p")[-1].split("_")[-1]  # e.g., 00000.vtu
        timestep_groups[suffix].append(f)

    sorted_timesteps = sorted(timestep_groups.keys())
    print(f"Found {len(sorted_timesteps)} timesteps")

    plotter = pv.Plotter(off_screen=True, window_size=(800, 600), title="Phase Field Viewer")
    plotter.open_gif(gif_path, fps=fps)

    saved_camera = None

    for i, suffix in enumerate(sorted_timesteps):
        group_files = timestep_groups[suffix]
        group_files.sort()  # ensure consistent processor order

        blocks = pv.MultiBlock()
        for f in group_files:
            mesh_part = pv.read(f)
            if isinstance(mesh_part, pv.MultiBlock):
                mesh_part = mesh_part[0]
            blocks.append(mesh_part)

        combined = blocks.combine()

        print(f"Timestep {i}: merged {len(group_files)} files")

        # Detect scalar field
        scalar_name = next(
            (name for name in combined.array_names
             if combined[name].ndim == 1),
            None
        )
        if scalar_name is None:
            raise ValueError(f"No scalar field found in files for timestep {suffix}")

        combined.set_active_scalars(scalar_name)

        plotter.clear()
        plotter.add_mesh(combined, scalars=scalar_name, cmap=cmap, clim=clim,
                         show_edges=False, lighting=False)
        plotter.view_xy()

        if i == 0:
            plotter.reset_camera()
            plotter.camera.zoom(0.9)
            saved_camera = plotter.camera_position
        else:
            plotter.camera_position = saved_camera
            plotter.camera.zoom(0.9)

        plotter.write_frame()

    plotter.close()
    print(f"GIF saved to {gif_path}")


def plot_force_disp(filename_txt, output_png):
    """
    Plot force-displacement curve and save it.
    """
    data = np.loadtxt(filename_txt)
    time = data[:, 0]
    force = np.abs(data[:, 1]) * 1e-3

    plt.figure()
    plt.plot(time, force, '-o', markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Reaction Force (kN)")
    plt.title("Force–Displacement Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Saved force-displacement plot to: {output_png}")


if __name__ == "__main__":
    # --- Run post-processing ---
    generate_phase_field_gif(
            vtu_pattern="results/phase_field_*.vtu",
            gif_path="results/phase_field.gif",
            clim=(0, 1),
            fps=5
        )

    filename_txt = "results/reaction_bottom.txt"
    output_png = "results/force_disp.png"
    plot_force_disp(filename_txt, output_png)
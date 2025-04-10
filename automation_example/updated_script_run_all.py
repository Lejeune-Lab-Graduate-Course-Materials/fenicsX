import os
import utils
import subprocess

# -----------------------
# What to run
# -----------------------
create_meshes_and_config_files = True
run_simulations = True
run_post_processing_mesh_refinement = True
run_post_processing_visualization = True

# -----------------------
# Pre-processing
# -----------------------
study_tag = "study_1"
mesh_dir = study_tag + "_meshes"       # Where to save all generated mesh files
config_dir = study_tag + "_inputs"     # Where to save YAML config files
output_dir = study_tag + "_results"    # Where to save simulation output (e.g., .pvd files)
mesh_sizes_string = "1.0,0.5,0.1,0.05,0.025"  # note that this string cannot have any spaces

# Note geometry and material parameters are all defined within the pre_processing script

if create_meshes_and_config_files:
    completed_process = subprocess.run(
            [
                "python", "updated_script_pre_processing.py",
                mesh_dir,
                config_dir,
                output_dir,
                mesh_sizes_string
            ],
            capture_output=True,
            text=True
        )

    # Check for errors
    if completed_process.returncode != 0:
        print("updated_script_pre_processing.py failed!")
        print("stderr:", completed_process.stderr)
    else:
        print("updated_script_pre_processing.py ran successfully!")
        print("stdout:", completed_process.stdout)


# -----------------------
# Run simulations
# -----------------------
if run_simulations:
    mesh_size_all = [float(x) for x in mesh_sizes_string.split(',')]
    for mesh_size in mesh_size_all:
        tag = utils.mesh_size_to_filename_tag(mesh_size)
        config_path = config_dir + f"/{tag}.yaml"

        completed_process = subprocess.run(
                [
                    "python", "updated_script_fea.py",
                    config_path
                ],
                capture_output=True,
                text=True
            )

        # Check for errors
        if completed_process.returncode != 0:
            print("updated_script_fea.py failed!")
            print("stderr:", completed_process.stderr)
        else:
            print("updated_script_fea.py ran successfully!")
            print("stdout:", completed_process.stdout)

# -----------------------
# Post processing, mesh refinement
# -----------------------
n_points_sample = str(100)  # sample 100 points along the domain centerline to compute error

if run_post_processing_mesh_refinement:
    mesh_size_all = [float(x) for x in mesh_sizes_string.split(',')]
    completed_process = subprocess.run(
            [
                "python", "updated_script_post_process.py",
                config_dir,
                output_dir,
                n_points_sample,
                mesh_sizes_string
            ],
            capture_output=True,
            text=True
        )

    # Check for errors
    if completed_process.returncode != 0:
        print("updated_script_post_process.py failed!")
        print("stderr:", completed_process.stderr)
    else:
        print("updated_script_post_process.py ran successfully!")
        print("stdout:", completed_process.stdout)


# -----------------------
# Post processing, visualization
# -----------------------
if run_post_processing_visualization:
    mesh_size_all = [float(x) for x in mesh_sizes_string.split(',')]
    for mesh_size in mesh_size_all:
        tag = utils.mesh_size_to_filename_tag(mesh_size)
        config_path = config_dir + f"/{tag}.yaml"
        config = utils.load_simulation_config(config_path)
        output_fname = os.path.join(config["files"]["output_dir"], config["files"]["output_file"])
        vtu_pattern = output_fname[:-4] + "*.vtu"
        gif_path = config["files"]["output_dir"] + "/" + tag + "_displacement.gif"
        utils.generate_displacement_gif(vtu_pattern, gif_path)
        print("mesh ", tag, " gif displacement completed")

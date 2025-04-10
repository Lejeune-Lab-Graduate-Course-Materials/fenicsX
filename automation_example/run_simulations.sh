#!/bin/bash -l

# Set SCC project
#$ -P me700

# Submit an array job with 5 tasks (this will change depending on how many tasks you have)
#$ -t 1-5

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=12:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N fea_sims


# -------------------------------
# Load conda/mamba and activate environment
# -------------------------------
module load miniconda

# If the environment is already created, just activate it:
source activate fenicsx-env

# Alternatively, if you want to *ensure creation and install once*,
# comment the `source activate` line above and uncomment the block below
# (only run once interactively, not per job)

# mamba create -n fenicsx-env -y
# source activate fenicsx-env
# mamba install -y -c conda-forge fenics-dolfinx mpich pyvista
# pip install imageio gmsh PyYAML

# -------------------------------
# Run the appropriate task input
# -------------------------------

# Get all .yaml input files into an array (sorted to ensure consistent order)
mapfile -t inputs < <(ls study_1_inputs/h*.yaml | sort)

taskinput=${inputs[$(($SGE_TASK_ID - 1))]}

# Print info for debugging/logging
echo "Running simulation for input file: $taskinput"

# Run your Python simulation
python updated_script_fea.py "$taskinput"

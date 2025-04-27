#!/bin/bash -l

# Set SCC project
#$ -P me700

# Request 16 total MPI tasks using the available PE
# Change to this if you want to use just 4: #$ -pe mpi_4_tasks_per_node 4
#$ -pe mpi_16_tasks_per_node 16

# Set working directory to where the job was submitted
#$ -cwd

# Set name of the job and files
#$ -N fenicsx_phasefield
#$ -o fenicsx_phasefield_$JOB_ID.out
#$ -e fenicsx_phasefield_$JOB_ID.err

# Request 12 hours of wall time
#$ -l h_rt=12:00:00

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

# Create log/output directories if they don't exist
mkdir -p logs
mkdir -p results

# Run FEniCSx simulation using MPI across $NSLOTS cores
mpirun -np $NSLOTS python phase_field_fracture.py

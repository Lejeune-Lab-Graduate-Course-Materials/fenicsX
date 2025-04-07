# Download FEniCSx

Note: I strongly recommend working on the SCC

Create a conda environment -- 

See: https://fenicsproject.org/download/

On the SCC:

```bash
module load miniconda
mamba create create -n fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
```

Note: this might take a couple of minutes. 

Then, you can launch a VSCode server and choose fenicsx-env a your conda environment.

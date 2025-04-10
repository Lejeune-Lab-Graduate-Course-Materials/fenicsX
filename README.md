# Download FEniCSx

Note: additional installs since last time!

Note: I strongly recommend working on the SCC

Create a conda environment -- 

See: https://fenicsproject.org/download/

On the SCC:

```bash
module load miniconda
mamba create -n fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
pip install PyYAML
```

Note: this might take a couple of minutes. 

Then, you can launch a VSCode server and choose fenicsx-env a your conda environment.

## Automation example

1. Run ``script.py``
2. Consider, how would you do a mesh refinement study with this script?
3. See ``updated_script_*.py`` for one example of how to do this
4. See ``run_simulations.sh`` for a bash script example to run an array job of mulitple simulations at once. This script is run with the command:
```bash
qsub run_simulations.sh
```
Consider looking into useful commands for interacting with jobs -- e.g.,
```bash
qstat -u <username>
```
For more examples so you can tailor a script specific to your needs, see: https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/

Note that for the job scheduler, it's not so good to request a bunch of tiny jobs. As a general rule (for this class at least), please think of 10 min - 12 hours as the target time for jobs. You can use the bash script to run subsets of array jobs in series as well.

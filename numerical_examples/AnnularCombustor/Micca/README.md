# Annular Test Cases

In this directory, we present test cases of Section 3.2 in the paper. There are two test cases; one for full annular computation (`fullAnnulus` directory) and the other for application of Bloch boundary condition (`bloch` directory). You want to check and inspect `runAll.sh` to see example commands. Here is the one for `fullAnnulus`;

```
# generate mesh
python3 generateMesh.py -nopopup
python3 params.py -nopopup

# test running in serial
python3 -u active_fpi.py -nopopup |tee Results/Active/FPI/active.log
python3 -u active_newton.py -nopopup |tee Results/Active/NewtonSolver/active.log

# test running in parallel
mpirun -np 8 python3 -u active_fpi.py -nopopup |tee Results/Active/FPI/activeParallel.log
mpirun -np 8 python3 -u active_newton.py -nopopup |tee Results/Active/NewtonSolver/activeParallel.log

# test different mode calculations
mpirun -np 8 python3 -u active_modes.py -nopopup -target 1000 |tee Results/Active/Modes/Parallel1000.log
mpirun -np 8 python3 -u active_modes.py -nopopup -target 2000 |tee Results/Active/Modes/Parallel2000.log
mpirun -np 8 python3 -u active_modes.py -nopopup -target 5000 |tee Results/Active/Modes/Parallel5000.log
mpirun -np 8 python3 -u active_modes.py -nopopup -target 9000 |tee Results/Active/Modes/Parallel9000.log
mpirun -np 8 python3 -u active_modes.py -nopopup -target 10000 |tee Results/Active/Modes/Parallel10000.log
mpirun -np 8 python3 -u active_modes.py -nopopup -target 11000 |tee Results/Active/Modes/Parallel11000.log
```

We perform tests for fixed point iteration and Newton solver in separate scripts. As you can see, we first start generating the mesh for the test, then run serial and parallel computations for active flame problems. To compute different eigenmodes (as presented in the paper) we target specific eigenvalues to capture nearest eigenmodes.

It is important to note that parallelization is not presented for the cases where Bloch boundary conditions applied, because parallellization of DOF matching requires further effort, which is not presented in this paper.  

Do not forget that you should always be in the helmholtz-x environment of docker container or python virtual environment, if you install FEniCSx from source. 
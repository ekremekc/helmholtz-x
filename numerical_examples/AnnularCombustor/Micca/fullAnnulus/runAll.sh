# generate mesh
python3 generateMesh.py -nopopup

# test running in serial
python3 -u active_fpi.py -nopopup |tee Results/Active/FPI/active.log
python3 -u active_newton.py -nopopup |tee Results/Active/NewtonSolver/active.log

# test running in parallel
mpirun -np 8 python3 -u active_fpi.py -nopopup |tee Results/Active/FPI/activeParallel.log
mpirun -np 8 python3 -u active_newton.py -nopopup |tee Results/Active/NewtonSolver/activeParallel.log
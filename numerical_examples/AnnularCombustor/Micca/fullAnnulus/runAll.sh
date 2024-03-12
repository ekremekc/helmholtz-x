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

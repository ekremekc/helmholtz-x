# test running in serial
python3 params.py -nopopup
python3 -u active.py -nopopup |tee Results/Active/active.log

# test running in parallel
mpirun -np 4 python3 -u active.py -nopopup |tee Results/Active/activeParallel.log
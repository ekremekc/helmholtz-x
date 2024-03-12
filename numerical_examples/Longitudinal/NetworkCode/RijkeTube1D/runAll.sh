# test running in serial
python3 params.py -nopopup
python3 -u passive.py -nopopup |tee Results/Passive/passive.log
python3 -u active.py -nopopup |tee Results/Active/active.log
python3 -u active_adj.py -nopopup |tee Results/Active/active_adj.log

# test running in parallel
mpirun -np 4 python3 -u passive.py -nopopup |tee Results/Passive/passiveParallel.log
mpirun -np 4 python3 -u active.py -nopopup |tee Results/Active/activeParallel.log
# test running in serial
python3 params.py -nopopup
python3 -u passive.py -nopopup > Results/Passive/passive.log
python3 -u active.py -nopopup > Results/Active/active.log

# test running in parallel
mpirun -np 4 python3 -u passive.py -nopopup > Results/Passive/passiveParallel.log
mpirun -np 4 python3 -u active.py -nopopup > Results/Active/activeParallel.log
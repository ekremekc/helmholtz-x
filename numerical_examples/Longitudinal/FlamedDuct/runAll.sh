# generate mesh
python3 generateMesh.py -nopopup

# test running in serial
python3 params.py -nopopup
python3 -u passive.py -nopopup |tee Results/Passive/passive.log
python3 -u active.py -nopopup |tee Results/Active/active.log

# test running in parallel
mpirun -np 6 python3 -u passive.py -nopopup |tee Results/Passive/passiveParallel.log
mpirun -np 6 python3 -u active.py -nopopup |tee Results/Active/activeParallel.log
# generate mesh
python3 generateMesh.py -nopopup

# test running in serial
python3 -u active.py -nopopup > Results/Active/active.log

# test running in parallel
mpirun -np 8 python3 -u active.py -nopopup > Results/Active/activeParallel.log
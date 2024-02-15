# generate mesh
python3 generateMesh.py -nopopup

# test running in serial
python3 -u passive.py -nopopup > Results/Passive/passive.log
python3 -u active.py -nopopup > Results/Active/active.log
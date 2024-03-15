# generate mesh
python3 -u generateMesh.py -nopopup

# testing input parameters
python3 -u params.py -nopopup
python3 -u generateDisplacementField.py -nopopup

# testing the eigenmode computation
python3 -u main.py -nopopup |tee Results/Original/results.log

# testing the shape derivatives computation
python3 -u main_shape.py -nopopup |tee Results/ShapeDerivatives/results.log

# generate optimized mesh
python3 -u generateOptimizedMesh.py -nopopup

# testing the optimized shape
mpirun -np 8 python3 -u main_opt.py -nopopup |tee Results/Optimized/results.log
# Shape Optimization Example with FFD

In this directory, we present application case of Section 4 in the paper. You can inspect `runAll.sh` to see the flow of the commands for shape optimization using adjoints. A `runAll.sh` includes the pipeline for shape optimization procedure;

```
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
```

The script we run to compute shape derivatives (`main_shape.py`) requires massive effort to parallize, so we avoid to parallel execution to remove complexities for this paper. 

Do not forget that you should always be in the helmholtz-x environment of docker container or python virtual environment, if you install FEniCSx from source. 
# Longitudinal Test Cases

In this directory, we present test cases of Section 3.1 (`NetworkCode` directory) and Appendix C (`PRF` directory) in the paper. You can navigate the test case you want to check and inspect `runAll.sh` to see example commands. A typical `runAll.sh` should consist both serial and parallel;

```
# generate mesh
python3 generateMesh.py -nopopup

# test running in serial
python3 params.py -nopopup
python3 -u passive.py -nopopup |tee Results/Passive/passive.log
python3 -u active.py -nopopup |tee Results/Active/active.log

# test running in parallel
mpirun -np 8 python3 -u passive.py -nopopup |tee Results/Passive/passiveParallel.log
mpirun -np 8 python3 -u active.py -nopopup |tee Results/Active/activeParallel.log
```
As you can see we first start generating the mesh for the test, then run serial and parallel computations for both passive and active flame problems. Do not forget that you should always be in the helmholtz-x environment of docker container or python virtual environment, if you install FEniCSx from source. 
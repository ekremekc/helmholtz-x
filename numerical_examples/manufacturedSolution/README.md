# Purely Acoustic Test Cases with Purely Resistive or Reactive Impendances

In this directory, we present test cases of Appendix B.1 (`manufacturedSolutions` directory) in the paper. We present the analytical eigenfrequencies calculated by MATLAB's `fsolve` function in `matlab_data` folder (As of today, there is equivalent of fsolve in python packages, which can solve complex numbered nonlinear dispersion relation.) We then run the same test cases with `helmholtz-x` to verify numerical eigenfrequencies. We run:
```
python3 manufacturedHelmholtz.py
```
to get numerical eigenfrequencies.
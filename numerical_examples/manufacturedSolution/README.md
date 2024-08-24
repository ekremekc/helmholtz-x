# Purely Acoustic Test Cases with Purely Resistive or Reactive Impendances

In this directory, we present test cases of Appendix B.1 in the paper (`manufacturedSolutions` directory in the repository). We present the analytical eigenfrequencies calculated by MATLAB's `fsolve` function in `matlab_data` folder (As of today, there is no equivalent of MATLAB's `fsolve` in python packages, which can solve complex numbered nonlinear dispersion relation.) We then run the same test cases with `helmholtz-x` to verify numerical eigenfrequencies. We run:
```
python3 manufacturedHelmholtz.py
```
to get numerical eigenfrequencies.
# specfab documentation

![](https://raw.githubusercontent.com/nicholasmr/specfab/main/images/logo.jpg){: style="width:200px"}

Spectral CPO model of polycrystalline materials that:

- Can model lattice rotation, discontinuous DRX, and rotation/continuous DRX.
- Can calculate CPO-induced viscous anisotropies using Sachs/Taylor homogenizations.
- Can calculate elastic P- and S-wave velocities using Voigt/Reuss homogenizations.
- Contains expressions for forward+inverse orthotropic and transversely isotropic rheologies.
- Can convert between structure tensors and spectral expansions coefficients.
- Can be integrated with finite-element models such as Elmer and FEniCS.

By Nicholas M. Rathmann and David A. Lilien

## Glacier ice demo

![](https://github.com/nicholasmr/specfab/raw/main/demo/cube-crush-animation/cube-crush.gif){: style="width:500px"}

## Install

Source code [available here](https://github.com/nicholasmr/specfab)

| Enviroment | How to |
| :--- | :--- |
| Python | A pre-compiled module exists for Linux:<br>`pip3 install numpy --upgrade && pip3 install specfabpy` |
| Compile Python module |- For a local-only install, run `make specfabpy` in `/src` (requires LAPACK and BLAS) <br>- To install for general use (in other folders), run `make python` in `/src`. Note that if you do not have write permissions for your python installation, you can instead run `make specfabpy; python setup.py install --user`|
| Fortran | The Fortran module is built by running `make specfab.o` |
| Elmer/Ice Interface | To interface with Elmer/Ice, you need a shared version of the libraries (built with the same Fortran compiler as Elmer). If needed, change the compiler in `src/Makefile`, then run `make libspecfab.so` |

## Documentation 

| Component | Reference |
| :--- | :--- |
| Lattice rotation | [Rathmann et al. (2021)](https://doi.org/10.1017/jog.2020.117) |
| Discontinuous dynamic recrystallization | [Rathmann and Lilien (2021)](https://doi.org/10.1017/jog.2021.88) |
| Orthotropic bulk rheologies | [Rathmann and Lilien (2022)](https://doi.org/10.1017/jog.2022.33) |
| Elastic wave velocities | [Rathmann et al. (2022)](https://doi.org/10.1098/rspa.2022.0574) |

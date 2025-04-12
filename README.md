# Minimal Essential Matrix Solver

This project demonstrates a synthetic setup for estimating the **essential matrix** and the **relative camera pose** (rotation and translation) using a **minimal 5-point algorithm** based on  ["An Efficient Solution to the Five-Point Relative Pose Problem"](https://ieeexplore.ieee.org/document/1288525).

## Features

- Estimation of essential matrix using 5-point correspondences.
- Selection of the best essential matrix based on Sampson error over all point correspondences.

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/)
- C++17 or above
- [Armadillo](https://arma.sourceforge.net/)
- A CMake-compatible build system

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./run_solver
```

## Output

The program prints:
- Ground truth rotation and translation
- Estimated rotation and translation
- Best essential matrix based on Sampson error
- Total pose error (combines rotation and translation, not an ideal way)

## File Overview

- `main.cpp`: Main pipeline for generating data, running the solver, and evaluating results.
- `utility.h/cpp`: Helper functions for projection, normalization, nullspace computation, RREF with partial pivoting, and error computation.
- `minimal_solver.h/cpp`: Implementation of the 5-point essential matrix solver.

## Credits
- More than the paper, the OCTAVE implementation here - (https://github.com/SergioRAgostinho/five_point_algorithm) - helped me understand the theory better. The cpp code presented here is inspired from the OCTAVE implementation. 
- And, of course, GPT 😊 

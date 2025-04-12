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

```
R (gt) = 
   0.999564   0.0111672   0.0273475
-0.00934735     0.99779  -0.0657911
 -0.0280218   0.0655068    0.997459
t (gt) =         -0.998439 -0.0017627 -0.0558187
Picking the Essential Matrix that yields the minimum absolute Sampson distance.
E (best) = 
-0.000472363    0.0555799  -0.00543059
  -0.0837724    0.0647812     0.994375
   0.0110947    -0.996213    0.0657366
R (best) = 
   0.999564   0.0111672   0.0273475
-0.00934735     0.99779  -0.0657911
 -0.0280218   0.0655068    0.997459
t (best) =       -0.998439 -0.0017627 -0.0558187
Pose error: 1.20742e-06
```
## File Overview

- `main.cpp`: Main pipeline for generating data, running the solver, and evaluating results.
- `utility.h/cpp`: Helper functions for projection, normalization, nullspace computation, RREF with partial pivoting, and error computation.
- `minimal_solver.h/cpp`: Implementation of the 5-point essential matrix solver.

## Credits
- More than the paper, the OCTAVE implementation here - (https://github.com/SergioRAgostinho/five_point_algorithm) - helped me understand the theory better. The cpp code presented here is inspired from the OCTAVE implementation. 
- And, of course, GPT 😊 

# Steepest Descent Optimization

This repository contains an implementation of steepest descent optimization method using Python and the Autograd library. Steepest descent is a numerical optimization technique used to find the minimum of a function by iteratively adjusting the parameters in the direction of the steepest descent.

## Project Objetive
The aim of this project is to demonstrate the effectiveness of steepest descent optimization in dynamically computing optimal points without the need for manual intervention.

## Key objectives of the project include:
1. Implementing the steepest descent optimization method in Python.
2. Dynamically computing optimal points by defining the objective function and the initial point.
3. Computing the gradient and Hessian matrix of the objective function using Autograd.
4. Calculating constants 𝑐 and 𝑘 for the standard form of the quadratic function.
5. Finding critical points using the Hessian matrix.

## Objective Function

The objective function used in this implementation is defined in the `f(x)` function within the `steepest_descent.py` file. You can modify this function and the intial point to define your own.

## Dependencies
The implementation of steepest descent optimization using Python and the Autograd library has the following dependencies:

* Python: The programming language used for implementation.
* Autograd: A Python library for automatic differentiation, used for computing gradients and Hessians of the objective function.
* NumPy: A fundamental package for scientific computing with Python, used for numerical operations and linear algebra computations.

You can install these dependencies using the following commands:
`pip install autograd numpy`
Ensure that you have Python installed on your system before proceeding with the installation of the dependencies.

## Getting Started

To use this implementation, follow these steps:

1. Clone the repository: https://github.com/Ragheed7/steepest-descent-optimization.git
2. Navigate to the repository directory: cd steepest-descent-optimization
3. Run the optimization script: `steepest_descent.py`

## Results

After running the optimization script, you will see the results printed to the console, including the optimal point, minimum value of the objective function, gradient at the initial point, gradient length, learning rate, and whether the function is convex or not.

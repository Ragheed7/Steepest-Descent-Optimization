import autograd.numpy as np
from autograd import grad, hessian

def f(x):
    return (2*x[0] - 4*x[1]) ** 2 + (x[1] + 6) ** 2 - 1.75 * (x[0] - 3*x[1]) * (x[1] + 8)
    # Two more functions:
    # (x[0] + 1) ** 2 + (x[1] + 6) ** 2 - 1.75 * (x[0] - 2) * (x[1] + 4)
    # (x[0] - 4) ** 2 + (x[1] + 5) ** 2 + 1.8 * (x[0] - 4) * (x[1] - 5)

# Compute the gradient of the objective function
gradient_f = grad(f)

def calculate_learning_rate(grad, hessian):
    learning_rate = np.dot(grad, grad) / np.dot(grad, np.dot(hessian, grad))
    return learning_rate

def steepest_descent(initial_point):
    gradient = gradient_f(initial_point)
    hessian_func = hessian(f)
    hessian_matrix = hessian_func(initial_point)

    learning_rate = calculate_learning_rate(gradient, hessian_matrix)

    # Steepest Descent Optimization Method
    x_next = initial_point - learning_rate * gradient

    # Check function convexity
    is_convex = np.all(np.linalg.eigvals(hessian_matrix) >= 0)

    return x_next, f(x_next), learning_rate, gradient, hessian_matrix, is_convex

# Initial point
initial_point = np.array([5, 7], dtype=float)

# Perform steepest descent optimization
optimal_point, min_value, learning_rate, gradient, hessian, is_convex = steepest_descent(initial_point)

# Calculate c and k
c = gradient_f(np.zeros_like(initial_point))
k = f(np.zeros_like(initial_point))

# Calculate the length of the gradient
gradient_length = np.linalg.norm(gradient)

# Critical point using the Hessian Matrix
critical_point = -np.linalg.inv(hessian).dot(c)

print("Constant k:", k)
print("Linear part c:", c)
print("Hessian matrix:")
print(hessian)
if is_convex:
    print("The function is convex.")
else:
    print("The function is not convex.")
print("Critical point using the Hessian:", critical_point)
print("Gradient at initial point:", gradient)
print("Gradient length:", gradient_length)
print("Learning rate:", learning_rate)
print("Optimal point using Steepest Descent:", optimal_point)
print("Minimum value:", min_value)

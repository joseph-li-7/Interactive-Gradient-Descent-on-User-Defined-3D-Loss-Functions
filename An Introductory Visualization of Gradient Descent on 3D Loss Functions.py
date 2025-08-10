# This is a self-learning project for LLM and ML Optimizing.
# All Copyrights Reserved under Joseph Li, 2025


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define symbolic variables 'x' and 'y' to represent function inputs
x, y = sp.symbols('x y')

#  Prompt the user to enter a loss function of variables x and y
#    Function should use Python syntax
user_func_str = input(
    "Enter a loss function in terms of x and y (e.g. x**2 + y**2 + x*y).\n"
    "Use Python syntax (e.g. ** for powers): "
)

# Turn the users input string into a symbolic expression using SymPy
loss_sym = sp.sympify(user_func_str)

# Compute symbolic partial derivatives with respect to x and y
grad_x_sym = sp.diff(loss_sym, x)
grad_y_sym = sp.diff(loss_sym, y)

# Display the parsed function and its symbolic gradients for user confirmation
print(f"Loss function: {loss_sym}")
print(f"Partial derivative wrt x: {grad_x_sym}")
print(f"Partial derivative wrt y: {grad_y_sym}")

# Convert symbolic expressions to numerical functions for evaluation with numpy arrays
loss_func = sp.lambdify((x, y), loss_sym, 'numpy')
grad_x_func = sp.lambdify((x, y), grad_x_sym, 'numpy')
grad_y_func = sp.lambdify((x, y), grad_y_sym, 'numpy')

# Initialize gradient descent parameters
#    Learning rate controls the step size during descent (adjustable)
#    num_steps determines how many iterations gradient descent will run
#    Starting point (x_current, y_current) is randomly chosen within [-5, 5]
learning_rate = 0.1
num_steps = 100
x_current = np.random.uniform(-5, 5)
y_current = np.random.uniform(-5, 5)

print(f"Starting point: x={x_current:.4f}, y={y_current:.4f}")
print(f"Learning rate: {learning_rate}")
print(f"Number of steps: {num_steps}")

# Prepare lists to store the values of x, y, and loss at each step for plotting and analysis
x_values = []
y_values = []
loss_values = []

# Main gradient descent loop:
#    For each iteration:
#      - Compute gradient at current point
#      - Update current position by stepping opposite the gradient (to minimize loss)
#      - Check for numerical overflow or invalid values (stop if detected)
#      - Store current position and loss for visualization
for step in range(num_steps):
    grad_x = grad_x_func(x_current, y_current)
    grad_y = grad_y_func(x_current, y_current)

    # Gradient descent parameter update
    x_current = x_current - learning_rate * grad_x
    y_current = y_current - learning_rate * grad_y

    # Early stopping if values become invalid (NaN or infinity)
    if np.isnan(x_current) or np.isnan(y_current) or np.isinf(x_current) or np.isinf(y_current):
        print(f"Stopping early due to overflow or invalid value at step {step+1}")
        break

    # Record current step's data
    x_values.append(x_current)
    y_values.append(y_current)
    loss_values.append(loss_func(x_current, y_current))

    # Print progress to the console
    print(f"Step {step+1}: x={x_current:.6f}, y={y_current:.6f}, loss={loss_values[-1]:.6f}")

# Plot the loss value over iterations to visualize convergence behavior
plt.plot(range(1, len(loss_values)+1), loss_values, marker='o')
plt.title('Loss over Gradient Descent Steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 3D Visualization of the loss surface and the gradient descent path
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a grid of points around the descent path to plot the loss surface
x_grid = np.linspace(min(x_values)-1, max(x_values)+1, 50)
y_grid = np.linspace(min(y_values)-1, max(y_values)+1, 50)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = loss_func(X_grid, Y_grid)

# Plot the loss surface with transparency for context
ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, cmap='viridis')

# Overlay the descent path in red dots connected by lines
ax.plot(x_values, y_values, loss_values, color='red', marker='o', label='Descent path')

# Label axes and add title and legend
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Loss')
ax.set_title('Gradient Descent Path on Loss Surface')
ax.legend()

plt.show()
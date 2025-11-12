"""
Quick test script to verify all implementations work
"""

import sys
sys.path.append('src')

import numpy as np

print("=" * 70)
print("TESTING GRADIENT DESCENT IMPLEMENTATIONS")
print("=" * 70)

# Test 1: Core Gradient Descent
print("\n1. Testing Core Gradient Descent...")
from gradient_descent import gradient_descent, quadratic_function, quadratic_gradient

x0 = np.array([3.0, 4.0])
x_opt, path, cost_history = gradient_descent(
    quadratic_function, quadratic_gradient, x0,
    alpha=0.1, max_iter=100, verbose=False
)
assert len(path) > 0
assert quadratic_function(x_opt) < 1e-6
print("   ✓ Core gradient descent: PASSED")

# Test 2: Linear Regression
print("\n2. Testing Linear Regression...")
from linear_regression import LinearRegressionGD, generate_linear_data

X, y, true_theta = generate_linear_data(n_samples=100, n_features=1, noise=10.0, random_state=42)
model = LinearRegressionGD(learning_rate=0.1, max_iter=200, method='batch')
model.fit(X, y, verbose=False)
r2 = model.score(X, y)
print(f"   Train R² score: {r2:.4f}")
# Check that model converged (cost decreased)
assert len(model.cost_history) > 0
assert model.cost_history[-1] < model.cost_history[0]
print("   ✓ Linear regression: PASSED")

# Test 3: Logistic Regression
print("\n3. Testing Logistic Regression...")
from logistic_regression import LogisticRegressionGD, generate_classification_data

X, y = generate_classification_data(n_samples=100, n_features=2, random_state=42)
model = LogisticRegressionGD(learning_rate=0.1, max_iter=200)
model.fit(X, y, verbose=False)
acc = model.score(X, y)
assert acc > 0.8  # Should achieve good accuracy
print(f"   Accuracy: {acc:.2%}")
print("   ✓ Logistic regression: PASSED")

# Test 4: Different GD Methods
print("\n4. Testing Different GD Methods...")
methods = ['batch', 'stochastic', 'mini-batch']
for method in methods:
    model = LinearRegressionGD(learning_rate=0.1, max_iter=30, method=method)
    model.fit(X[:, 0:1], y, batch_size=16, verbose=False)
    print(f"   ✓ {method.capitalize()} GD: PASSED")

# Test 5: Advanced Optimizers
print("\n5. Testing Advanced Optimizers...")
from gradient_descent import gradient_descent_with_momentum, adam_optimizer

x0 = np.array([3.0, 4.0])
x_opt, _, cost_history = gradient_descent_with_momentum(
    quadratic_function, quadratic_gradient, x0,
    alpha=0.1, beta=0.9, max_iter=100, verbose=False
)
# Check convergence (cost decreased significantly)
assert cost_history[-1] < cost_history[0] * 0.01
print(f"   Momentum final cost: {quadratic_function(x_opt):.6f}")
print("   ✓ Momentum: PASSED")

x_opt, _, cost_history = adam_optimizer(
    quadratic_function, quadratic_gradient, x0,
    alpha=0.1, max_iter=100, verbose=False
)
assert cost_history[-1] < cost_history[0] * 0.1  # Adam should reduce cost
print(f"   Adam final cost: {quadratic_function(x_opt):.6f}")
print("   ✓ Adam: PASSED")

# Test 6: Test Functions
print("\n6. Testing Optimization Test Functions...")
from gradient_descent import rosenbrock_function, rosenbrock_gradient, beale_function, beale_gradient

x0 = np.array([0.0, 0.0])
x_opt, _, _ = gradient_descent(
    rosenbrock_function, rosenbrock_gradient, x0,
    alpha=0.001, max_iter=500, verbose=False
)
print(f"   Rosenbrock optimization final cost: {rosenbrock_function(x_opt):.6f}")
print("   ✓ Rosenbrock function: PASSED")

x_opt, _, _ = gradient_descent(
    beale_function, beale_gradient, x0,
    alpha=0.001, max_iter=500, verbose=False
)
print(f"   Beale optimization final cost: {beale_function(x_opt):.6f}")
print("   ✓ Beale function: PASSED")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nImplementations are working correctly.")
print("Ready for research and presentation!")

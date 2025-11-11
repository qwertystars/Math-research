"""
Core Gradient Descent Implementation
Provides various gradient descent algorithms from scratch using NumPy
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict


def gradient_descent(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Basic Gradient Descent (Batch Gradient Descent)

    Parameters:
    -----------
    f : callable
        The objective function to minimize
    grad_f : callable
        The gradient of the objective function
    x0 : np.ndarray
        Initial point (starting position)
    alpha : float
        Learning rate (step size)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance (stops when gradient norm < tol)
    verbose : bool
        If True, print progress

    Returns:
    --------
    x_opt : np.ndarray
        The optimized point
    path : list of np.ndarray
        The optimization path (all points visited)
    cost_history : list of float
        The cost function value at each iteration
    """
    x = x0.copy()
    path = [x.copy()]
    cost_history = [f(x)]

    for i in range(max_iter):
        # Compute gradient
        grad = grad_f(x)

        # Update rule: x_new = x_old - alpha * gradient
        x = x - alpha * grad

        # Store path and cost
        path.append(x.copy())
        cost_history.append(f(x))

        # Check convergence
        if np.linalg.norm(grad) < tol:
            if verbose:
                print(f"Converged in {i+1} iterations")
            break

        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: f(x) = {f(x):.6f}, ||grad|| = {np.linalg.norm(grad):.6f}")

    return x, path, cost_history


def stochastic_gradient_descent(
    f: Callable,
    grad_f_single: Callable,
    x0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Stochastic Gradient Descent (SGD)
    Updates parameters using one sample at a time

    Parameters:
    -----------
    f : callable
        The objective function to minimize
    grad_f_single : callable
        Gradient function for a single sample: grad_f_single(x, xi, yi)
    x0 : np.ndarray
        Initial parameters
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    alpha : float
        Learning rate
    max_iter : int
        Maximum number of epochs
    tol : float
        Convergence tolerance
    verbose : bool
        If True, print progress

    Returns:
    --------
    x_opt : np.ndarray
        The optimized parameters
    path : list of np.ndarray
        The optimization path
    cost_history : list of float
        The cost function value at each epoch
    """
    x = x0.copy()
    path = [x.copy()]
    cost_history = [f(x)]
    n_samples = X.shape[0]

    for epoch in range(max_iter):
        # Shuffle data for each epoch
        indices = np.random.permutation(n_samples)

        for i in indices:
            # Compute gradient for single sample
            grad = grad_f_single(x, X[i], y[i])

            # Update parameters
            x = x - alpha * grad

        # Store path and cost after each epoch
        path.append(x.copy())
        current_cost = f(x)
        cost_history.append(current_cost)

        # Check convergence (based on cost change)
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
            if verbose:
                print(f"Converged in {epoch+1} epochs")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: f(x) = {current_cost:.6f}")

    return x, path, cost_history


def mini_batch_gradient_descent(
    f: Callable,
    grad_f_batch: Callable,
    x0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    batch_size: int = 32,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Mini-batch Gradient Descent
    Updates parameters using small batches of data

    Parameters:
    -----------
    f : callable
        The objective function to minimize
    grad_f_batch : callable
        Gradient function for a batch: grad_f_batch(x, X_batch, y_batch)
    x0 : np.ndarray
        Initial parameters
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    alpha : float
        Learning rate
    batch_size : int
        Size of mini-batches
    max_iter : int
        Maximum number of epochs
    tol : float
        Convergence tolerance
    verbose : bool
        If True, print progress

    Returns:
    --------
    x_opt : np.ndarray
        The optimized parameters
    path : list of np.ndarray
        The optimization path
    cost_history : list of float
        The cost function value at each epoch
    """
    x = x0.copy()
    path = [x.copy()]
    cost_history = [f(x)]
    n_samples = X.shape[0]

    for epoch in range(max_iter):
        # Shuffle data for each epoch
        indices = np.random.permutation(n_samples)

        # Process mini-batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Compute gradient for mini-batch
            grad = grad_f_batch(x, X[batch_indices], y[batch_indices])

            # Update parameters
            x = x - alpha * grad

        # Store path and cost after each epoch
        path.append(x.copy())
        current_cost = f(x)
        cost_history.append(current_cost)

        # Check convergence
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
            if verbose:
                print(f"Converged in {epoch+1} epochs")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: f(x) = {current_cost:.6f}")

    return x, path, cost_history


def gradient_descent_with_momentum(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Gradient Descent with Momentum
    Accelerates convergence by adding a momentum term

    Parameters:
    -----------
    f : callable
        The objective function to minimize
    grad_f : callable
        The gradient of the objective function
    x0 : np.ndarray
        Initial point
    alpha : float
        Learning rate
    beta : float
        Momentum coefficient (typically 0.9)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        If True, print progress

    Returns:
    --------
    x_opt : np.ndarray
        The optimized point
    path : list of np.ndarray
        The optimization path
    cost_history : list of float
        The cost function value at each iteration
    """
    x = x0.copy()
    v = np.zeros_like(x)  # Velocity (momentum)
    path = [x.copy()]
    cost_history = [f(x)]

    for i in range(max_iter):
        # Compute gradient
        grad = grad_f(x)

        # Update velocity with momentum
        v = beta * v + alpha * grad

        # Update position
        x = x - v

        # Store path and cost
        path.append(x.copy())
        cost_history.append(f(x))

        # Check convergence
        if np.linalg.norm(grad) < tol:
            if verbose:
                print(f"Converged in {i+1} iterations")
            break

        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: f(x) = {f(x):.6f}, ||grad|| = {np.linalg.norm(grad):.6f}")

    return x, path, cost_history


def adam_optimizer(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    alpha: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Adam Optimizer (Adaptive Moment Estimation)
    Combines momentum and RMSprop

    Parameters:
    -----------
    f : callable
        The objective function to minimize
    grad_f : callable
        The gradient of the objective function
    x0 : np.ndarray
        Initial point
    alpha : float
        Learning rate
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    epsilon : float
        Small constant for numerical stability
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        If True, print progress

    Returns:
    --------
    x_opt : np.ndarray
        The optimized point
    path : list of np.ndarray
        The optimization path
    cost_history : list of float
        The cost function value at each iteration
    """
    x = x0.copy()
    m = np.zeros_like(x)  # First moment (mean)
    v = np.zeros_like(x)  # Second moment (variance)
    path = [x.copy()]
    cost_history = [f(x)]

    for t in range(1, max_iter + 1):
        # Compute gradient
        grad = grad_f(x)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Store path and cost
        path.append(x.copy())
        cost_history.append(f(x))

        # Check convergence
        if np.linalg.norm(grad) < tol:
            if verbose:
                print(f"Converged in {t} iterations")
            break

        if verbose and t % 100 == 0:
            print(f"Iteration {t}: f(x) = {f(x):.6f}, ||grad|| = {np.linalg.norm(grad):.6f}")

    return x, path, cost_history


# Example objective functions for testing
def quadratic_function(x: np.ndarray) -> float:
    """Simple quadratic: f(x,y) = x^2 + y^2"""
    return np.sum(x ** 2)


def quadratic_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of quadratic: grad = 2x"""
    return 2 * x


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def beale_function(x: np.ndarray) -> float:
    """Beale function: a common test function for optimization"""
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)


def beale_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of Beale function"""
    t1 = 1.5 - x[0] + x[0]*x[1]
    t2 = 2.25 - x[0] + x[0]*x[1]**2
    t3 = 2.625 - x[0] + x[0]*x[1]**3

    dx = (2*t1*(-1 + x[1]) +
          2*t2*(-1 + x[1]**2) +
          2*t3*(-1 + x[1]**3))

    dy = (2*t1*x[0] +
          2*t2*x[0]*2*x[1] +
          2*t3*x[0]*3*x[1]**2)

    return np.array([dx, dy])


if __name__ == "__main__":
    # Example usage: minimize f(x,y) = x^2 + y^2
    print("=" * 60)
    print("Gradient Descent on f(x,y) = x^2 + y^2")
    print("=" * 60)

    x0 = np.array([3.0, 4.0])
    x_opt, path, cost_history = gradient_descent(
        quadratic_function,
        quadratic_gradient,
        x0,
        alpha=0.1,
        max_iter=100,
        verbose=True
    )

    print(f"\nStarting point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Final cost: {quadratic_function(x_opt):.8f}")
    print(f"Number of iterations: {len(path) - 1}")

"""
Linear Regression with Gradient Descent
Implements linear regression from scratch using various gradient descent methods
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from gradient_descent import (gradient_descent, stochastic_gradient_descent,
                              mini_batch_gradient_descent)


class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent

    Model: y = X @ theta
    Cost Function: MSE = (1/2m) * sum((h(x) - y)^2)
    """

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000,
                 tol: float = 1e-6, method: str = 'batch'):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate (alpha)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        method : str
            Optimization method: 'batch', 'stochastic', or 'mini-batch'
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.theta = None
        self.cost_history = []
        self.theta_history = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term (column of ones) to feature matrix"""
        return np.c_[np.ones(X.shape[0]), X]

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost function
        J(theta) = (1/2m) * sum((X @ theta - y)^2)
        """
        m = len(y)
        predictions = X @ theta
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cost function
        grad = (1/m) * X^T @ (X @ theta - y)
        """
        m = len(y)
        predictions = X @ theta
        gradient = (1 / m) * X.T @ (predictions - y)
        return gradient

    def _compute_gradient_single(self, theta: np.ndarray, xi: np.ndarray, yi: float) -> np.ndarray:
        """Compute gradient for a single sample (for SGD)"""
        prediction = xi @ theta
        gradient = xi * (prediction - yi)
        return gradient

    def _compute_gradient_batch(self, theta: np.ndarray, X_batch: np.ndarray,
                                y_batch: np.ndarray) -> np.ndarray:
        """Compute gradient for a mini-batch"""
        m = len(y_batch)
        predictions = X_batch @ theta
        gradient = (1 / m) * X_batch.T @ (predictions - y_batch)
        return gradient

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32,
            verbose: bool = False) -> 'LinearRegressionGD':
        """
        Fit linear regression model using gradient descent

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            Training features
        y : np.ndarray, shape (m,)
            Target values
        batch_size : int
            Batch size for mini-batch GD
        verbose : bool
            If True, print progress

        Returns:
        --------
        self : LinearRegressionGD
            Fitted model
        """
        # Add intercept term
        X_b = self._add_intercept(X)
        m, n = X_b.shape

        # Initialize parameters
        self.theta = np.zeros(n)

        # Cost function for gradient descent
        cost_func = lambda theta: self._compute_cost(X_b, y, theta)
        grad_func = lambda theta: self._compute_gradient(X_b, y, theta)

        if self.method == 'batch':
            # Batch Gradient Descent
            self.theta, theta_path, self.cost_history = gradient_descent(
                cost_func, grad_func, self.theta,
                alpha=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=verbose
            )
            self.theta_history = theta_path

        elif self.method == 'stochastic':
            # Stochastic Gradient Descent
            self.theta, theta_path, self.cost_history = stochastic_gradient_descent(
                cost_func,
                self._compute_gradient_single,
                self.theta,
                X_b, y,
                alpha=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=verbose
            )
            self.theta_history = theta_path

        elif self.method == 'mini-batch':
            # Mini-batch Gradient Descent
            self.theta, theta_path, self.cost_history = mini_batch_gradient_descent(
                cost_func,
                self._compute_gradient_batch,
                self.theta,
                X_b, y,
                alpha=self.learning_rate,
                batch_size=batch_size,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=verbose
            )
            self.theta_history = theta_path

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the learned model

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            Features for prediction

        Returns:
        --------
        predictions : np.ndarray, shape (m,)
            Predicted values
        """
        if self.theta is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_b = self._add_intercept(X)
        return X_b @ self.theta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score

        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True values

        Returns:
        --------
        r2_score : float
            R² coefficient of determination
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def generate_linear_data(n_samples: int = 100, n_features: int = 1,
                        noise: float = 10.0, random_state: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise : float
        Standard deviation of Gaussian noise
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Target values
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features + 1) * 10  # Including intercept
    X_b = np.c_[np.ones((n_samples, 1)), X]
    y = X_b @ true_theta + np.random.randn(n_samples) * noise

    return X, y, true_theta


def plot_regression_line(X: np.ndarray, y: np.ndarray, model: LinearRegressionGD,
                        title: str = "Linear Regression Fit",
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None):
    """
    Plot regression line for 1D data

    Parameters:
    -----------
    X : np.ndarray, shape (m, 1)
        Features (must be 1-dimensional)
    y : np.ndarray
        Target values
    model : LinearRegressionGD
        Fitted model
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    if X.shape[1] != 1:
        raise ValueError("This function only works for 1D features")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    ax.scatter(X, y, alpha=0.5, s=50, label='Data points')

    # Plot regression line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_line)
    ax.plot(X_line, y_pred, 'r-', linewidth=2, label='Regression line')

    # Add equation to plot
    theta = model.theta
    equation = f'y = {theta[0]:.2f} + {theta[1]:.2f}x'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # R² score
    r2 = model.score(X, y)
    ax.text(0.05, 0.87, f'R² = {r2:.4f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_cost_convergence(models_dict: dict,
                         title: str = "Cost Function Convergence",
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None):
    """
    Plot cost function convergence for multiple models

    Parameters:
    -----------
    models_dict : dict
        Dictionary mapping model names to fitted LinearRegressionGD objects
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, (name, model) in enumerate(models_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(model.cost_history, label=name, linewidth=2,
               color=color, alpha=0.8)

    ax.set_xlabel('Iteration/Epoch', fontsize=12)
    ax.set_ylabel('Cost (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def compare_gd_methods(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      max_iter: int = 100,
                      figsize: Tuple[int, int] = (15, 5),
                      save_path: Optional[str] = None):
    """
    Compare Batch, SGD, and Mini-batch gradient descent

    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target values
    learning_rate : float
        Learning rate for all methods
    max_iter : int
        Maximum iterations/epochs
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    # Train models with different methods
    methods = ['batch', 'stochastic', 'mini-batch']
    models = {}

    for method in methods:
        model = LinearRegressionGD(
            learning_rate=learning_rate,
            max_iter=max_iter,
            method=method
        )
        model.fit(X, y, verbose=False)
        models[method.capitalize()] = model

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Regression lines (if 1D)
    if X.shape[1] == 1:
        for idx, (name, model) in enumerate(models.items()):
            ax = axes[0]
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = model.predict(X_line)
            ax.plot(X_line, y_pred, linewidth=2, label=name, alpha=0.7)

        ax.scatter(X, y, alpha=0.3, s=30, c='black', label='Data')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title('Regression Lines', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # Plot 2: Cost convergence
    ax = axes[1]
    colors = ['blue', 'red', 'green']
    for idx, (name, model) in enumerate(models.items()):
        ax.plot(model.cost_history, label=name, linewidth=2,
               color=colors[idx], alpha=0.8)

    ax.set_xlabel('Iteration/Epoch', fontsize=11)
    ax.set_ylabel('Cost (MSE)', fontsize=11)
    ax.set_title('Cost Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: R² scores comparison
    ax = axes[2]
    names = list(models.keys())
    r2_scores = [model.score(X, y) for model in models.values()]
    bars = ax.bar(names, r2_scores, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return models


if __name__ == "__main__":
    print("=" * 60)
    print("Linear Regression with Gradient Descent Demo")
    print("=" * 60)

    # Generate synthetic data
    X, y, true_theta = generate_linear_data(n_samples=100, n_features=1,
                                           noise=10.0, random_state=42)

    print(f"\nGenerated {len(X)} samples")
    print(f"True parameters: {true_theta}")

    # Train model
    model = LinearRegressionGD(learning_rate=0.1, max_iter=100, method='batch')
    model.fit(X, y, verbose=True)

    print(f"\nLearned parameters: {model.theta}")
    print(f"R² score: {model.score(X, y):.4f}")

    # Plot results
    plot_regression_line(X, y, model)

    # Compare methods
    print("\n" + "=" * 60)
    print("Comparing GD Methods")
    print("=" * 60)
    models = compare_gd_methods(X, y, learning_rate=0.1, max_iter=50)

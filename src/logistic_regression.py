"""
Logistic Regression with Gradient Descent
Implements binary logistic regression from scratch using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from gradient_descent import gradient_descent


class LogisticRegressionGD:
    """
    Logistic Regression using Gradient Descent

    Model: h(x) = sigmoid(X @ theta)
    Cost Function: Cross-entropy loss
    """

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000,
                 tol: float = 1e-6):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate (alpha)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.theta = None
        self.cost_history = []
        self.theta_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        sigmoid(z) = 1 / (1 + exp(-z))
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term (column of ones) to feature matrix"""
        return np.c_[np.ones(X.shape[0]), X]

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute cross-entropy cost function
        J(theta) = -(1/m) * sum(y*log(h) + (1-y)*log(1-h))
        """
        m = len(y)
        h = self.sigmoid(X @ theta)

        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)

        cost = -(1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
        return cost

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cost function
        grad = (1/m) * X^T @ (h - y)
        where h = sigmoid(X @ theta)
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        gradient = (1 / m) * X.T @ (h - y)
        return gradient

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LogisticRegressionGD':
        """
        Fit logistic regression model using gradient descent

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            Training features
        y : np.ndarray, shape (m,)
            Target values (0 or 1)
        verbose : bool
            If True, print progress

        Returns:
        --------
        self : LogisticRegressionGD
            Fitted model
        """
        # Add intercept term
        X_b = self._add_intercept(X)
        m, n = X_b.shape

        # Initialize parameters
        self.theta = np.zeros(n)

        # Cost function and gradient for gradient descent
        cost_func = lambda theta: self._compute_cost(X_b, y, theta)
        grad_func = lambda theta: self._compute_gradient(X_b, y, theta)

        # Run gradient descent
        self.theta, theta_path, self.cost_history = gradient_descent(
            cost_func, grad_func, self.theta,
            alpha=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=verbose
        )
        self.theta_history = theta_path

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            Features for prediction

        Returns:
        --------
        probabilities : np.ndarray, shape (m,)
            Predicted probabilities for class 1
        """
        if self.theta is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_b = self._add_intercept(X)
        return self.sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            Features for prediction
        threshold : float
            Classification threshold (default 0.5)

        Returns:
        --------
        predictions : np.ndarray, shape (m,)
            Predicted classes (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score

        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True labels

        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def generate_classification_data(n_samples: int = 100, n_features: int = 2,
                                n_classes: int = 2, random_state: Optional[int] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic binary classification data

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes (must be 2 for binary classification)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Binary labels (0 or 1)
    """
    if n_classes != 2:
        raise ValueError("Only binary classification (n_classes=2) is supported")

    if random_state is not None:
        np.random.seed(random_state)

    # Generate two clusters
    n_samples_per_class = n_samples // 2

    # Class 0: centered around origin with some spread
    X0 = np.random.randn(n_samples_per_class, n_features) + np.array([2.0] * n_features)
    y0 = np.zeros(n_samples_per_class)

    # Class 1: centered away from origin
    X1 = np.random.randn(n_samples_per_class, n_features) + np.array([-2.0] * n_features)
    y1 = np.ones(n_samples_per_class)

    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: LogisticRegressionGD,
                          title: str = "Logistic Regression Decision Boundary",
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None):
    """
    Plot decision boundary for 2D data

    Parameters:
    -----------
    X : np.ndarray, shape (m, 2)
        Features (must be 2-dimensional)
    y : np.ndarray
        Binary labels
    model : LogisticRegressionGD
        Fitted model
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    if X.shape[1] != 2:
        raise ValueError("This function only works for 2D features")

    fig, ax = plt.subplots(figsize=figsize)

    # Create mesh grid
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh grid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and regions
    contourf = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    contour = ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.colorbar(contourf, ax=ax, label='P(y=1)')

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                        s=100, edgecolors='black', linewidths=1.5, alpha=0.8)

    # Add decision boundary label
    ax.clabel(contour, inline=True, fontsize=10, fmt='Decision Boundary')

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add accuracy to plot
    accuracy = model.score(X, y)
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_sigmoid_function(figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None):
    """
    Plot the sigmoid activation function

    Parameters:
    -----------
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    z = np.linspace(-10, 10, 200)
    sigmoid = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(z, sigmoid, 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Decision threshold')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add annotations
    ax.annotate('σ(0) = 0.5', xy=(0, 0.5), xytext=(2, 0.5),
               fontsize=11, arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('z = θ^T x', fontsize=12)
    ax.set_ylabel('σ(z)', fontsize=12)
    ax.set_title('Sigmoid Activation Function', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_cost_and_accuracy(model: LogisticRegressionGD, X: np.ndarray, y: np.ndarray,
                          figsize: Tuple[int, int] = (12, 5),
                          save_path: Optional[str] = None):
    """
    Plot cost convergence and accuracy evolution

    Parameters:
    -----------
    model : LogisticRegressionGD
        Fitted model
    X : np.ndarray
        Features
    y : np.ndarray
        True labels
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot cost history
    ax1.plot(model.cost_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost (Cross-Entropy)', fontsize=11)
    ax1.set_title('Cost Function Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Compute accuracy at each iteration
    X_b = model._add_intercept(X)
    accuracies = []
    for theta in model.theta_history:
        proba = model.sigmoid(X_b @ theta)
        pred = (proba >= 0.5).astype(int)
        acc = np.mean(pred == y)
        accuracies.append(acc)

    # Plot accuracy history
    ax2.plot(accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Add final accuracy annotation
    final_acc = accuracies[-1]
    ax2.annotate(f'Final: {final_acc:.2%}',
                xy=(len(accuracies)-1, final_acc),
                xytext=(len(accuracies)*0.6, final_acc-0.1),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_learning_rate_effect(X: np.ndarray, y: np.ndarray,
                              learning_rates: List[float],
                              max_iter: int = 100,
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None):
    """
    Compare logistic regression with different learning rates

    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    learning_rates : list of float
        Learning rates to compare
    max_iter : int
        Maximum iterations
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    models = {}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(learning_rates)))

    for lr, color in zip(learning_rates, colors):
        model = LogisticRegressionGD(learning_rate=lr, max_iter=max_iter, tol=1e-10)
        model.fit(X, y, verbose=False)
        models[f'α={lr}'] = model

        # Plot cost history
        ax1.plot(model.cost_history, color=color, linewidth=2,
                label=f'α={lr}', alpha=0.8)

        # Plot accuracy
        X_b = model._add_intercept(X)
        accuracies = []
        for theta in model.theta_history:
            proba = model.sigmoid(X_b @ theta)
            pred = (proba >= 0.5).astype(int)
            acc = np.mean(pred == y)
            accuracies.append(acc)
        ax2.plot(accuracies, color=color, linewidth=2, label=f'α={lr}', alpha=0.8)

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost', fontsize=11)
    ax1.set_title('Cost Convergence', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy Evolution', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot final accuracies comparison
    names = list(models.keys())
    final_accs = [model.score(X, y) for model in models.values()]
    bars = ax3.bar(names, final_accs, color=colors, alpha=0.7, edgecolor='black')

    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=9)

    ax3.set_ylabel('Final Accuracy', fontsize=11)
    ax3.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.1])
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot convergence iterations
    convergence_iters = [len(model.cost_history) for model in models.values()]
    bars = ax4.bar(names, convergence_iters, color=colors, alpha=0.7, edgecolor='black')

    for bar, iters in zip(bars, convergence_iters):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{iters}', ha='center', va='bottom', fontsize=9)

    ax4.set_ylabel('Iterations to Convergence', fontsize=11)
    ax4.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Logistic Regression with Gradient Descent Demo")
    print("=" * 60)

    # Generate synthetic data
    X, y = generate_classification_data(n_samples=200, n_features=2, random_state=42)

    print(f"\nGenerated {len(X)} samples")
    print(f"Class distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

    # Train model
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=1000)
    model.fit(X, y, verbose=True)

    print(f"\nLearned parameters: {model.theta}")
    print(f"Accuracy: {model.score(X, y):.2%}")

    # Visualizations
    print("\nGenerating visualizations...")

    # Plot sigmoid function
    plot_sigmoid_function()

    # Plot decision boundary
    if X.shape[1] == 2:
        plot_decision_boundary(X, y, model)

    # Plot cost and accuracy
    plot_cost_and_accuracy(model, X, y)

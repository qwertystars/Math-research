"""
Real Dataset Applications
Provides functions to load and apply gradient descent to real datasets
"""

import numpy as np
from typing import Tuple
from linear_regression import LinearRegressionGD
from logistic_regression import LogisticRegressionGD


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load California Housing dataset (synthetic version)

    Returns:
    --------
    X : np.ndarray
        Features
    y : np.ndarray
        Target (house prices)
    """
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        return data.data, data.target
    except:
        # Fallback: generate synthetic housing data
        print("Generating synthetic housing data...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 8

        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features) * 10
        y = X @ true_weights + np.random.randn(n_samples) * 5

        return X, y


def load_iris_binary() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Iris dataset (binary classification: setosa vs others)

    Returns:
    --------
    X : np.ndarray
        Features (sepal length, sepal width, petal length, petal width)
    y : np.ndarray
        Binary labels (0 or 1)
    """
    try:
        from sklearn.datasets import load_iris
        data = load_iris()
        # Use first two classes only for binary classification
        mask = data.target < 2
        return data.data[mask], data.target[mask]
    except:
        # Fallback: generate synthetic flower data
        print("Generating synthetic classification data...")
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        # Class 0
        X0 = np.random.randn(n_samples // 2, n_features) + np.array([1, 1, -1, -1])
        y0 = np.zeros(n_samples // 2)

        # Class 1
        X1 = np.random.randn(n_samples // 2, n_features) + np.array([-1, -1, 1, 1])
        y1 = np.ones(n_samples // 2)

        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])

        # Shuffle
        indices = np.random.permutation(n_samples)
        return X[indices], y[indices]


def load_boston_housing() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Boston Housing dataset (or synthetic equivalent)

    Returns:
    --------
    X : np.ndarray
        Features
    y : np.ndarray
        Target (house prices)
    """
    # Generate synthetic housing data
    print("Generating synthetic housing data...")
    np.random.seed(42)
    n_samples = 500
    n_features = 13

    X = np.random.randn(n_samples, n_features)
    # Create meaningful relationships
    true_weights = np.array([5, -2, 3, -1, 4, 2, -3, 1, 2, -1, 3, -2, 4])
    y = X @ true_weights + np.random.randn(n_samples) * 3
    y = y - y.min() + 10  # Shift to positive range

    return X, y


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix

    Returns:
    --------
    X_scaled : np.ndarray
        Standardized features
    mean : np.ndarray
        Mean of each feature
    std : np.ndarray
        Standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets

    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    test_size : float
        Fraction of data to use for testing
    random_state : int, optional
        Random seed

    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def demo_regression_on_real_data():
    """
    Demonstrate linear regression on real dataset
    """
    import matplotlib.pyplot as plt

    print("="*70)
    print("LINEAR REGRESSION ON CALIFORNIA HOUSING DATASET")
    print("="*70)

    # Load data
    X, y = load_california_housing()
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Standardize features
    X_train_scaled, mean, std = standardize_features(X_train)
    X_test_scaled = (X_test - mean) / std

    # Compare different methods
    methods = ['batch', 'stochastic', 'mini-batch']
    results = {}

    for method in methods:
        print(f"\n{method.upper()}:")
        model = LinearRegressionGD(
            learning_rate=0.01,
            max_iter=100,
            method=method
        )
        model.fit(X_train_scaled, y_train, batch_size=32, verbose=False)

        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²:  {test_score:.4f}")

        results[method] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score
        }

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot cost convergence
    for method, result in results.items():
        model = result['model']
        ax1.plot(model.cost_history, linewidth=2, label=method.capitalize(), alpha=0.8)

    ax1.set_xlabel('Iteration/Epoch', fontsize=11)
    ax1.set_ylabel('Cost (MSE)', fontsize=11)
    ax1.set_title('Cost Convergence', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot R² scores
    methods_list = list(results.keys())
    train_scores = [results[m]['train_score'] for m in methods_list]
    test_scores = [results[m]['test_score'] for m in methods_list]

    x = np.arange(len(methods_list))
    width = 0.35

    bars1 = ax2.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
    bars2 = ax2.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)

    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_title('Model Performance', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in methods_list])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('../results/real_data_regression.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to: results/real_data_regression.png")
    plt.show()


def demo_classification_on_real_data():
    """
    Demonstrate logistic regression on real dataset
    """
    import matplotlib.pyplot as plt

    print("\n" + "="*70)
    print("LOGISTIC REGRESSION ON IRIS DATASET")
    print("="*70)

    # Load data
    X, y = load_iris_binary()
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    X_train_scaled, mean, std = standardize_features(X_train)
    X_test_scaled = (X_test - mean) / std

    # Train model
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=1000)
    model.fit(X_train_scaled, y_train, verbose=True)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"\nTraining accuracy: {train_acc:.2%}")
    print(f"Testing accuracy:  {test_acc:.2%}")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot cost history
    ax1.plot(model.cost_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost (Cross-Entropy)', fontsize=11)
    ax1.set_title('Cost Convergence', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    categories = ['Training', 'Testing']
    accuracies = [train_acc, test_acc]
    colors = ['steelblue', 'coral']

    bars = ax2.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=12)

    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Model Performance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/real_data_classification.png', dpi=300, bbox_inches='tight')
    print("Figure saved to: results/real_data_classification.png")
    plt.show()


if __name__ == "__main__":
    # Run demonstrations
    demo_regression_on_real_data()
    demo_classification_on_real_data()

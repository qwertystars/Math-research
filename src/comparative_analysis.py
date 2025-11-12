"""
Comparative Analysis of Gradient Descent Methods
Compares Batch GD, Stochastic GD, Mini-batch GD, Momentum, and Adam
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from gradient_descent import (
    gradient_descent,
    gradient_descent_with_momentum,
    adam_optimizer,
    quadratic_function,
    quadratic_gradient,
    rosenbrock_function,
    rosenbrock_gradient,
    beale_function,
    beale_gradient
)
from linear_regression import LinearRegressionGD, generate_linear_data
from visualizations import plot_comparison_algorithms, plot_cost_history


def compare_optimization_algorithms(
    f, grad_f,
    x0: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 200,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    title: str = "Algorithm Comparison",
    figsize: Tuple[int, int] = (18, 10),
    save_path: str = None
):
    """
    Compare different optimization algorithms on the same problem

    Parameters:
    -----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    x0 : np.ndarray
        Initial point
    alpha : float
        Learning rate
    max_iter : int
        Maximum iterations
    x_range : tuple
        X-axis range for visualization
    y_range : tuple
        Y-axis range for visualization
    title : str
        Main title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    algorithms = {
        'Batch GD': lambda: gradient_descent(f, grad_f, x0, alpha=alpha,
                                            max_iter=max_iter, tol=1e-10),
        'Momentum': lambda: gradient_descent_with_momentum(f, grad_f, x0, alpha=alpha,
                                                           beta=0.9, max_iter=max_iter, tol=1e-10),
        'Adam': lambda: adam_optimizer(f, grad_f, x0, alpha=alpha*0.1,
                                      max_iter=max_iter, tol=1e-10)
    }

    results = {}
    paths = {}
    costs = {}
    times = {}

    print("=" * 70)
    print(f"Comparing Optimization Algorithms: {title}")
    print("=" * 70)

    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        start_time = time.time()
        x_opt, path, cost_history = algo()
        elapsed_time = time.time() - start_time

        results[name] = x_opt
        paths[name] = path
        costs[name] = cost_history
        times[name] = elapsed_time

        print(f"  Final point: {x_opt}")
        print(f"  Final cost: {f(x_opt):.8f}")
        print(f"  Iterations: {len(path)-1}")
        print(f"  Time: {elapsed_time:.4f} seconds")

    # Create comprehensive comparison plot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Create grid for contour plots
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    colors = {'Batch GD': 'red', 'Momentum': 'blue', 'Adam': 'green'}

    # Plot 1-3: Individual paths on contour
    for idx, (name, path) in enumerate(paths.items()):
        ax = fig.add_subplot(gs[0, idx])
        ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.4)
        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.2)

        path_array = np.array(path)
        color = colors[name]
        ax.plot(path_array[:, 0], path_array[:, 1], '.-',
               color=color, linewidth=2, markersize=4, alpha=0.8)
        ax.plot(path_array[0, 0], path_array[0, 1], 'ko',
               markersize=12, label='Start', zorder=5)
        ax.plot(path_array[-1, 0], path_array[-1, 1], 'r*',
               markersize=15, label='End', zorder=5)

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'{name}\n({len(path)-1} iterations)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot 4: Cost comparison
    ax = fig.add_subplot(gs[1, 0])
    for name, cost_history in costs.items():
        color = colors[name]
        ax.plot(cost_history, color=color, linewidth=2, label=name, alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Cost', fontsize=10)
    ax.set_title('Cost Convergence', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 5: Iterations comparison
    ax = fig.add_subplot(gs[1, 1])
    names = list(paths.keys())
    iterations = [len(path)-1 for path in paths.values()]
    bars = ax.bar(names, iterations,
                  color=[colors[n] for n in names],
                  alpha=0.7, edgecolor='black')

    for bar, iters in zip(bars, iterations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{iters}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Iterations to Convergence', fontsize=10)
    ax.set_title('Convergence Speed', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Time comparison
    ax = fig.add_subplot(gs[1, 2])
    time_values = [times[name] for name in names]
    bars = ax.bar(names, time_values,
                  color=[colors[n] for n in names],
                  alpha=0.7, edgecolor='black')

    for bar, t in zip(bars, time_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{t:.4f}s', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Execution Time (seconds)', fontsize=10)
    ax.set_title('Computational Cost', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return results, paths, costs, times


def compare_gd_variants_on_linear_regression(
    n_samples: int = 1000,
    n_features: int = 1,
    noise: float = 20.0,
    learning_rate: float = 0.1,
    max_iter: int = 50,
    batch_size: int = 32,
    figsize: Tuple[int, int] = (16, 10),
    save_path: str = None
):
    """
    Compare Batch GD, SGD, and Mini-batch GD on linear regression

    Parameters:
    -----------
    n_samples : int
        Number of training samples
    n_features : int
        Number of features
    noise : float
        Noise level in data
    learning_rate : float
        Learning rate
    max_iter : int
        Maximum iterations/epochs
    batch_size : int
        Batch size for mini-batch GD
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    print("=" * 70)
    print("Comparing GD Variants on Linear Regression")
    print("=" * 70)

    # Generate data
    X, y, true_theta = generate_linear_data(n_samples=n_samples, n_features=n_features,
                                           noise=noise, random_state=42)

    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"True parameters: {true_theta}")

    # Train models
    methods = {
        'Batch GD': 'batch',
        'Stochastic GD': 'stochastic',
        'Mini-batch GD': 'mini-batch'
    }

    models = {}
    times = {}

    for name, method in methods.items():
        print(f"\nTraining {name}...")
        model = LinearRegressionGD(
            learning_rate=learning_rate,
            max_iter=max_iter,
            method=method
        )

        start_time = time.time()
        model.fit(X, y, batch_size=batch_size, verbose=False)
        elapsed_time = time.time() - start_time

        models[name] = model
        times[name] = elapsed_time

        r2 = model.score(X, y)
        print(f"  R² score: {r2:.4f}")
        print(f"  Final cost: {model.cost_history[-1]:.4f}")
        print(f"  Time: {elapsed_time:.4f} seconds")

    # Create comparison plot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    colors = {'Batch GD': 'blue', 'Stochastic GD': 'red', 'Mini-batch GD': 'green'}

    # Plot 1: Data and regression lines (if 1D)
    if n_features == 1:
        ax = fig.add_subplot(gs[0, :])
        ax.scatter(X, y, alpha=0.3, s=30, c='gray', label='Data')

        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        for name, model in models.items():
            y_pred = model.predict(X_line)
            color = colors[name]
            ax.plot(X_line, y_pred, linewidth=2.5, label=name,
                   color=color, alpha=0.8)

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title('Regression Lines Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot 2: Cost convergence
    ax = fig.add_subplot(gs[1, 0])
    for name, model in models.items():
        color = colors[name]
        ax.plot(model.cost_history, color=color, linewidth=2.5,
               label=name, alpha=0.8)

    ax.set_xlabel('Iteration/Epoch', fontsize=11)
    ax.set_ylabel('Cost (MSE)', fontsize=11)
    ax.set_title('Cost Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: R² scores
    ax = fig.add_subplot(gs[1, 1])
    names = list(models.keys())
    r2_scores = [model.score(X, y) for model in models.values()]
    bars = ax.bar(names, r2_scores,
                  color=[colors[n] for n in names],
                  alpha=0.7, edgecolor='black')

    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Model Performance', fontsize=12, fontweight='bold')

    # Compute dynamic y-limits to show negative R² values
    margin = 0.1
    y_lower = min(r2_scores) - margin
    y_upper = max(max(r2_scores), 1.0) + margin
    ax.set_ylim(y_lower, y_upper)

    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Time comparison
    ax = fig.add_subplot(gs[1, 2])
    time_values = [times[name] for name in names]
    bars = ax.bar(names, time_values,
                  color=[colors[n] for n in names],
                  alpha=0.7, edgecolor='black')

    for bar, t in zip(bars, time_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{t:.4f}s', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Training Time (seconds)', fontsize=11)
    ax.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Batch vs SGD vs Mini-batch GD on Linear Regression',
                fontsize=15, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return models, times


def analyze_learning_rate_sensitivity(
    f, grad_f,
    x0: np.ndarray,
    learning_rates: List[float],
    max_iter: int = 100,
    figsize: Tuple[int, int] = (14, 10),
    save_path: str = None
):
    """
    Analyze how learning rate affects convergence

    Parameters:
    -----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    x0 : np.ndarray
        Initial point
    learning_rates : list of float
        Learning rates to test
    max_iter : int
        Maximum iterations
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    print("=" * 70)
    print("Learning Rate Sensitivity Analysis")
    print("=" * 70)

    results = {}

    for lr in learning_rates:
        print(f"\nLearning rate: {lr}")
        try:
            x_opt, path, cost_history = gradient_descent(
                f, grad_f, x0, alpha=lr, max_iter=max_iter, tol=1e-10
            )
        except (FloatingPointError, OverflowError, ValueError) as err:
            print(f"  Diverged! ({err})")
            results[lr] = {'converged': False, 'error': str(err)}
            continue

        results[lr] = {
            'x_opt': x_opt,
            'path': path,
            'cost_history': cost_history,
            'converged': True
        }
        print(f"  Converged in {len(path)-1} iterations")
        print(f"  Final cost: {f(x_opt):.8f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Cost histories
    ax = axes[0, 0]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(learning_rates)))
    for lr, color in zip(learning_rates, colors):
        if results[lr]['converged']:
            ax.plot(results[lr]['cost_history'], color=color,
                   linewidth=2, label=f'α={lr}', alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Cost', fontsize=11)
    ax.set_title('Cost Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Convergence speed
    ax = axes[0, 1]
    converged_lrs = [lr for lr in learning_rates if results[lr]['converged']]
    iterations = [len(results[lr]['path'])-1 for lr in converged_lrs]
    bars = ax.bar([f'α={lr}' for lr in converged_lrs], iterations,
                  color=[colors[i] for i, lr in enumerate(learning_rates) if results[lr]['converged']],
                  alpha=0.7, edgecolor='black')

    for bar, iters in zip(bars, iterations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{iters}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Iterations to Convergence', fontsize=11)
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Final costs
    ax = axes[1, 0]
    final_costs = [results[lr]['cost_history'][-1] for lr in converged_lrs]
    bars = ax.bar([f'α={lr}' for lr in converged_lrs], final_costs,
                  color=[colors[i] for i, lr in enumerate(learning_rates) if results[lr]['converged']],
                  alpha=0.7, edgecolor='black')

    for bar, cost in zip(bars, final_costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{cost:.2e}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Final Cost', fontsize=11)
    ax.set_title('Final Cost Values', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = []
    for lr in learning_rates:
        if results[lr]['converged']:
            summary_data.append([
                f"{lr}",
                f"{len(results[lr]['path'])-1}",
                f"{results[lr]['cost_history'][-1]:.2e}"
            ])
        else:
            summary_data.append([f"{lr}", "Diverged", "-"])

    table = ax.table(cellText=summary_data,
                    colLabels=['Learning Rate', 'Iterations', 'Final Cost'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    fig.suptitle('Learning Rate Sensitivity Analysis', fontsize=15, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    # Example 1: Compare optimization algorithms on quadratic function
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Quadratic Function f(x,y) = x² + y²")
    print("=" * 70)

    x0 = np.array([4.0, 4.0])
    compare_optimization_algorithms(
        quadratic_function,
        quadratic_gradient,
        x0,
        alpha=0.1,
        max_iter=100,
        title="Optimization on Quadratic Function"
    )

    # Example 2: Compare on Rosenbrock function
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Rosenbrock Function")
    print("=" * 70)

    x0 = np.array([0.0, 0.0])
    compare_optimization_algorithms(
        rosenbrock_function,
        rosenbrock_gradient,
        x0,
        alpha=0.001,
        max_iter=500,
        x_range=(-2, 2),
        y_range=(-1, 3),
        title="Optimization on Rosenbrock Function"
    )

    # Example 3: Compare GD variants on linear regression
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Linear Regression Comparison")
    print("=" * 70)

    compare_gd_variants_on_linear_regression(
        n_samples=1000,
        learning_rate=0.1,
        max_iter=50
    )

    # Example 4: Learning rate sensitivity
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Learning Rate Sensitivity")
    print("=" * 70)

    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    analyze_learning_rate_sensitivity(
        quadratic_function,
        quadratic_gradient,
        np.array([4.0, 4.0]),
        learning_rates,
        max_iter=100
    )

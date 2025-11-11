"""
Visualization Functions for Gradient Descent
Provides various plotting utilities to visualize optimization processes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List, Tuple, Optional
import matplotlib.patches as mpatches


def plot_contour_with_path(
    f: Callable,
    path: List[np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    title: str = "Gradient Descent Path on Contour Plot",
    levels: int = 30,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot contour map with gradient descent path

    Parameters:
    -----------
    f : callable
        The objective function
    path : list of np.ndarray
        The optimization path
    x_range : tuple
        Range for x-axis (min, max)
    y_range : tuple
        Range for y-axis (min, max)
    title : str
        Plot title
    levels : int
        Number of contour levels
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 300)
    y = np.linspace(y_range[0], y_range[1], 300)
    X, Y = np.meshgrid(x, y)

    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot contours
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, ax=ax, label='f(x, y)')

    # Extract path coordinates
    path_array = np.array(path)
    x_path = path_array[:, 0]
    y_path = path_array[:, 1]

    # Plot path
    ax.plot(x_path, y_path, 'r.-', linewidth=2, markersize=8, label='GD Path')
    ax.plot(x_path[0], y_path[0], 'go', markersize=15, label='Start', zorder=5)
    ax.plot(x_path[-1], y_path[-1], 'r*', markersize=20, label='End', zorder=5)

    # Add arrows to show direction
    for i in range(0, len(x_path)-1, max(1, len(x_path)//10)):
        ax.annotate('', xy=(x_path[i+1], y_path[i+1]), xytext=(x_path[i], y_path[i]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_3d_surface_with_path(
    f: Callable,
    path: List[np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    title: str = "Gradient Descent on 3D Surface",
    figsize: Tuple[int, int] = (12, 9),
    save_path: Optional[str] = None
):
    """
    Plot 3D surface with gradient descent path

    Parameters:
    -----------
    f : callable
        The objective function
    path : list of np.ndarray
        The optimization path
    x_range : tuple
        Range for x-axis
    y_range : tuple
        Range for y-axis
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    # Extract path coordinates and function values
    path_array = np.array(path)
    x_path = path_array[:, 0]
    y_path = path_array[:, 1]
    z_path = np.array([f(p) for p in path])

    # Plot path
    ax.plot(x_path, y_path, z_path, 'r.-', linewidth=3, markersize=6, label='GD Path')
    ax.scatter(x_path[0], y_path[0], z_path[0], c='green', s=200, marker='o',
               label='Start', edgecolors='black', linewidths=2)
    ax.scatter(x_path[-1], y_path[-1], z_path[-1], c='red', s=300, marker='*',
               label='End', edgecolors='black', linewidths=2)

    # Labels and title
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('f(x, y)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def animate_gradient_descent(
    f: Callable,
    path: List[np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    title: str = "Gradient Descent Animation",
    interval: int = 100,
    save_path: Optional[str] = None
):
    """
    Create animation of gradient descent on contour plot

    Parameters:
    -----------
    f : callable
        The objective function
    path : list of np.ndarray
        The optimization path
    x_range : tuple
        Range for x-axis
    y_range : tuple
        Range for y-axis
    title : str
        Animation title
    interval : int
        Delay between frames in milliseconds
    save_path : str, optional
        If provided, save animation to this path (requires ffmpeg)
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)

    # Initialize path plot
    path_array = np.array(path)
    line, = ax.plot([], [], 'r.-', linewidth=2, markersize=8, label='GD Path')
    point, = ax.plot([], [], 'ro', markersize=12, label='Current Position')
    start_point = ax.plot(path_array[0, 0], path_array[0, 1], 'go',
                          markersize=15, label='Start', zorder=5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Animation function
    def animate(frame):
        x_data = path_array[:frame+1, 0]
        y_data = path_array[:frame+1, 1]
        line.set_data(x_data, y_data)
        point.set_data([x_data[-1]], [y_data[-1]])
        return line, point

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(path),
                        interval=interval, blit=True, repeat=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return anim


def plot_cost_history(
    cost_histories: dict,
    title: str = "Cost Function vs Iterations",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot cost function value over iterations for multiple algorithms

    Parameters:
    -----------
    cost_histories : dict
        Dictionary mapping algorithm names to their cost histories
        Example: {'GD': [costs], 'SGD': [costs], 'Adam': [costs]}
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for idx, (name, history) in enumerate(cost_histories.items()):
        color = colors[idx % len(colors)]
        ax.plot(history, label=name, linewidth=2, color=color, alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost Function Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often better for viewing convergence

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_learning_rate_comparison(
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    learning_rates: List[float],
    max_iter: int = 100,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    Compare gradient descent with different learning rates

    Parameters:
    -----------
    f : callable
        The objective function
    grad_f : callable
        The gradient function
    x0 : np.ndarray
        Initial point
    learning_rates : list of float
        Learning rates to compare
    max_iter : int
        Maximum iterations
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    from gradient_descent import gradient_descent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Create grid for contour
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Plot contours on first subplot
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.4)
    ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.2)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(learning_rates)))
    cost_histories = {}

    for lr, color in zip(learning_rates, colors):
        # Run gradient descent
        x_opt, path, cost_history = gradient_descent(f, grad_f, x0, alpha=lr,
                                                     max_iter=max_iter, tol=1e-10)

        # Plot path on contour
        path_array = np.array(path)
        ax1.plot(path_array[:, 0], path_array[:, 1], '.-',
                color=color, linewidth=1.5, markersize=4,
                label=f'α={lr}', alpha=0.7)

        # Store cost history
        cost_histories[f'α={lr}'] = cost_history

    # Mark start point
    ax1.plot(x0[0], x0[1], 'ko', markersize=12, label='Start', zorder=5)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('Paths with Different Learning Rates', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot cost histories on second subplot
    for lr, color in zip(learning_rates, colors):
        ax2.plot(cost_histories[f'α={lr}'], color=color,
                linewidth=2, label=f'α={lr}', alpha=0.8)

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Cost Function Value', fontsize=11)
    ax2.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_gradient_vectors(
    f: Callable,
    grad_f: Callable,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    grid_density: int = 20,
    title: str = "Gradient Vector Field",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot gradient vector field on contour map

    Parameters:
    -----------
    f : callable
        The objective function
    grad_f : callable
        The gradient function
    x_range : tuple
        Range for x-axis
    y_range : tuple
        Range for y-axis
    grid_density : int
        Number of grid points in each direction for vectors
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    # Create fine grid for contours
    x_fine = np.linspace(x_range[0], x_range[1], 200)
    y_fine = np.linspace(y_range[0], y_range[1], 200)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z = np.zeros_like(X_fine)
    for i in range(X_fine.shape[0]):
        for j in range(X_fine.shape[1]):
            Z[i, j] = f(np.array([X_fine[i, j], Y_fine[i, j]]))

    # Create coarse grid for vectors
    x_coarse = np.linspace(x_range[0], x_range[1], grid_density)
    y_coarse = np.linspace(y_range[0], y_range[1], grid_density)
    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

    # Compute gradients
    U = np.zeros_like(X_coarse)
    V = np.zeros_like(Y_coarse)
    for i in range(X_coarse.shape[0]):
        for j in range(X_coarse.shape[1]):
            grad = grad_f(np.array([X_coarse[i, j], Y_coarse[i, j]]))
            U[i, j] = -grad[0]  # Negative because we move opposite to gradient
            V[i, j] = -grad[1]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot contours
    contour = ax.contour(X_fine, Y_fine, Z, levels=30, cmap='viridis', alpha=0.6)
    ax.contourf(X_fine, Y_fine, Z, levels=30, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, ax=ax, label='f(x, y)')

    # Plot gradient vectors (negative gradient = descent direction)
    ax.quiver(X_coarse, Y_coarse, U, V, color='red', alpha=0.6,
             scale=None, scale_units='xy', angles='xy', width=0.003)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_comparison_algorithms(
    f: Callable,
    paths_dict: dict,
    cost_histories_dict: dict,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Compare multiple optimization algorithms side by side

    Parameters:
    -----------
    f : callable
        The objective function
    paths_dict : dict
        Dictionary mapping algorithm names to paths
    cost_histories_dict : dict
        Dictionary mapping algorithm names to cost histories
    x_range : tuple
        Range for x-axis
    y_range : tuple
        Range for y-axis
    figsize : tuple
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Create subplots
    n_algorithms = len(paths_dict)
    fig, axes = plt.subplots(1, n_algorithms + 1, figsize=figsize)

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Plot each algorithm's path
    for idx, (name, path) in enumerate(paths_dict.items()):
        ax = axes[idx]

        # Plot contours
        ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.4)
        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.2)

        # Plot path
        path_array = np.array(path)
        color = colors[idx % len(colors)]
        ax.plot(path_array[:, 0], path_array[:, 1], '.-',
               color=color, linewidth=2, markersize=6)
        ax.plot(path_array[0, 0], path_array[0, 1], 'go',
               markersize=12, label='Start', zorder=5)
        ax.plot(path_array[-1, 0], path_array[-1, 1], 'r*',
               markersize=15, label='End', zorder=5)

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot cost comparison
    ax = axes[-1]
    for idx, (name, history) in enumerate(cost_histories_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(history, color=color, linewidth=2, label=name, alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Cost', fontsize=10)
    ax.set_title('Cost Comparison', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    from gradient_descent import (gradient_descent, quadratic_function,
                                  quadratic_gradient)

    print("Running visualization examples...")

    # Example 1: Basic contour plot with path
    x0 = np.array([4.0, 4.0])
    x_opt, path, cost_history = gradient_descent(
        quadratic_function, quadratic_gradient, x0,
        alpha=0.1, max_iter=50
    )

    plot_contour_with_path(quadratic_function, path,
                          title="GD on f(x,y) = x² + y²")

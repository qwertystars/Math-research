# Gradient Descent & Optimization in Machine Learning

A comprehensive implementation and visualization toolkit for gradient descent optimization algorithms, from scratch using NumPy. This project demonstrates various gradient descent methods and their applications in machine learning.

## 📋 Project Overview

This repository contains:
- **Core gradient descent algorithms** (Batch, SGD, Mini-batch, Momentum, Adam)
- **Visualization tools** for understanding optimization paths
- **ML applications** (Linear Regression, Logistic Regression)
- **Comparative analysis** of different optimization methods
- **Interactive Jupyter notebooks** with demos
- **Real dataset applications**

## 🗂️ Project Structure

```
Math-research/
├── src/
│   ├── gradient_descent.py         # Core GD algorithms
│   ├── visualizations.py           # Plotting functions
│   ├── linear_regression.py        # Linear regression with GD
│   ├── logistic_regression.py      # Logistic regression with GD
│   ├── comparative_analysis.py     # Algorithm comparisons
│   └── datasets.py                 # Real dataset applications
├── notebooks/
│   ├── 01_gradient_descent_basics.ipynb
│   ├── 02_linear_regression_demo.ipynb
│   └── 03_algorithm_comparison.ipynb
├── data/                           # Data directory
├── results/                        # Output directory for figures
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Math-research

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

#### 1. Basic Gradient Descent

```python
from src.gradient_descent import gradient_descent, quadratic_function, quadratic_gradient
import numpy as np

# Define starting point
x0 = np.array([4.0, 3.0])

# Run gradient descent
x_opt, path, cost_history = gradient_descent(
    f=quadratic_function,
    grad_f=quadratic_gradient,
    x0=x0,
    alpha=0.1,
    max_iter=100
)

print(f"Optimal point: {x_opt}")
print(f"Final cost: {quadratic_function(x_opt)}")
```

#### 2. Linear Regression

```python
from src.linear_regression import LinearRegressionGD, generate_linear_data

# Generate synthetic data
X, y, true_theta = generate_linear_data(n_samples=100, n_features=1)

# Train model
model = LinearRegressionGD(learning_rate=0.1, max_iter=100, method='batch')
model.fit(X, y, verbose=True)

# Make predictions
predictions = model.predict(X)
r2_score = model.score(X, y)
print(f"R² score: {r2_score:.4f}")
```

#### 3. Visualizations

```python
from src.visualizations import plot_contour_with_path, plot_3d_surface_with_path

# Visualize optimization path on contour plot
plot_contour_with_path(
    quadratic_function,
    path,
    x_range=(-5, 5),
    y_range=(-5, 5),
    title="Gradient Descent Optimization"
)

# Visualize on 3D surface
plot_3d_surface_with_path(quadratic_function, path)
```

#### 4. Algorithm Comparison

```python
from src.comparative_analysis import compare_optimization_algorithms

x0 = np.array([4.0, 4.0])
results, paths, costs, times = compare_optimization_algorithms(
    f=quadratic_function,
    grad_f=quadratic_gradient,
    x0=x0,
    alpha=0.1,
    max_iter=100
)
```

### Interactive Jupyter Notebooks

Launch Jupyter and explore the interactive notebooks:

```bash
jupyter notebook notebooks/
```

**Available notebooks:**
1. `01_gradient_descent_basics.ipynb` - Understanding GD fundamentals
2. `02_linear_regression_demo.ipynb` - Linear regression with GD
3. `03_algorithm_comparison.ipynb` - Comprehensive algorithm comparison

## 📚 Implemented Algorithms

### 1. Batch Gradient Descent
- Uses all training samples for each update
- Stable convergence
- Best for small to medium datasets

### 2. Stochastic Gradient Descent (SGD)
- Uses one sample at a time
- Fast updates, can escape local minima
- Best for large datasets

### 3. Mini-batch Gradient Descent
- Uses small batches of samples
- Balance between speed and stability
- Most practical for real-world applications

### 4. Gradient Descent with Momentum
- Adds momentum term to accelerate convergence
- Helps navigate ravines and valleys
- Reduces oscillations

### 5. Adam Optimizer
- Adaptive learning rates
- Combines momentum and RMSprop
- Robust default choice for deep learning

## 📊 Visualizations

The project includes extensive visualization capabilities:

### Contour Plots
- Visualize optimization paths on 2D contour maps
- Show gradient vectors
- Display learning rate effects

### 3D Surface Plots
- Interactive 3D visualization of cost landscapes
- Path visualization on surfaces

### Convergence Analysis
- Cost vs iteration plots
- Learning rate comparison
- Algorithm performance comparison

### ML Application Plots
- Regression line fitting
- Decision boundaries for classification
- Cost convergence curves

## 🎯 Key Features

### Core Implementations
- ✅ Pure NumPy implementations (no ML frameworks)
- ✅ Comprehensive documentation
- ✅ Type hints and clean code
- ✅ Extensive examples

### Visualizations
- ✅ Contour plots with optimization paths
- ✅ 3D surface visualizations
- ✅ Gradient vector fields
- ✅ Cost convergence curves
- ✅ Learning rate comparison plots

### Machine Learning Applications
- ✅ Linear Regression with all GD variants
- ✅ Logistic Regression for binary classification
- ✅ Real dataset applications
- ✅ Performance metrics (R², accuracy)

### Interactive Features
- ✅ Jupyter notebook demos
- ✅ Interactive widgets for parameter tuning
- ✅ Animated optimization paths
- ✅ Side-by-side algorithm comparisons

## 📖 Detailed Documentation

### Gradient Descent Algorithm

**Mathematical Formulation:**
```
Initialize: θ₀ (parameters)
Repeat until convergence:
    1. Compute gradient: ∇J(θ)
    2. Update: θ := θ - α * ∇J(θ)
```

**Parameters:**
- `alpha` (α): Learning rate - controls step size
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance

### Linear Regression with Gradient Descent

**Model:** h(x) = θᵀx

**Cost Function (MSE):**
```
J(θ) = (1/2m) * Σ(h(xⁱ) - yⁱ)²
```

**Gradient:**
```
∇J(θ) = (1/m) * Xᵀ(Xθ - y)
```

### Logistic Regression with Gradient Descent

**Model:** h(x) = sigmoid(θᵀx)

**Cost Function (Cross-Entropy):**
```
J(θ) = -(1/m) * Σ[y log(h(x)) + (1-y) log(1-h(x))]
```

**Gradient:**
```
∇J(θ) = (1/m) * Xᵀ(h(X) - y)
```

## 🧪 Testing and Validation

### Test Functions Included:
1. **Quadratic Function**: f(x,y) = x² + y²
   - Simple convex function
   - Global minimum at (0, 0)

2. **Rosenbrock Function**: f(x,y) = (1-x)² + 100(y-x²)²
   - Banana-shaped valley
   - Global minimum at (1, 1)

3. **Beale Function**: Complex test function
   - Multiple local features
   - Tests algorithm robustness

### Running Tests

```bash
# Test core gradient descent
python src/gradient_descent.py

# Test linear regression
python src/linear_regression.py

# Test logistic regression
python src/logistic_regression.py

# Test comparative analysis
python src/comparative_analysis.py

# Test real dataset applications
python src/datasets.py
```

## 📈 Performance Comparison

| Algorithm | Convergence Speed | Stability | Memory | Best Use Case |
|-----------|------------------|-----------|--------|---------------|
| Batch GD | Slow | High | High | Small datasets |
| SGD | Fast | Low | Low | Large datasets |
| Mini-batch | Medium | Medium | Medium | General purpose |
| Momentum | Fast | Medium | Medium | Noisy objectives |
| Adam | Fast | High | Medium | Deep learning |

## 🎓 Educational Use

This project is designed for:
- **Students** learning optimization algorithms
- **Researchers** studying gradient descent methods
- **Practitioners** understanding ML fundamentals
- **Teachers** demonstrating optimization concepts

### Learning Path:
1. Start with `01_gradient_descent_basics.ipynb`
2. Explore `02_linear_regression_demo.ipynb`
3. Study `03_algorithm_comparison.ipynb`
4. Experiment with parameters in interactive demos
5. Apply to real datasets using `datasets.py`

## 📝 Code Examples

### Example 1: Custom Optimization Problem

```python
import numpy as np
from src.gradient_descent import gradient_descent

# Define custom function
def my_function(x):
    return x[0]**2 + 2*x[1]**2

def my_gradient(x):
    return np.array([2*x[0], 4*x[1]])

# Optimize
x0 = np.array([5.0, 5.0])
x_opt, path, cost_history = gradient_descent(
    my_function, my_gradient, x0, alpha=0.1
)
```

### Example 2: Compare All Algorithms

```python
from src.comparative_analysis import compare_optimization_algorithms

results, paths, costs, times = compare_optimization_algorithms(
    f=my_function,
    grad_f=my_gradient,
    x0=np.array([5.0, 5.0]),
    alpha=0.1,
    max_iter=200
)
```

### Example 3: Real Data Application

```python
from src.datasets import load_california_housing, train_test_split, standardize_features
from src.linear_regression import LinearRegressionGD

# Load and prepare data
X, y = load_california_housing()
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, mean, std = standardize_features(X_train)
X_test = (X_test - mean) / std

# Train and evaluate
model = LinearRegressionGD(learning_rate=0.01, max_iter=100)
model.fit(X_train, y_train)
print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

## 🎨 Visualization Gallery

The project generates publication-quality figures:
- Contour plots with optimization paths
- 3D surface plots
- Cost convergence curves
- Learning rate comparison plots
- Algorithm side-by-side comparisons
- Decision boundaries for classification

All figures can be saved with:
```python
plot_contour_with_path(..., save_path='results/my_plot.png')
```

## 🔧 Advanced Usage

### Custom Learning Rate Schedules

```python
# Implement custom learning rate decay
def train_with_lr_decay(X, y):
    model = LinearRegressionGD(learning_rate=0.1, max_iter=100)
    # Modify learning rate during training
    # (extend the class for this functionality)
```

### Early Stopping

```python
# Use tolerance parameter for early stopping
model = LinearRegressionGD(learning_rate=0.1, max_iter=1000, tol=1e-6)
model.fit(X, y, verbose=True)
# Will stop when gradient norm < tol
```

## 📊 Results and Outputs

All results are saved to the `results/` directory:
- Optimization path visualizations
- Cost convergence plots
- Algorithm comparison figures
- Performance metrics

## 🤝 Contributing

This is a research/educational project. Key areas for contribution:
- Additional optimization algorithms (RMSprop, AdaGrad, etc.)
- More test functions
- Additional ML applications
- Performance optimizations
- Documentation improvements

## 📄 License

This project is for educational and research purposes.

## 👥 Team Collaboration

This implementation supports a 3-person research team:
- **Member 1**: Mathematical theory and formulations
- **Member 2**: Algorithm descriptions and analysis
- **Member 3**: Implementation and visualization (this repository)

## 🔗 References

Key concepts implemented:
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-batch Gradient Descent
- Momentum-based optimization
- Adam optimizer
- Linear Regression
- Logistic Regression

## 📞 Support

For questions or issues:
1. Check the Jupyter notebooks for examples
2. Review the code documentation
3. Run the test scripts to verify setup

## 🎯 Learning Objectives

After working with this project, you will understand:
1. How gradient descent works mathematically
2. Different variants of gradient descent
3. When to use each optimization method
4. How to implement ML algorithms from scratch
5. How to visualize optimization processes
6. Trade-offs between different algorithms
7. Practical considerations for real-world applications

---

**Happy Learning! 🚀**

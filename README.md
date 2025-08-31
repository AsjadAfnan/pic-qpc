# Tree-PIC → QPC: Probabilistic Inference Circuits with Static Quadrature

A PyTorch library for Tree-structured Probabilistic Inference Circuits (Tree-PICs) with static quadrature for continuous latent variables, featuring an analytic Gaussian baseline.

## Overview

This library implements Tree-PICs with static quadrature (QPC) for continuous latent variables, supporting:

- **Two leaf types**: Gaussian (μ,σ) and Bernoulli (via logistic decoder)
- **Two conditional families**: 
  - Analytic Linear-Gaussian (closed-form PIC, ground truth)
  - Neural Energy-Based Models (small MLP + Fourier features)
- **Core inference operations**: `log_prob(x)`, `marginal_log_prob(x_S)`, `most_probable_explanation()`
- **Training**: Maximum likelihood on synthetic and UCI datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pic-qpc.git
cd pic-qpc

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from pic import LatentTree, TreePIC, QPC, LinearGaussian, GaussianLeaf, Quadrature

# Define tree structure
parents = {"root": None, "z1": "root", "x1": "z1", "x2": "z1"}
scopes = {
    "root": {"z1", "x1", "x2"},
    "z1": {"z1", "x1", "x2"},
    "x1": {"x1"},
    "x2": {"x2"}
}

tree = LatentTree.from_parents(parents, scopes)

# Create conditionals and leaves
A = torch.tensor([[1.0]])
b = torch.tensor([0.0])
Sigma = torch.tensor([[1.0]])

conditionals = {"z1": LinearGaussian("z1", A, b, Sigma)}
leaves = {
    "x1": GaussianLeaf("x1", mu=0.0, sigma=1.0),
    "x2": GaussianLeaf("x2", mu=0.0, sigma=1.0)
}

# Create TreePIC and compile to QPC
tree_pic = TreePIC(tree, conditionals, leaves)
quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 32)
qpc = tree_pic.compile_to_qpc(quadrature)

# Compute log probabilities
x = torch.randn(10, 2)
log_probs = qpc.log_prob(x)
print(f"Log probabilities: {log_probs}")
```

## Reproducing Results

### Synthetic Data Training

```bash
python examples/train_synth.py
```

This script:
- Generates synthetic data from known distributions
- Trains both Linear-Gaussian and Neural EBM models
- Plots training curves and convergence
- Saves results to `results/synthetic/`

### UCI Dataset Training

```bash
python examples/train_uci.py
```

This script:
- Loads UCI datasets (Boston, Wine, Breast Cancer)
- Creates appropriate tree structures
- Trains models and evaluates performance
- Saves results to `results/uci/`

### Analytic Sanity Check

```bash
python examples/analytic_sanity.py
```

This script:
- Tests against known analytic solutions
- Performs convergence studies
- Validates numerical stability
- Generates visualization plots

## Documentation

- **[Design Guide](docs/design.md)**: Architecture and implementation details
- **[Mathematical Background](docs/math.md)**: Theory and derivations
- **[Experimental Results](docs/experiments.md)**: Benchmarks and analysis

## API Reference

### Core Classes

- `LatentTree`: Tree structure specification and validation
- `TreePIC`: Symbolic probabilistic model
- `QPC`: Materialized circuit for inference
- `Quadrature`: Static quadrature rules (Gauss-Legendre, Trapezoid, Uniform)

### Leaf Nodes

- `GaussianLeaf`: Gaussian distribution leaf
- `BernoulliLeaf`: Bernoulli distribution leaf

### Conditional Distributions

- `LinearGaussian`: Analytic linear-Gaussian conditionals
- `NeuralEnergyConditional`: Neural energy-based conditionals

### Inference Operations

- `log_prob(x)`: Compute log probability of observations
- `marginal_log_prob(x_S)`: Compute marginal log probability
- `most_probable_explanation()`: Find MAP estimate

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pic

# Run specific test categories
pytest tests/test_structures.py
pytest tests/test_quadrature.py
pytest tests/test_conditionals.py
```

## Development

### Code Quality

```bash
# Format code
black pic/ tests/ examples/

# Lint code
ruff check pic/ tests/ examples/

# Type checking
mypy pic/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{pic_qpc_2024,
  title={Tree-PIC → QPC: Probabilistic Inference Circuits with Static Quadrature},
  author={Tree-PIC Team},
  year={2024},
  url={https://github.com/your-org/pic-qpc}
}
```

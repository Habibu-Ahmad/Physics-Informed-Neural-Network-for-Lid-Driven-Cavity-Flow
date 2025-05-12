# Unified PINN for Steady Navier-Stokes Equations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/reponame/blob/main/main.ipynb)

Physics-Informed Neural Network (PINN) implementation for solving 2D steady incompressible Navier-Stokes equations using TensorFlow. Designed for lid-driven cavity flow simulation.

## Table of Contents
- [Mathematical Formulation](#mathematical-formulation)
- [Network Architecture](#network-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results Visualization](#results-visualization)
- [References](#references)

## Mathematical Formulation

### Governing Equations
**Continuity equation**:
$$\nabla \cdot \mathbf{u} = 0 \quad \Rightarrow \quad \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**Momentum equations**:
[
\begin{cases}
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \\
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
\end{cases}
]

**Stream function formulation** (automatically satisfies continuity):
$$
u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}
$$

### Boundary Conditions (Lid-Driven Cavity)
| Boundary        | ψ      | u         | v     |
|-----------------|--------|-----------|-------|
| Top wall (y=1)  | 0      | u₀        | 0     |
| Other walls     | 0      | 0         | 0     |

## Network Architecture

```python
# Neural Network Structure
Network(
  layers=[32, 16, 16, 32],  # Hidden layers
  activation='swish',        # Activation function
  num_outputs=2              # ψ and p
)

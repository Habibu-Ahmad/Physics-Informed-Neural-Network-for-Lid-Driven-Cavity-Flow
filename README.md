# Physics-Informed Neural Network (PINN) for Lid-Driven Cavity Flow

**A mesh-free solver for the incompressible Navier-Stokes equations using deep learning**

## 📌 Overview
This project implements a **Physics-Informed Neural Network (PINN)** to simulate 2D steady-state **lid-driven cavity flow**, a classic computational fluid dynamics (CFD) benchmark. The PINN solves the Navier-Stokes equations without traditional discretization methods using automatic differentiation to enforce physics constraints.

**Key Features:**
- Solves incompressible Navier-Stokes equations for ψ (stream function) and p (pressure)
- Enforces boundary conditions (no-slip walls, moving lid) via physics-informed loss
- Uses L-BFGS optimization for training
- Validates against Ghia et al.'s benchmark data (Re=100)

## 🧮 Governing Equations
The PINN solves these steady 2D flow equations:

### Continuity (Mass Conservation):
```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
\begin{aligned}
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} &= -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \\
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} &= -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
\end{aligned}

u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}
```math



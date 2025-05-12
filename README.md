# Physics-Informed Neural Network (PINN) for Lid-Driven Cavity Flow  
**A mesh-free solver for the incompressible Navier-Stokes equations using deep learning.**  

##  Overview  
This project implements a **Physics-Informed Neural Network (PINN)** to simulate 2D steady-state **lid-driven cavity flow**, a classic computational fluid dynamics (CFD) benchmark. The PINN solves the Navier-Stokes equations *without traditional discretization methods* (e.g., finite differences), using automatic differentiation to enforce physics constraints directly in the neural network's loss function.  

**Key Features:**  
- Solves the **incompressible Navier-Stokes equations** for stream function (ψ) and pressure (p).  
- Enforces **boundary conditions** (no-slip walls, moving lid) via physics-informed loss terms.  
- Uses **L-BFGS optimization** for training, avoiding stochastic gradient descent.  
- Validates results against Ghia et al.'s benchmark data (Re=100).  

##  Governing Equations  
The PINN solves the following equations for steady 2D flow:  

1. **Continuity (Incompressibility):**  
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad \text{(Automatically satisfied by stream function ψ)}
$$  

3. **Momentum Equations:**  
   \[
   \begin{aligned}
   u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} &= -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \\
   u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} &= -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right),
   \end{aligned}
   \]  
   where:  
   - \(u, v\) = velocity components (derived from ψ: \(u = \partial \psi / \partial y\), \(v = -\partial \psi / \partial x\)),  
   - \(p\) = pressure,  
   - \(\rho\) = density (set to 1),  
   - \(\nu\) = kinematic viscosity (set to 0.01).  

4. **Boundary Conditions:**  
   - **Walls (bottom/left/right):** \(u = v = 0\), \(\psi = \text{constant}\).  
   - **Lid (top):** \(u = 1\) (driven velocity), \(v = 0\).  

## 🛠️ Code Structure  
| File/Class          | Description                                                                 |  
|---------------------|-----------------------------------------------------------------------------|  
| **`GradientLayer`** | Custom TensorFlow layer to compute derivatives of ψ and p via `GradientTape`. |  
| **`Network`**       | Builds the neural network (MLP) predicting ψ and p. Supports `tanh`/`swish`/`mish` activations. |  
| **`L_BFGS_B`**      | Wrapper for SciPy's L-BFGS-B optimizer to train the PINN.                   |  
| **`PINN`**          | Combines the NN and physics, computes Navier-Stokes residuals and BC losses. |  
| **`uv()`**          | Utility function to derive velocities \(u, v\) from ψ.                      |  
| **`contour()`**     | Plots predicted fields (ψ, p, u, v) with Matplotlib.                        |  

## 🚀 Usage  
1. **Install Dependencies**:  
   ```bash
   pip install tensorflow numpy scipy matplotlib

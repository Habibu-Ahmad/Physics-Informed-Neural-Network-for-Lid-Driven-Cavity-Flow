# Physics-Informed Neural Network for 2D Steady Navier–Stokes Equations

A Physics-Informed Neural Network (PINN) implementation in TensorFlow for solving the 2D steady-state Navier–Stokes equations using the **stream function–pressure formulation**. This project trains the model using the **L-BFGS-B optimizer** and visualizes the resulting **u- and v-velocity fields**.

---

##  Overview

This project aims to demonstrate how PINNs can be applied to simulate incompressible, steady-state fluid flows governed by the 2D Navier–Stokes equations. The implementation combines physics-based PDE loss terms with deep learning to approximate the stream function and pressure fields, from which the velocity components are derived.

---

##  Key Features

-  PINN model built using TensorFlow/Keras  
-  Solves the 2D steady-state incompressible Navier–Stokes equations  
-  Stream function–pressure formulation for automatic enforcement of divergence-free condition  
-  Training with L-BFGS-B optimizer (using `scipy.optimize.minimize`)  
-  Velocity field visualization (`u`, `v` components)  
-  Clean, modular, and extensible implementation  

---

##  Mathematical Formulation

The steady-state, incompressible Navier–Stokes equations are given by:

$
\begin{aligned}
u &= \frac{\partial \psi}{\partial y}, \quad
v = -\frac{\partial \psi}{\partial x} \\
0 &= u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\
0 &= u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
\end{aligned}
$

The stream function formulation automatically satisfies the continuity equation:
$
\nabla \cdot \mathbf{u} = 0
$

---

# Project Structure


## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- SciPy
- Matplotlib

You can install dependencies using:

```bash
pip install tensorflow numpy scipy matplotlib


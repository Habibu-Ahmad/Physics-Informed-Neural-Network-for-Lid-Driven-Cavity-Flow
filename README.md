# PINN-NavierStokes-2D

A Physics-Informed Neural Network (PINN) implementation in TensorFlow for solving the 2D steady-state Navier–Stokes equations using the **stream function–pressure formulation**. This project trains the model using the **L-BFGS-B optimizer** and visualizes the resulting **u- and v-velocity fields**.

---

## 🚀 Overview

This project demonstrates the use of Physics-Informed Neural Networks (PINNs) for simulating incompressible, steady-state fluid flows governed by the 2D Navier–Stokes equations. It utilizes a stream function–pressure formulation, ensuring that the velocity field automatically satisfies the incompressibility condition. The training is performed using the L-BFGS-B optimizer from `scipy`.

---

## 🧠 Key Features

- ✅ PINN model built with TensorFlow/Keras  
- ✅ Solves 2D steady-state incompressible Navier–Stokes equations  
- ✅ Stream function–pressure formulation to enforce divergence-free velocity  
- ✅ Training with L-BFGS-B optimizer  
- ✅ Velocity field visualization (u, v components)  
- ✅ Clean, modular, and extensible code  

---

## 🧮 Mathematical Formulation

We define the stream function \\( \psi(x, y) \\) and pressure \\( p(x, y) \\). The velocity field \\( \mathbf{u} = (u, v) \\) is recovered from:

\\[
u = \frac{\partial \psi}{\partial y}, \quad
v = -\frac{\partial \psi}{\partial x}
\\]

This automatically satisfies the incompressibility condition:

\\[
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
\\]

The steady-state Navier–Stokes equations become:

\\[
\begin{aligned}
0 &= u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\\\
0 &= u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
\end{aligned}
\\]

---



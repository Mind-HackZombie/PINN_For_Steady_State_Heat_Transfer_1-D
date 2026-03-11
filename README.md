# PINN for 1D Steady-State Heat Transfer

## Introduction

This repository contains an implementation of a **Physics-Informed Neural Network (PINN)** to solve the **one-dimensional steady-state heat conduction equation**.

Instead of using traditional numerical methods such as **finite difference** or **finite element methods**, this approach trains a neural network while embedding the governing physical law directly into the loss function.

The model learns the temperature distribution along a rod while respecting both the **heat equation** and the **boundary conditions**.

The implementation is written in **TensorFlow** and uses automatic differentiation to compute the derivatives required in the governing equation.

---

## Problem Description

We consider steady heat conduction in a 1-D rod of length (L).

The governing equation is

[
\frac{d^2 T(x)}{dx^2} = 0
]

where

* (T(x)) is the temperature distribution
* (x) represents the position along the rod

The spatial domain is

[
0 \le x \le L
]

In this example,

[
L = 2
]

---

## Boundary Conditions

The temperature is specified at both ends of the rod.

[
T(0) = T_1
]

[
T(L) = T_2
]

For this implementation:

* (T_1 = 30)
* (T_2 = 100)

---

## Analytical Solution

For steady-state conduction with no internal heat generation, the temperature distribution is linear:

[
T(x) = T_1 + \frac{T_2 - T_1}{L}x
]

This analytical expression is used to verify the accuracy of the PINN prediction.

---

## PINN Approach

In the PINN framework, a neural network (T_\theta(x)) is used to approximate the temperature distribution.

The training process ensures that the network satisfies:

1. The governing differential equation inside the domain
2. The boundary conditions at the ends of the rod

Automatic differentiation in TensorFlow allows us to compute derivatives of the neural network output with respect to the input.

---

### PDE Residual

The residual of the governing equation is defined as

[
R(x) = \frac{d^2 T_\theta(x)}{dx^2}
]

The model is trained so that this residual becomes close to zero.

---

### Loss Function

The loss used during training is composed of two parts.

**Physics loss**

[
L_{physics} =
\frac{1}{N}
\sum
\left(
\frac{d^2 T_\theta}{dx^2}
\right)^2
]

This enforces the governing equation at selected points inside the domain.

**Boundary loss**

[
L_{boundary} =
(T_\theta(0) - T_1)^2 +
(T_\theta(L) - T_2)^2
]

This ensures that the network predictions satisfy the boundary conditions.

The total loss is

[
L = L_{physics} + L_{boundary}
]

---

## Neural Network Architecture

The neural network used in this implementation is a fully connected feed-forward network.

Architecture used in the code:

```id="b71g1f"
[1, 20, 20, 20, 1]
```

* Input layer: spatial coordinate (x)
* Three hidden layers with **20 neurons each**
* Activation function: **tanh**
* Output layer: predicted temperature

---

## Training Details

Optimizer used:

```id="pskcbn"
Adam
```

Learning rate:

```id="cqqgfc"
0.01
```

Training epochs:

```id="y68hs8"
5000
```

Collocation points inside the domain are used to enforce the governing equation.

---

## Normalization

To improve training stability, both inputs and outputs are normalized.

Input normalization

[
x_{norm} = \frac{x}{L}
]

Output normalization

[
T_{norm} =
\frac{T - T_1}{T_2 - T_1}
]

The predicted values are converted back to the original temperature scale after training.

---

## Results

After training, the PINN model predicts the temperature distribution across the rod.

The results are compared with the analytical solution and show very good agreement.

The code also computes several error metrics including

* L2 error (RMSE)
* maximum absolute error
* relative accuracy

---

## Repository Structure

```id="w44vop"
PINN-1D-Steady-Heat
│
├── pinn_heat_equation.py
├── README.md
└── requirements.txt
```

---

## Libraries Used

The implementation relies on the following libraries:

* TensorFlow
* NumPy
* Matplotlib

---

## Possible Extensions

This simple example can be extended to more complex problems such as

* 2D heat conduction
* transient heat transfer
* convection-diffusion equations
* Navier–Stokes equations

PINNs are increasingly being used in **scientific machine learning and computational physics**.

---


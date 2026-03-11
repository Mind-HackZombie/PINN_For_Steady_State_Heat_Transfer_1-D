# Physics-Informed Neural Network for 1D Steady-State Heat Transfer

## Introduction

This repository contains a simple implementation of a **Physics-Informed Neural Network (PINN)** for solving the **one-dimensional steady-state heat conduction equation**.

Instead of solving the problem using traditional numerical techniques such as the Finite Difference Method (FDM) or Finite Element Method (FEM), the governing physical equation is incorporated directly into the neural network training process.

The model learns the temperature distribution along a rod while simultaneously satisfying the governing differential equation and the boundary conditions.

The implementation is written in **TensorFlow** and uses **automatic differentiation** to compute derivatives required in the physics constraints.



## Problem Description

We consider heat conduction in a one-dimensional rod of length **L**.

### Governing Equation

The steady-state heat conduction equation is

d²T/dx² = 0

where

* T(x) = temperature distribution along the rod
* x = spatial coordinate

The spatial domain is
0 ≤ x ≤ L


In this implementation
L = 2


## Boundary Conditions

The temperature is specified at both ends of the rod.

T(0) = T₁
T(L) = T₂


For this example

T₁ = 30
T₂ = 100


## Analytical Solution

For steady heat conduction without internal heat generation, the temperature distribution is linear.

T(x) = T₁ + (T₂ − T₁)x / L


This analytical expression is used to compare and validate the predictions of the PINN model.



## PINN Methodology

In the PINN approach, a neural network is used to approximate the function **T(x)**.

The model is trained such that:

1. The neural network satisfies the governing differential equation inside the domain
2. The predicted temperature satisfies the boundary conditions

TensorFlow's automatic differentiation is used to compute derivatives of the neural network output with respect to the input.



### PDE Residual

The residual of the governing equation is

R(x) = d²Tθ(x) / dx²


The neural network is trained so that this residual becomes close to zero throughout the domain.



### Loss Function

The training loss consists of two parts.

**Physics Loss**

This enforces the governing equation at interior collocation points.

L_physics = mean((d²Tθ/dx²)²)


**Boundary Loss**

This enforces the boundary conditions.


L_boundary = (Tθ(0) − T₁)² + (Tθ(L) − T₂)²


**Total Loss**
L_total = L_physics + L_boundary


The network parameters are optimized by minimizing the total loss.



## Neural Network Architecture

The neural network used in this implementation is a fully connected feed-forward network.

Architecture used in the code:


[1, 20, 20, 20, 1]


This corresponds to

* 1 input neuron (x)
* 3 hidden layers
* 20 neurons per hidden layer
* tanh activation function
* 1 output neuron (temperature)



## Training Setup

Optimizer used


Adam Optimizer


Learning rate

0.01


Training epochs


5000


Collocation points
50 points inside the domain


These points enforce the physics constraint of the differential equation.



## Normalization

To improve numerical stability during training, both the input and output variables are normalized.

Input normalization
x_normalized = x / L

Output normalization
T_normalized = (T − T₁) / (T₂ − T₁)


After training, the predicted values are converted back to the original temperature scale.



## Results

After training, the model predicts the temperature distribution along the rod.

The predicted solution closely matches the analytical linear solution.

The script also generates a plot comparing

* Analytical solution
* PINN prediction
* Boundary points



## Error Metrics

The following metrics are used to evaluate the model.

**L2 Error (RMSE)**


L2 Error = sqrt(mean((T_pred − T_exact)²))


**Maximum Absolute Error**


Max Error = max(|T_pred − T_exact|)


**Relative Accuracy**
Accuracy = 100 × (1 − L2_error / L2_norm_exact)




## Libraries Used

This project uses the following Python libraries

* TensorFlow
* NumPy
* Matplotlib





---

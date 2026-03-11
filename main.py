import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# === Problem parameters ===
L = 2.0          # Length of the rod
T1 = 30.0        # Temperature at x=0 (change as needed)
T2 = 100.0       # Temperature at x=L (change as needed)

# Analytical solution for comparison
def analytical_solution(x):
    return (T2 - T1) / L * x + T1

# === Normalization functions ===
def normalize_x(x):
    # Normalize x to [0, 1]
    return x / L

def denormalize_x(x_norm):
    # Convert normalized x back to original scale
    return x_norm * L

def normalize_u(u):
    # Normalize u to [0, 1] based on boundary values
    return (u - T1) / (T2 - T1)

def denormalize_u(u_norm):
    # Convert normalized u back to temperature scale
    return u_norm * (T2 - T1) + T1

# === PINN Model definition ===
class PINN(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()
        # Create hidden layers with tanh activation
        self.hidden = []
        for width in layers[:-1]:
            self.hidden.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh))
        # Output layer (no activation)
        self.out = tf.keras.layers.Dense(layers[-1], activation=None)
    def call(self, x):
        # Forward pass through the network
        z = x
        for layer in self.hidden:
            z = layer(z)
        return self.out(z)

# === Data Preparation ===
n_collocation = 50  # Number of collocation points inside the domain
# Collocation points (exclude boundaries)
x_collocation = np.linspace(0, L, n_collocation)[1:-1].reshape(-1, 1).astype(np.float32)
# Boundary points
x_boundary = np.array([[0.0], [L]], dtype=np.float32)
u_boundary = np.array([[T1], [T2]], dtype=np.float32)

# Normalize data
x_colloc_norm = normalize_x(x_collocation)
x_boundary_norm = normalize_x(x_boundary)
u_boundary_norm = normalize_u(u_boundary)

# Convert to TensorFlow tensors
x_colloc_tf = tf.convert_to_tensor(x_colloc_norm)
x_boundary_tf = tf.convert_to_tensor(x_boundary_norm)
u_boundary_tf = tf.convert_to_tensor(u_boundary_norm)

# === Model and optimizer setup ===
layers = [1, 20, 20, 20, 1]  # Network architecture: input, hidden layers, output
model = PINN(layers)
_ = model(x_boundary_tf)      # Build model by calling it once

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Adam optimizer

# === Training step function ===
def train_step():
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Physics loss: enforce u_xx = 0 at collocation points
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_colloc_tf)
            with tf.GradientTape() as tape1:
                tape1.watch(x_colloc_tf)
                u_norm = model(x_colloc_tf)  # Network prediction (normalized)
            u_x = tape1.gradient(u_norm, x_colloc_tf)  # First derivative du/dx
        u_xx = tape2.gradient(u_x, x_colloc_tf)        # Second derivative d2u/dx2
        physics_loss = tf.reduce_mean(tf.square(u_xx)) # Mean squared PDE residual
        del tape1
        del tape2

        # Boundary loss: enforce boundary conditions at x=0 and x=L
        u_b_norm = model(x_boundary_tf)  # Network prediction at boundaries (normalized)
        boundary_loss = tf.reduce_mean(tf.square(u_b_norm - u_boundary_tf))  # MSE at boundaries

        # Total loss is the sum of physics and boundary losses
        loss = physics_loss + boundary_loss
    # Compute gradients and update network weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Return losses as Python floats for easy printing
    return float(loss.numpy()), float(physics_loss.numpy()), float(boundary_loss.numpy())

# === Training loop ===
n_epochs = 5000  # Number of training epochs
for epoch in range(n_epochs):
    loss, physics_loss, boundary_loss = train_step()
    # Print progress every 500 epochs
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.6f}, Physics Loss: {physics_loss:.6f}, Boundary Loss: {boundary_loss:.6f}")

# === Evaluation ===
# Test points for evaluation and plotting
x_test = np.linspace(0, L, 100).reshape(-1, 1).astype(np.float32)
x_test_norm = normalize_x(x_test)
u_pred_norm = model(x_test_norm).numpy().flatten()      # Network prediction (normalized)
u_pred = denormalize_u(u_pred_norm)                     # Convert prediction back to temperature scale
u_exact = analytical_solution(x_test.flatten())         # Analytical solution

# === Plotting results ===
plt.figure(figsize=(8,5))
plt.plot(x_test, u_exact, 'r-', label='Analytical Solution', linewidth=2)
plt.plot(x_test, u_pred, 'b--', label='PINN Prediction', linewidth=2)
plt.scatter([0, L], [T1, T2], c='k', label='Boundary Points')
plt.xlabel('x')
plt.ylabel('Temperature u(x)')
plt.legend()
plt.title('1D Steady-State Heat Equation: PINN vs Analytical (TensorFlow, Normalized)')
plt.grid(True)
plt.show()

# === Accuracy metrics calculation ===
l2_error = np.sqrt(np.mean((u_pred - u_exact)**2))         # L2 (RMSE) error
max_error = np.max(np.abs(u_pred - u_exact))               # Max absolute error
l2_norm_exact = np.sqrt(np.mean(u_exact**2))               # L2 norm of analytical solution
accuracy_percent = 100 * (1 - l2_error / l2_norm_exact)    # Relative accuracy in percent

# === Print accuracy ===
print(f"L2 Error: {l2_error:.6e}")
print(f"Max Error: {max_error:.6e}")
print(f"Model Accuracy: {accuracy_percent:.2f}%")

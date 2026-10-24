import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Parameters
np.random.seed(42)
T = 1.0           # Terminal time
N = 200           # Time steps for SDE
dt = T / N
M = 5000          # Number of paths per starting point to approximate Expectation

mu = 0.2
sigma = 0.4

# Define a spatial grid of initial positions X_0 (the "x" in our PDE)
x_grid = np.linspace(1.0, 5.0, 20)

# 2. Analytical PDE Solution (via Feynman-Kac expectation formula)
# For f(x) = x^2, the analytical solution to the PDE at t=0 is:
# V(x, 0) = x^2 * exp((2*mu + sigma^2) * T)
pde_solution = (x_grid**2) * np.exp((2 * mu + sigma**2) * T)

# 3. Stochastic Path Estimation (The Monte Carlo Expectation)
feynman_kac_estimation = []

for x0 in x_grid:
    # Initialize M paths at this specific starting location
    X = np.ones(M) * x0
    
    # Evolve the paths forward to terminal time T
    for k in range(N):
        Z = np.random.standard_normal(M)
        X = X + mu * X * dt + sigma * X * np.sqrt(dt) * Z
    
    # Evaluate the terminal condition f(X_T) = X_T^2
    terminal_payoff = X**2
    
    # Take the Expectation (Ensemble Average)
    expectation = np.mean(terminal_payoff)
    feynman_kac_estimation.append(expectation)

# 4. Plotting to see the Duality
plt.figure(figsize=(9, 6))
plt.plot(x_grid, pde_solution, 'k-', linewidth=2.5, label='Deterministic PDE Solution $V(x,0)$')
plt.scatter(x_grid, feynman_kac_estimation, color='crimson', marker='o', s=60, 
            zorder=3, label='Feynman-Kac Expectation $\mathbb{E}[X_T^2 \mid X_0=x]$')

plt.xlabel('Initial Space / Position $x$ (at $t=0$)', fontsize=12)
plt.ylabel('Value $V(x,0)$', fontsize=12)
plt.title('The Feynman-Kac Duality: Free-Space SDE Paths Solving a Parabolic PDE', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
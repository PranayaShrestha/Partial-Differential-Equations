import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
L = 10.0          # Length of the domain
T = 5.0           # Total time
nx = 100          # Number of spatial steps
nt = 2000         # Number of time steps

dx = L / (nx - 1) # Spatial step size
dt = T / nt       # Time step size

# Wave equation parameters
c = 10.0           # Wave speed

# Heat equation parameters
alpha = 0.05     # Thermal diffusivity

# Stability criteria
if dt >= dx / c:
    raise ValueError("The time step is too large for stability in the wave equation")
if dt >= dx**2 / (2 * alpha):
    raise ValueError("The time step is too large for stability in the heat equation")

# Initial condition function
def initial_condition(x):
    return np.sin(np.pi * x / L)

# Create the grid
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
X, T_grid = np.meshgrid(x, t)

# Initialize solutions
u_wave = np.zeros((nt, nx))
u_heat = np.zeros((nt, nx))
u_wave[0, :] = initial_condition(x)
u_heat[0, :] = initial_condition(x)
u_wave_new = np.zeros(nx)
u_wave_old = np.copy(u_wave[0, :])

# Initial time step for wave equation assuming zero initial velocity
u_wave[1, 1:-1] = u_wave[0, 1:-1] + 0.5 * (c * dt / dx)**2 * (u_wave[0, 2:] - 2 * u_wave[0, 1:-1] + u_wave[0, :-2])

# Time-stepping loop
for n in range(1, nt - 1):
    # Wave equation update
    u_wave_new[1:-1] = 2 * u_wave[n, 1:-1] - u_wave_old[1:-1] + (c * dt / dx)**2 * (u_wave[n, 2:] - 2 * u_wave[n, 1:-1] + u_wave[n, :-2])
    u_wave_old = u_wave[n, :]
    u_wave[n + 1, :] = u_wave_new

    # Heat equation update
    u_heat[n + 1, 1:-1] = u_heat[n, 1:-1] + alpha * dt / dx**2 * (u_heat[n, 2:] - 2 * u_heat[n, 1:-1] + u_heat[n, :-2])

# Plotting the 3D surface plots
fig = plt.figure(figsize=(14, 6))

# Wave equation plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, T_grid, u_wave, cmap='viridis')
ax1.set_title('Wave Equation:')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('u')

# Heat equation plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, T_grid, u_heat, cmap='inferno')
ax2.set_title('Heat Equation:')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u')

plt.tight_layout()
plt.show()

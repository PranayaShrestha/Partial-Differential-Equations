import numpy as np
import matplotlib.pyplot as plt

# Parameters for both equations
L = 10.0          # Length of the domain
T = 5.0           # Total time
nx = 200          # Number of spatial steps
nt = 2000         # Number of time steps

dx = L / (nx - 1) # Spatial step size
dt = T / nt       # Time step size

# Wave equation parameters
c = 20.0           # Wave speed

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

# Initialize solutions
u_wave = initial_condition(x)
u_wave_new = np.zeros(nx)
u_wave_old = np.copy(u_wave)

u_heat = initial_condition(x)
u_heat_new = np.zeros(nx)

# Initial time step for wave equation assuming zero initial velocity
u_wave[1:-1] = u_wave[1:-1] + 0.5 * (c * dt / dx)**2 * (u_wave[2:] - 2 * u_wave[1:-1] + u_wave[:-2])

# Time-stepping loop
for n in range(1, nt):
    # Wave equation update
    u_wave_new[1:-1] = 2 * u_wave[1:-1] - u_wave_old[1:-1] + (c * dt / dx)**2 * (u_wave[2:] - 2 * u_wave[1:-1] + u_wave[:-2])
    u_wave_old, u_wave = u_wave, u_wave_new

    # Heat equation update
    u_heat_new[1:-1] = u_heat[1:-1] + alpha * dt / dx**2 * (u_heat[2:] - 2 * u_heat[1:-1] + u_heat[:-2])
    u_heat = u_heat_new

    # Plot the solution at certain time steps
    if n % (nt // 10) == 0:
        plt.plot(x, u_wave, label=f'Wave t={n*dt:.2f}', linestyle='--')
        plt.plot(x, u_heat, label=f'Heat t={n*dt:.2f}', linestyle='-')

plt.title('Wave Equation and Heat Equation Solutions')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()

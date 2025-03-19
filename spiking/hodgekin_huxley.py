import numpy as np
import matplotlib.pyplot as plt

# Constants
C_m = 1.0  # Membrane capacitance (uF/cm^2)
g_Na = 120.0  # Maximum conductances (mS/cm^2)
g_K = 36.0
g_L = 0.3
E_Na = 50.0  # Reversal potentials (mV)
E_K = -77.0
E_L = -54.4

# Time parameters
dt = 0.01  # Time step (ms)
t = np.arange(0, 100, dt)

# Applied current (uA/cm^2)
I = np.zeros(len(t))
I[1000:1500] = 10  # Apply current between 10 ms and 40 ms

# Initial conditions
V = -65.0  # Membrane potential (mV)
m = 0.05
h = 0.6
n = 0.32


# Functions for gating variables
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))


def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


# Results storage
V_store = []

# Simulation loop
for i in range(len(t)):
    # Gating variables
    m += dt * (alpha_m(V) * (1 - m) - beta_m(V) * m)
    h += dt * (alpha_h(V) * (1 - h) - beta_h(V) * h)
    n += dt * (alpha_n(V) * (1 - n) - beta_n(V) * n)

    # Channel conductances
    g_Na_t = g_Na * (m**3) * h
    g_K_t = g_K * (n**4)
    g_L_t = g_L

    # Currents
    I_Na = g_Na_t * (V - E_Na)
    I_K = g_K_t * (V - E_K)
    I_L = g_L_t * (V - E_L)

    # Voltage update (Euler method)
    dV = (I[i] - (I_Na + I_K + I_L)) / C_m
    V += dt * dV

    # Store voltage
    V_store.append(V)

# Plotting the result
plt.figure(figsize=(4, 3))
plt.plot(t, V_store)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.tight_layout()
plt.savefig("C:/Users/wmvan/Downloads/HH.png", dpi=200)

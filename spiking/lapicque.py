import numpy as np
import matplotlib.pyplot as plt

# ========== Leaky Integrate-and-Fire Model with Refractory Period ==========
# LIF Constants
C_m = 1.0  # Membrane capacitance (uF/cm^2)
R_m = 10.0  # Membrane resistance (kOhm)
tau_m = R_m * C_m  # Membrane time constant (ms)
V_rest = -65.0  # Resting potential (mV)
V_th = -55.0  # Threshold potential (mV)
V_reset = -65.0  # Reset potential (mV)
refractory_period = 5.0  # Absolute refractory period (ms)

# Time parameters
dt = 0.1  # Time step (ms)
t = np.arange(0, 100, dt)  # Simulation time (ms)

# Applied current
I = np.zeros(len(t))
I[100:1000] = 1.5  # Constant current injection between 20 ms and 80 ms

# Initial values
V_LIF = V_rest
V_LIF_store = []
last_spike_time = -np.inf  # Tracks the last spike time for enforcing refractory period
spike_times = []

# Simulation loop for LIF
for i in range(len(t)):
    # Check if neuron is out of refractory period
    if (t[i] - last_spike_time) >= refractory_period:
        # Update membrane potential
        dV = (-(V_LIF - V_rest) + R_m * I[i]) / tau_m
        V_LIF += dV * dt

        V_LIF_store.append(V_LIF)
        if V_LIF >= V_th:  # Spike condition
            print("spike")
            spike_times.append(i * dt)
            V_LIF = V_reset  # Reset potential
            last_spike_time = t[i]  # Record the time of this spike
    else:
        # During the refractory period, keep V_LIF at reset potential
        V_LIF_store.append(V_LIF)

# ========== Plotting ==========
plt.figure(figsize=(4, 3))
plt.plot(t, V_LIF_store)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.tight_layout()
plt.savefig("C:/Users/wmvan/Downloads/LIF.png", dpi=200)
# for x in spike_times:
#     plt.axvline(x, color="black")

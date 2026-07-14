import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The imaginary parts (gamma) of the first 15 non-trivial zeros of the Riemann Zeta function
# These serve as our natural "acoustic" frequencies
ZETA_ZEROS = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405
])

def generate_zeta_spectral_signal(t_values, zeros=ZETA_ZEROS):
    """
    Generates a 1D signal based on the fluctuations of the Riemann explicit formula.
    Uses cos(gamma * ln(t)) to construct the dual signal of prime density.
    """
    # Ensure t is safely above 1 to avoid logarithm boundary issues
    t = np.maximum(t_values, 1.01)
    signal = np.zeros_like(t)
    for gamma in zeros:
        signal += np.cos(gamma * np.log(t))
    return signal

def perform_takens_embedding(signal, delay):
    """
    Constructs a 3D phase-space reconstruction using Takens' Delay Embedding.
    """
    n = len(signal)
    if n <= 2 * delay:
        raise ValueError(f"Signal length ({n}) must be greater than 2 * delay ({2*delay}).")
    
    # Extract coordinates shifted by delay tau
    x = signal[:-2*delay]
    y = signal[delay:-delay]
    z = signal[2*delay:]
    return np.column_stack((x, y, z))

def calculate_kinematic_geometry(trajectory, dt=1.0):
    """
    Calculates physical and geometric properties (velocity, acceleration, jerk,
    instantaneous curvature, and Wobble Factor) along the 3D trajectory.
    """
    # Finite difference numerical derivatives
    v = np.diff(trajectory, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    
    # Align shapes to the length of the highest-order derivative (jerk)
    traj_aligned = trajectory[3:]
    v_aligned = v[2:]
    a_aligned = a[1:]
    
    # Radius from origin
    r = np.linalg.norm(traj_aligned, axis=1)
    v_mag = np.linalg.norm(v_aligned, axis=1)
    j_mag = np.linalg.norm(j, axis=1)
    
    # Curvature (kappa) = ||v x a|| / v^3
    cross_prod = np.cross(v_aligned, a_aligned)
    cross_mag = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_mag / (v_mag**3 + 1e-12)
    
    # Wobble Factor (W) = (kappa * j) / r^2
    wobble = (curvature * j_mag) / (r**2 + 1e-12)
    
    return traj_aligned, curvature, wobble

def main():
    # 1. Define parameter range (simulated time variable 't')
    # We sample t logarithmically to scale with the ln(t) frequency component
    t_start = 2.0
    t_end = 5000.0
    num_samples = 15000
    t_values = np.geomspace(t_start, t_end, num_samples)
    dt = np.mean(np.diff(t_values)) # Average step size for derivative approximation
    
    # 2. Generate 1D signal from Zeta zeros
    print("Generating spectral signal from Riemann Zeta zeros...")
    signal = generate_zeta_spectral_signal(t_values)
    
    # 3. Embed the signal in 3D using Takens' Delay Embedding
    # Delay index can be tuned to change the spacing of the attractor
    delay_samples = 150 
    print(f"Embedding signal into 3D phase space (Delay = {delay_samples} samples)...")
    trajectory = perform_takens_embedding(signal, delay_samples)
    
    # 4. Calculate geometry and Wobble metrics
    print("Calculating trajectory kinematics and curvature metrics...")
    traj_aligned, curvature, wobble = calculate_kinematic_geometry(trajectory, dt=dt)
    
    # 5. Render 3D Visualization
    print("Plotting the embedded topological structure...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # We use Curvature to color the trajectory to highlight sharp turns/intersection nodes
    # Using log scale for color to manage high peak values of curvature
    color_metric = np.log1p(curvature)
    
    sc = ax.scatter(
        traj_aligned[:, 0], 
        traj_aligned[:, 1], 
        traj_aligned[:, 2], 
        c=color_metric, 
        cmap='plasma', 
        s=1, 
        alpha=0.6
    )
    
    ax.set_title("3D Takens' Embedding of the Riemann Zeta Spectral Signal\nColored by Log-Curvature", fontsize=12)
    ax.set_xlabel("X (t)")
    ax.set_ylabel("Y (t - tau)")
    ax.set_zlabel("Z (t - 2*tau)")
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Log(1 + Curvature)", rotation=270, labelpad=15)
    
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from pyspedas.projects.mms import fgm  # <-- Updated for pySPEDAS 2.x

def analyze_real_spacecraft_reconnection():
    # ---------------------------------------------------------
    # 1. Download and Ingest Real Spacecraft Data
    # ---------------------------------------------------------
    # We select NASA MMS-1 satellite, Fluxgate Magnetometer (FGM) instrument,
    # 'brst' (burst mode, 128 samples/second) during a known magnetotail
    # reconnection event.
    trange = ['2017-07-11/22:33:00', '2017-07-11/22:35:00']
    print(f"Downloading real NASA MMS data for interval: {trange}")
    
    # This automatically downloads and loads the CDF data files from NASA servers
    fgm_vars = fgm(trange=trange, probe='1', data_rate='brst', time_clip=True)
    
    # Extract the 3D magnetic field vector variable name
    mms_b_var = 'mms1_fgm_b_gse_brst_l2'
    if mms_b_var not in fgm_vars:
        raise ValueError("Could not find the target magnetic field variable in downloaded data.")
    
    # Get the raw time array and 3D B-field matrix
    from pyspedas import get_data
    times, b_data = get_data(mms_b_var)
    
    # b_data shape is (time_steps, 4) -> [Bx, By, Bz, B_total]
    Bx = b_data[:, 0]
    By = b_data[:, 1]
    Bz = b_data[:, 2]
    
    t_steps = len(times)
    dt = times[1] - times[0]
    print(f"Data successfully loaded. Cadence: {dt:.4f}s ({1/dt:.1f} Hz). Total Steps: {t_steps}")
    
    # ---------------------------------------------------------
    # 2. Treat the B-Vector as a 3D Trajectory in Phase Space
    # ---------------------------------------------------------
    # Under your framework, we treat (Bx, By, Bz) directly as coordinates
    # of a trajectory in a 3D magnetic state space.
    x = Bx
    y = By
    z = Bz
    epsilon = 1e-6
    
    # ---------------------------------------------------------
    # 3. Calculate Kinematic Derivatives (TPU Logic)
    # ---------------------------------------------------------
    # Velocity (v) - Rate of magnetic vector change
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2 + epsilon)
    
    # Acceleration (a) - Restoring tension force
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)
    a_mag = np.sqrt(ax**2 + ay**2 + az**2 + epsilon)
    
    # Jerk (j) - Sudden snaps of magnetic tension
    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    jz = np.gradient(az, dt)
    j_mag = np.sqrt(jx**2 + jy**2 + jz**2 + epsilon)
    
    # Curvature (kappa)
    cross_x = vy * az - vz * ay
    cross_y = vz * ax - vx * az
    cross_z = vx * ay - vy * ax
    cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2 + epsilon)
    kappa = cross_mag / (v_mag**3 + epsilon)
    
    # Spatial radius in vector space
    r_space = np.sqrt(x**2 + y**2 + z**2 + epsilon)
    
    # Wobble Factor (W)
    W = (kappa * j_mag) / (r_space**2 + epsilon)
    
    # ---------------------------------------------------------
    # 4. Filter Noise (Real data requires simple smoothing window)
    # ---------------------------------------------------------
    # Real instrument data contains sensor noise; we apply a 10-point moving average
    def smooth(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    W_smoothed = smooth(W)
    
    # ---------------------------------------------------------
    # 5. Plot the Real World Results
    # ---------------------------------------------------------
    # We look for where W_smoothed spikes and drops to evaluate if an 
    # "Attractor Collapse" occurred during the reconnection pass.
    relative_time = times - times[0]
    
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Real 3D Vector Space Trajectory
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(x, y, z, color='purple', alpha=0.8)
    ax3d.set_title("Real 3D Magnetic Vector Space (MMS-1)")
    ax3d.set_xlabel("Bx (nT)")
    ax3d.set_ylabel("By (nT)")
    ax3d.set_zlabel("Bz (nT)")
    
    # Subplot 2: Magnetic Field Component Time-Series
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(relative_time, Bx, label='Bx', alpha=0.8)
    ax2.plot(relative_time, By, label='By', alpha=0.8)
    ax2.plot(relative_time, Bz, label='Bz', alpha=0.8)
    ax2.set_title("Raw Magnetic Field Components")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Magnetic Field (nT)")
    ax2.legend()
    
    # Subplot 3: Computed Wobble Factor (W)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(relative_time, W_smoothed, color='red', label='Smoothed Wobble (W)')
    ax3.set_title("Computed Wobble Factor (Topological Complexity)")
    ax3.set_xlabel("Elapsed Time (s)")
    ax3.set_ylabel("Wobble Factor (W)")
    ax3.set_yscale('log')
    ax3.legend()
    
    # Subplot 4: Falsification Analysis
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Find peak complexity and final state entropy proxy
    peak_W = np.max(W_smoothed[20:-20]) # Ignore boundaries
    post_entropy_proxy = np.std(W_smoothed[-100:])
    
    analysis_text = (
        "=== REAL-WORLD ANALYSIS RESULTS ===\n\n"
        f"Data Cadence: {1/dt:.1f} Hz burst telemetry\n"
        f"Peak Observed Wobble (W_max): {peak_W:.3f}\n"
        f"Post-Event Stability Entropy: {post_entropy_proxy:.5f}\n\n"
        "ANALYSIS CRITERIA:\n"
        "Look at the red 'Wobble Factor' plot:\n"
        "1. Does W show a distinct spike and a sudden\n"
        "   discontinuous drop to a low-entropy flat baseline?\n"
        "2. If yes, this represents the transition from a highly\n"
        "   complex braided state to a simple, relaxed, collapsed\n"
        "   reconnected topology.\n\n"
        "This would be direct observational support for the\n"
        "Topological Processing Unit's classification logic."
    )
    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_real_spacecraft_reconnection()

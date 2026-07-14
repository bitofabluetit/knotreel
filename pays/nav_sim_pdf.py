import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Simulation Constants
DT = 0.001          # Integration time step
STEPS = 10000       # Total simulation steps
EPSILON = 1e-6      # Prevents division by zero

# Thresholds from the Topological Processing Unit IP portfolio
WOBBLE_FROZEN = 0.20  # Lower bound constraint
WOBBLE_EXTREME = 2.50 # Upper bound constraint (preventing singularity)
FRICTION_LIMIT = 12.0 # Velocity threshold for the "baulk cushion" limit

def run_simulation():
    # Initial state: Fluid parcel position and velocity
    initial_x = np.array([1.0, 0.5, 0.0])
    initial_v = np.array([2.0, -1.0, 0.5])
    
    x = initial_x.copy()
    v = initial_v.copy()
    
    # System Parameters
    gamma = 0.15                      # Entropic dissipation/drag factor
    initial_omega = np.array([0.0, 0.0, 1.5]) 
    omega = initial_omega.copy()      # Target Knot Geometry vector (angular rotation constraint)
    
    a_prev = np.array([0.0, 0.0, 0.0])
    jerk_ema = np.array([0.0, 0.0, 0.0]) # Exponential Moving Average to smooth jerk noise
    
    # Tracking lists for analysis and visualization
    positions = []
    velocities = []
    wobbles = []
    inversion_events = []             # 0 = Normal, 1 = Inversion (Upper), 2 = Perturbation (Lower)
    logs = []                        # To store console log text for the PDF report

    # Cooldown timer to prevent high-frequency sliding mode chatter (inversion deadlock)
    cooldown_counter = 0
    COOLDOWN_STEPS = 150 # Number of steps (~0.15s) to allow physics to resolve after an inversion

    print("=====================================================================")
    print("   Navier-Stokes Simulation: Coffey Topological Framework Boundary   ")
    print("=====================================================================")
    print(f"Time Step (DT): {DT} | Total Steps: {STEPS}\n")

    for step in range(STEPS):
        t = step * DT
        
        # 1. Core Modified Navier-Stokes Relation (Acceleration calculation)
        drag = -gamma * v
        magnus = np.cross(v, omega)
        a = drag + magnus
        
        # Softened central force to prevent singularity jumps at r -> 0
        r_dist = np.linalg.norm(x)
        central_force = -8.0 * x / (r_dist**2 + 0.1) 
        a += central_force
        
        # 2. Kinematic Jerk Extraction with Exponential Moving Average (EMA) smoothing
        if step > 0:
            jerk_raw = (a - a_prev) / DT
            # Smooth out step-by-step chatter using a 95/5 filter
            jerk_ema = 0.95 * jerk_ema + 0.05 * jerk_raw
        else:
            jerk_raw = np.array([0.0, 0.0, 0.0])
            jerk_ema = np.array([0.0, 0.0, 0.0])
            
        # 3. Instantaneous Curvature (kappa) with a softened denominator
        v_mag = np.linalg.norm(v)
        v_cross_a = np.cross(v, a)
        kappa = np.linalg.norm(v_cross_a) / (v_mag**3 + 0.1) 
        
        # 4. Phase-space Radial Distance (r)
        r = np.linalg.norm(x)
        
        # 5. Softened Wobble Calculation using the smoothed Jerk
        j_mag = np.linalg.norm(jerk_ema)
        W = (kappa * j_mag) / (r**2 + 0.1)
        
        event_type = 0
        
        # Decrement cooldown counter if active
        if cooldown_counter > 0:
            cooldown_counter -= 1
        
        # 6. Evaluate Boundary Constraints and Phase Inversions
        # Singularity Mitigation: Triggered ONLY when not in a cooldown state
        if (W > WOBBLE_EXTREME or v_mag > FRICTION_LIMIT) and cooldown_counter == 0:
            event_type = 1
            
            # "Baulk Cushion" Inversion: Reflect velocity and dissipate momentum
            v = -0.5 * v 
            
            # Increase rotational helicity, capped to prevent infinite energy pump
            omega[2] = min(50.0, omega[2] + 5.0) 
            
            # Activate the cooldown timer to allow the flow to escape the boundary layer
            cooldown_counter = COOLDOWN_STEPS
            
            # Re-evaluate physics state immediately
            drag = -gamma * v
            magnus = np.cross(v, omega)
            a = drag + magnus
            v_mag = np.linalg.norm(v)
            
            log_msg = f"Step {step:04d} | t={t:.3f}s | Inversion | Wobble: {W:.3f}, Vel: {v_mag:.2f}, Omega_z: {omega[2]:.1f}"
            print(log_msg)
            logs.append(log_msg)
            
        # Rigidity Mitigation: Lower Bound (Frozen Wobble Constraint)
        elif W < WOBBLE_FROZEN and step > 100:
            event_type = 2
            
            # Introduce localized transverse micro-perturbation to restore continuous chaotic variance
            perturbation = np.cross(v, np.array([0.1, 0.1, 0.1]))
            v += perturbation
            
            # Helicity relaxation: slowly decay back to baseline if system is stable
            omega[2] = max(1.5, omega[2] - 0.05)
            v_mag = np.linalg.norm(v)

        # Record states
        positions.append(x.copy())
        velocities.append(v_mag)
        wobbles.append(W)
        inversion_events.append(event_type)
        
        # 7. Update System States (Euler Integration)
        x += v * DT
        v += a * DT
        
        a_prev = a.copy()
        
    simulation_meta = {
        "initial_x": initial_x,
        "initial_v": initial_v,
        "gamma": gamma,
        "initial_omega": initial_omega,
        "logs": logs
    }
        
    return np.array(positions), np.array(velocities), np.array(wobbles), np.array(inversion_events), simulation_meta


def generate_pdf_report(positions, velocities, wobbles, events, meta):
    """Compiles a professional multi-page PDF document containing the simulation results."""
    print("\nCompiling PDF report...")
    pdf_filename = "Simulation_Report.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        
        # --- PAGE 1: EXECUTIVE SUMMARY & METADATA ---
        fig_summary, ax_sum = plt.subplots(figsize=(8.5, 11))
        ax_sum.axis('off')
        
        fig_summary.text(0.1, 0.92, "TOPOLOGICAL FLUID DYNAMICS REPORT", fontsize=18, fontweight='bold', color='navy')
        fig_summary.text(0.1, 0.89, "Coffey Topological Framework Integration (Anti-Chatter)", fontsize=11, fontstyle='italic', color='dimgray')
        fig_summary.text(0.1, 0.87, "_" * 68, color='navy', fontweight='bold')
        
        # Section 1: Parameters
        fig_summary.text(0.1, 0.81, "1. RUN PARAMETERS", fontsize=12, fontweight='bold', color='navy')
        param_text = (
            f"• Integration Step size (DT):    {DT}\n"
            f"• Simulation Duration (steps):   {STEPS}\n"
            f"• Dissipation Coeff. (gamma):    {meta['gamma']}\n"
            f"• Initial Coordinate Position:   {meta['initial_x']}\n"
            f"• Initial Velocity Vector:       {meta['initial_v']}\n"
            f"• Base Helicity Vector (omega):  {meta['initial_omega']}"
        )
        fig_summary.text(0.1, 0.70, param_text, fontsize=9.5, family='monospace', linespacing=1.4)
        
        # Section 2: Boundary Settings
        fig_summary.text(0.1, 0.64, "2. THEORETICAL BOUNDS", fontsize=12, fontweight='bold', color='navy')
        bounds_text = (
            f"• Frozen Wobble Limit (W_min):    {WOBBLE_FROZEN:.2f} (Restores local turbulence)\n"
            f"• Extreme Wobble Limit (W_max):   {WOBBLE_EXTREME:.2f} (Triggers spatial inversion)\n"
            f"• Friction Density Limit (V_max):  {FRICTION_LIMIT:.2f} (Limits maximum lattice density)\n"
            f"• Anti-Chatter Cooldown Steps:    {150} (Prevents sliding-mode deadlock)"
        )
        fig_summary.text(0.1, 0.53, bounds_text, fontsize=9.5, family='monospace', linespacing=1.4)
        
        # Section 3: Event Logs
        fig_summary.text(0.1, 0.47, "3. SIMULATION EVENT LOG (SINGULARITY AVOIDANCE)", fontsize=12, fontweight='bold', color='navy')
        if meta['logs']:
            visible_logs = meta['logs'][:14]
            log_display = "\n".join(visible_logs)
            if len(meta['logs']) > 14:
                log_display += f"\n... [{len(meta['logs']) - 14} additional inversion events logged]"
        else:
            log_display = "No boundary inversions triggered. Velocity stayed within stable limits."
            
        fig_summary.text(0.1, 0.18, log_display, fontsize=8.5, family='monospace', color='darkred', linespacing=1.3)
        
        fig_summary.text(0.1, 0.08, "Document compiled natively via matplotlib.backend_pdf", fontsize=8, color='gray')
        pdf.savefig(fig_summary)
        plt.close(fig_summary)
        
        # --- PAGE 2: 3D TRAJECTORY VISUALIZATION ---
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # Plot continuous path
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="teal", alpha=0.8, linewidth=1.2, label="Fluid Path")
        
        # Highlight inversion nodes
        inv_idx = np.where(events == 1)[0]
        if len(inv_idx) > 0:
            ax_3d.scatter(positions[inv_idx, 0], positions[inv_idx, 1], positions[inv_idx, 2], 
                         color="red", s=35, label="Baulk Cushion Inversions", zorder=5)
            
        ax_3d.set_title("3D Space Attractor Orbit", fontsize=14, fontweight='bold', pad=20)
        ax_3d.set_xlabel("X (Position Coordinate)")
        ax_3d.set_ylabel("Y (Velocity Coordinate)")
        ax_3d.set_zlabel("Z (Acceleration Coordinate)")
        ax_3d.legend()
        
        pdf.savefig(fig_3d)
        plt.close(fig_3d)
        
        # --- PAGE 3: KINEMATIC METRICS ---
        fig_metrics, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Subplot 1: Velocity Over Time
        ax1.plot(velocities, color="blue", alpha=0.8, linewidth=1.2, label="Velocity Magnitude")
        ax1.axhline(FRICTION_LIMIT, color="red", linestyle="--", alpha=0.7, label="Lattice Density Limit")
        ax1.set_title("Velocity Evolution Profile", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Velocity Magnitude (v)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()
        
        # Subplot 2: Wobble Factor Over Time
        ax2.plot(wobbles, color="purple", alpha=0.8, linewidth=1.2, label="Wobble Metric (W)")
        ax2.axhline(WOBBLE_EXTREME, color="red", linestyle="--", alpha=0.7, label="Extreme Upper Bound (2.5)")
        ax2.axhline(WOBBLE_FROZEN, color="cyan", linestyle="--", alpha=0.7, label="Frozen Lower Bound (0.2)")
        ax2.set_title("Dynamic Wobble Profile", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Wobble Factor (W)")
        ax2.set_xlabel("Simulation Increments (Steps)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        pdf.savefig(fig_metrics)
        plt.close(fig_metrics)

    print(f"Report compiled. File saved as '{pdf_filename}' in your directory.")


if __name__ == "__main__":
    positions, velocities, wobbles, events, meta = run_simulation()
    generate_pdf_report(positions, velocities, wobbles, events, meta)
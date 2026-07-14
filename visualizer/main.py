import os
import sys
import subprocess
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- CONFIGURATION --- PUT FILE NAME HERE
AUDIO_FILE = "YOUR_FILE_NAME"                  # Replace with your audio file (WAV, MP3, etc.)
OUTPUT_VIDEO = "EXIT_FILE_NAME.mp4" # Final rendered video name
FPS = 30                           # Video frame rate
WIND_SIZE_SEC = 1.5                # Duration of the "snake" memory window in seconds
VIS_SR = 300                       # Downsampled trajectory rate (Hz) for rendering performance
TAU_MS = 10                        # Delay embedding offset (10ms matches vocal fold fundamental specs)

def check_dependencies():
    """Verifies ffmpeg is accessible via command line."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not in your system PATH. It is required to mux audio and video.")
        sys.exit(1)

def extract_kinematics(y, sr, tau_samples):
    """
    Reconstructs 3D Phase Space and calculates numerical kinematic derivatives
    utilizing smoothing regularizers to combat noise escalation.
    """
    print("Executing Takens' Delay Embedding...")
    # Delay Coordinates
    x = y[:-2 * tau_samples]
    y_delay = y[tau_samples : -tau_samples]
    z_delay = y[2 * tau_samples :]
    
    trajectory = np.vstack((x, y_delay, z_delay))  # Shape (3, N)
    dt = 1.0 / sr

    # To counter the variance escalation problem, we apply a Savitzky-Golay 
    # smoothing filter to the raw coordinates before computing derivatives.
    print("Smoothing trajectory and extracting kinematic derivatives...")
    window_length = 101  # Must be odd
    poly_order = 3
    
    traj_smoothed = signal.savgol_filter(trajectory, window_length, poly_order, axis=1)
    
    # 1st Derivative: Velocity (v)
    v = np.gradient(traj_smoothed, axis=1) / dt
    v_smoothed = signal.savgol_filter(v, window_length, poly_order, axis=1)
    
    # 2nd Derivative: Acceleration (a)
    a = np.gradient(v_smoothed, axis=1) / dt
    a_smoothed = signal.savgol_filter(a, window_length, poly_order, axis=1)
    
    # 3rd Derivative: Jerk (j)
    j = np.gradient(a_smoothed, axis=1) / dt
    j_smoothed = signal.savgol_filter(j, window_length, poly_order, axis=1)
    
    # Instantaneous Curvature (kappa)
    cross_prod = np.cross(v_smoothed, a_smoothed, axis=0)
    norm_cross = np.linalg.norm(cross_prod, axis=0)
    norm_v = np.linalg.norm(v_smoothed, axis=0)
    kappa = norm_cross / (norm_v**3 + 1e-8)
    
    # Radial coordinate from origin
    r = np.linalg.norm(traj_smoothed, axis=0)
    
    # Jerk Norm
    norm_j = np.linalg.norm(j_smoothed, axis=0)
    
    # Psychoacoustic Decoupling / Wobble Factor (W = kappa * j / r^2)
    wobble = (kappa * norm_j) / (r**2 + 1e-4)
    wobble_smoothed = signal.savgol_filter(wobble, window_length=151, polyorder=3)
    
    # Normalize Wobble metric to [0, 1] range for visualization maps
    # Biological vocals generally sit within [0.2, 2.5]
    wobble_norm = np.clip((wobble_smoothed - 0.2) / (2.5 - 0.2), 0.0, 1.0)
    
    return traj_smoothed, wobble_norm

def compute_intensity(y, sr, length):
    """Computes a local RMS envelope of the audio representing language intensity."""
    print("Calculating local intensity envelope...")
    win_len = int(0.05 * sr)  # 50ms window
    rms = np.sqrt(np.convolve(y**2, np.ones(win_len)/win_len, mode='same'))
    rms = rms[:length]
    # Normalize to [0.0, 1.0]
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
    return rms_norm

def main():
    check_dependencies()
    
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Target audio file '{AUDIO_FILE}' not found. Please place it in this directory.")
        return

    # 1. Load & Normalize Audio
    print(f"Loading '{AUDIO_FILE}'...")
    y, sr = librosa.load(AUDIO_FILE, sr=22050)
    y = librosa.util.normalize(y)
    
    tau_samples = int((TAU_MS / 1000.0) * sr)  # 220 samples at 22050 Hz
    
    # 2. Extract Topological & Kinematic Properties
    trajectory, wobble = extract_kinematics(y, sr, tau_samples)
    intensity = compute_intensity(y, sr, trajectory.shape[1])
    
    # 3. Decimate Data for Performance Scaling
    # Reduces total points to VIS_SR Hz, ensuring smooth real-time visualizer rendering
    decim = int(sr / VIS_SR)
    traj_vis = trajectory[:, ::decim]
    wobble_vis = wobble[::decim]
    intensity_vis = intensity[::decim]
    
    num_samples_vis = traj_vis.shape[1]
    times_vis = np.arange(num_samples_vis) * decim / sr
    total_duration = times_vis[-1]
    
    print(f"File processed: {total_duration:.2f} seconds.")
    
    # 4. Set up Plot Environment
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    # Establish dynamic bounds
    lim = np.max(np.abs(traj_vis)) * 1.1
    ax.set_xlim3d([-lim, lim])
    ax.set_ylim3d([-lim, lim])
    ax.set_zlim3d([-lim, lim])
    
    # Visualizer Color Map (Inferno ranges from dark purple [soft] to hot orange/yellow [harsh])
    cmap = plt.get_cmap('inferno')
    
    # Initialize line collection with a placeholder segment to prevent empty concatenation errors in Matplotlib
    placeholder_segment = [np.zeros((2, 3))]
    lc = Line3DCollection(placeholder_segment, cmap=cmap)
    ax.add_collection3d(lc)
    
    total_frames = int(total_duration * FPS)
    temp_silent_video = "temp_silent_render.mp4"
    
    def update(frame):
        # Current playback position (seconds)
        t_current = frame / FPS
        
        # Calculate indices within active sliding window
        idx_end = np.searchsorted(times_vis, t_current)
        idx_start = np.searchsorted(times_vis, max(0.0, t_current - WIND_SIZE_SEC))
        
        if (idx_end - idx_start) < 2:
            # If there's not enough data yet, fall back to the placeholder segment to avoid errors
            lc.set_segments([np.zeros((2, 3))])
            return [lc]
        
        # Grab segments inside window
        segment_coords = traj_vis[:, idx_start:idx_end].T  # Shape: (M, 3)
        segs = np.concatenate([segment_coords[:-1, np.newaxis, :], segment_coords[1:, np.newaxis, :]], axis=1)
        
        num_segs = len(segs)
        
        # Dissipation gradient (from 0 at tail to 1 at head)
        decay = np.linspace(0.01, 1.0, num_segs)
        
        # Color: Map Wobble factor to color sequence
        wob_subset = wobble_vis[idx_start : idx_end - 1]
        colors = cmap(wob_subset)
        
        # Tail Dissipation: Fade alpha quadratically to achieve a smooth "snake tail" dissolve
        colors[:, 3] = decay ** 2.0
        
        # Thickness: Scale width based on language intensity, shaped by tail decay
        intensity_subset = intensity_vis[idx_start : idx_end - 1]
        widths = 0.5 + 12.0 * intensity_subset * (decay ** 0.5)
        
        lc.set_segments(segs)
        lc.set_color(colors)
        lc.set_linewidths(widths)
        
        # Slow, continuous camera rotation for dynamic viewing perspective
        ax.view_init(elev=22, azim=frame * 0.4)
        
        return [lc]

    print("Rendering frames to silent video container...")
    writer = FFMpegWriter(fps=FPS, bitrate=2500, metadata=dict(title='Topological Phase Space Visualizer'))
    
    ani = FuncAnimation(fig, update, frames=total_frames, blit=False)
    ani.save(temp_silent_video, writer=writer)
    plt.close(fig)
    
    # 5. Mux audio and video tracks
    print("Muxing original soundtrack into final render...")
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_silent_video,
        '-i', AUDIO_FILE,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        OUTPUT_VIDEO
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_silent_video):
        os.remove(temp_silent_video)
        
    print(f"Process complete. Output saved: '{OUTPUT_VIDEO}'")

if __name__ == "__main__":
    main()
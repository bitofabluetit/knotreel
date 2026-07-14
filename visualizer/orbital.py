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
FPS = 30                                   # Video frame rate
WIND_SIZE_SEC = 1.5                        # Duration of the "snake" memory window in seconds
VIS_SR = 300                               # Downsampled trajectory rate (Hz) for rendering performance
TAU_MS = 10                                # Delay embedding offset (10ms)

def check_dependencies():
    """Verifies ffmpeg is accessible via command line."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not in your system PATH. It is required to mux audio and video.")
        sys.exit(1)

def separate_audio_tracks(y, sr):
    """
    Splits composite audio into Vocals, Drums, and Guitar/Melody
    using Harmonic-Percussive Source Separation and spectral filtering.
    """
    print("Performing Harmonic-Percussive Source Separation (HPSS)...")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    y_drums = y_percussive
    
    print("Isolating vocal mid-band frequencies...")
    nyq = 0.5 * sr
    b_voc, a_voc = signal.butter(4, [300 / nyq, 2800 / nyq], btype='band')
    y_vocals = signal.filtfilt(b_voc, a_voc, y_harmonic)
    
    y_guitar = y_harmonic - y_vocals
    
    y_vocals = librosa.util.normalize(y_vocals)
    y_drums = librosa.util.normalize(y_drums)
    y_guitar = librosa.util.normalize(y_guitar)
    
    return y_vocals, y_drums, y_guitar

def extract_kinematics(y, sr, tau_samples):
    """
    Reconstructs 3D Phase Space and extracts the Wobble Factor (W) 
    using numerical derivatives stabilized with Savitzky-Golay filters.
    """
    x = y[:-2 * tau_samples]
    y_delay = y[tau_samples : -tau_samples]
    z_delay = y[2 * tau_samples :]
    
    trajectory = np.vstack((x, y_delay, z_delay))  # Shape (3, N)
    dt = 1.0 / sr

    window_length = 101  # Noise mitigation smoothing window
    poly_order = 3
    
    traj_smoothed = signal.savgol_filter(trajectory, window_length, poly_order, axis=1)
    
    v = np.gradient(traj_smoothed, axis=1) / dt
    v_smoothed = signal.savgol_filter(v, window_length, poly_order, axis=1)
    
    a = np.gradient(v_smoothed, axis=1) / dt
    a_smoothed = signal.savgol_filter(a, window_length, poly_order, axis=1)
    
    j = np.gradient(a_smoothed, axis=1) / dt
    j_smoothed = signal.savgol_filter(j, window_length, poly_order, axis=1)
    
    cross_prod = np.cross(v_smoothed, a_smoothed, axis=0)
    norm_cross = np.linalg.norm(cross_prod, axis=0)
    norm_v = np.linalg.norm(v_smoothed, axis=0)
    kappa = norm_cross / (norm_v**3 + 1e-8)
    
    r = np.linalg.norm(traj_smoothed, axis=0)
    norm_j = np.linalg.norm(j_smoothed, axis=0)
    
    wobble = (kappa * norm_j) / (r**2 + 1e-4)
    wobble_smoothed = signal.savgol_filter(wobble, window_length=151, polyorder=3)
    wobble_norm = np.clip((wobble_smoothed - 0.2) / (2.5 - 0.2), 0.0, 1.0)
    
    return traj_smoothed, wobble_norm

def compute_intensity(y, sr, length):
    """Calculates the rolling energy envelope for track width scaling."""
    win_len = int(0.05 * sr)
    rms = np.sqrt(np.convolve(y**2, np.ones(win_len)/win_len, mode='same'))
    rms = rms[:length]
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
    return rms_norm

def main():
    check_dependencies()
    
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Target audio file '{AUDIO_FILE}' not found.")
        return

    # 1. Load Audio
    print(f"Loading '{AUDIO_FILE}'...")
    y, sr = librosa.load(AUDIO_FILE, sr=22050)
    y_norm = librosa.util.normalize(y)
    
    tau_samples = int((TAU_MS / 1000.0) * sr)
    
    # 2. Source Separation
    y_vocals, y_drums, y_guitar = separate_audio_tracks(y_norm, sr)
    
    # 3. Kinematic Extraction
    print("Processing individual trajectory dynamics...")
    traj_voc, wob_voc = extract_kinematics(y_vocals, sr, tau_samples)
    traj_drm, wob_drm = extract_kinematics(y_drums, sr, tau_samples)
    traj_gtr, wob_gtr = extract_kinematics(y_guitar, sr, tau_samples)
    
    # 4. Energy Envelope Extraction
    int_voc = compute_intensity(y_vocals, sr, traj_voc.shape[1])
    int_drm = compute_intensity(y_drums, sr, traj_drm.shape[1])
    int_gtr = compute_intensity(y_guitar, sr, traj_gtr.shape[1])
    
    # 5. Decimation
    decim = int(sr / VIS_SR)
    
    traj_voc_v, wob_voc_v, int_voc_v = traj_voc[:, ::decim], wob_voc[::decim], int_voc[::decim]
    traj_drm_v, wob_drm_v, int_drm_v = traj_drm[:, ::decim], wob_drm[::decim], int_drm[::decim]
    traj_gtr_v, wob_gtr_v, int_gtr_v = traj_gtr[:, ::decim], wob_gtr[::decim], int_gtr[::decim]
    
    # Baseline size formatting for central combined master
    traj_voc_v = traj_voc_v * 1.3
    traj_drm_v = traj_drm_v * 0.5
    traj_gtr_v = traj_gtr_v * 0.9
    
    num_samples_vis = traj_voc_v.shape[1]
    times_vis = np.arange(num_samples_vis) * decim / sr
    total_duration = times_vis[-1]
    
    # Calculate Spatial Offsets for the satellite components
    # Find base limit size
    lim_base = max(np.max(np.abs(traj_voc_v)), np.max(np.abs(traj_drm_v)), np.max(np.abs(traj_gtr_v)))
    orbital_radius = 2.2 * lim_base
    
    # Direction offsets (120-degree intervals)
    offset_voc = np.array([[0.0], [orbital_radius], [0.0]])                                           # North (90°)
    offset_gtr = np.array([[orbital_radius * np.cos(11*np.pi/6)], [orbital_radius * np.sin(11*np.pi/6)], [0.0]]) # South-East (330°)
    offset_drm = np.array([[orbital_radius * np.cos(7*np.pi/6)], [orbital_radius * np.sin(7*np.pi/6)], [0.0]])   # South-West (210°)
    
    # Create isolated satellite paths (scaled to 75% size to visually separate from center master)
    traj_voc_sat = traj_voc_v * 0.75 + offset_voc
    traj_gtr_sat = traj_gtr_v * 0.75 + offset_gtr
    traj_drm_sat = traj_drm_v * 0.75 + offset_drm
    
    # 6. Setup Plot Canvas
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    # Set boundaries to comfortably fit center and satellites
    plot_lim = orbital_radius + (1.2 * lim_base)
    ax.set_xlim3d([-plot_lim, plot_lim])
    ax.set_ylim3d([-plot_lim, plot_lim])
    ax.set_zlim3d([-plot_lim, plot_lim])
    
    # Color Maps
    cmap_voc = plt.get_cmap('plasma')   
    cmap_drm = plt.get_cmap('inferno')  
    cmap_gtr = plt.get_cmap('viridis')  
    
    placeholder_segment = [np.zeros((2, 3))]
    
    # Declare 12 distinct collections (6 master + 6 satellite layers)
    # 1. Master Center Layer Collections
    lc_m_voc_glow = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_m_voc_core = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_m_drm_glow = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_m_drm_core = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_m_gtr_glow = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    lc_m_gtr_core = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    
    # 2. Satellite Layer Collections
    lc_s_voc_glow = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_s_voc_core = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_s_drm_glow = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_s_drm_core = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_s_gtr_glow = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    lc_s_gtr_core = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    
    # Hierarchical Scene Draw Stack Additions: Drums -> Guitar -> Vocals
    # Drums (Base Layer)
    ax.add_collection3d(lc_m_drm_glow); ax.add_collection3d(lc_m_drm_core)
    ax.add_collection3d(lc_s_drm_glow); ax.add_collection3d(lc_s_drm_core)
    
    # Guitar (Mid Layer)
    ax.add_collection3d(lc_m_gtr_glow); ax.add_collection3d(lc_m_gtr_core)
    ax.add_collection3d(lc_s_gtr_glow); ax.add_collection3d(lc_s_gtr_core)
    
    # Vocals (Top Layer)
    ax.add_collection3d(lc_m_voc_glow); ax.add_collection3d(lc_m_voc_core)
    ax.add_collection3d(lc_s_voc_glow); ax.add_collection3d(lc_s_voc_core)
    
    # On-Screen HUD Overlay Labeling
    # ax.text2D(0.04, 0.95, "Computational Acoustic Topology", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
    # ax.text2D(0.04, 0.91, "● Center: Combined Master Knot", transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
    # ax.text2D(0.04, 0.87, "● North: Vocals Satellite (Warm / Plasma)", transform=ax.transAxes, color='#f768a1', fontsize=9)
    # ax.text2D(0.04, 0.84, "● South-East: Guitar Satellite (Teal / Viridis)", transform=ax.transAxes, color='#41b6c4', fontsize=9)
    # ax.text2D(0.04, 0.81, "● South-West: Drums Satellite (Fire / Inferno)", transform=ax.transAxes, color='#fec44f', fontsize=9)
    
    total_frames = int(total_duration * FPS)
    temp_silent_video = "temp_silent_render.mp4"
    
    # Array mapping parameters for all 12 scene elements
    track_configs = [
        # --- Combined Master Center Knots ---
        {
            "glow": lc_m_voc_glow, "core": lc_m_voc_core,
            "traj": traj_voc_v, "wob": wob_voc_v, "intensity": int_voc_v, "cmap": cmap_voc,
            "base_w": 2.0, "scale_w": 18.0, "alpha_factor": 1.0, "decay_power": 1.2
        },
        {
            "glow": lc_m_gtr_glow, "core": lc_m_gtr_core,
            "traj": traj_gtr_v, "wob": wob_gtr_v, "intensity": int_gtr_v, "cmap": cmap_gtr,
            "base_w": 0.5, "scale_w": 5.0, "alpha_factor": 0.6, "decay_power": 1.8
        },
        {
            "glow": lc_m_drm_glow, "core": lc_m_drm_core,
            "traj": traj_drm_v, "wob": wob_drm_v, "intensity": int_drm_v, "cmap": cmap_drm,
            "base_w": 0.4, "scale_w": 4.0, "alpha_factor": 0.4, "decay_power": 2.2
        },
        # --- Isolated Orbiting Satellite Knots ---
        {
            "glow": lc_s_voc_glow, "core": lc_s_voc_core,
            "traj": traj_voc_sat, "wob": wob_voc_v, "intensity": int_voc_v, "cmap": cmap_voc,
            "base_w": 1.5, "scale_w": 12.0, "alpha_factor": 0.9, "decay_power": 1.2
        },
        {
            "glow": lc_s_gtr_glow, "core": lc_s_gtr_core,
            "traj": traj_gtr_sat, "wob": wob_gtr_v, "intensity": int_gtr_v, "cmap": cmap_gtr,
            "base_w": 0.4, "scale_w": 3.5, "alpha_factor": 0.5, "decay_power": 1.8
        },
        {
            "glow": lc_s_drm_glow, "core": lc_s_drm_core,
            "traj": traj_drm_sat, "wob": wob_drm_v, "intensity": int_drm_v, "cmap": cmap_drm,
            "base_w": 0.3, "scale_w": 3.0, "alpha_factor": 0.4, "decay_power": 2.2
        }
    ]
    
    def update(frame):
        t_current = frame / FPS
        
        idx_end = np.searchsorted(times_vis, t_current)
        idx_start = np.searchsorted(times_vis, max(0.0, t_current - WIND_SIZE_SEC))
        
        if (idx_end - idx_start) < 2:
            for track in track_configs:
                track["glow"].set_segments([np.zeros((2, 3))])
                track["core"].set_segments([np.zeros((2, 3))])
            return [lc_m_voc_glow, lc_m_voc_core, lc_m_gtr_glow, lc_m_gtr_core, lc_m_drm_glow, lc_m_drm_core,
                    lc_s_voc_glow, lc_s_voc_core, lc_s_gtr_glow, lc_s_gtr_core, lc_s_drm_glow, lc_s_drm_core]
        
        for track in track_configs:
            coords = track["traj"][:, idx_start:idx_end].T
            segs = np.concatenate([coords[:-1, np.newaxis, :], coords[1:, np.newaxis, :]], axis=1)
            
            num_segs = len(segs)
            decay = np.linspace(0.01, 1.0, num_segs)
            
            # Map colors and calculate decay envelope
            wob_subset = track["wob"][idx_start : idx_end - 1]
            colors = track["cmap"](wob_subset)
            
            # Sub-segment widths
            int_subset = track["intensity"][idx_start : idx_end - 1]
            core_widths = track["base_w"] + track["scale_w"] * int_subset * (decay ** 0.5)
            
            # Configure CORE
            core_colors = colors.copy()
            core_colors[:, 3] = (decay ** track["decay_power"]) * track["alpha_factor"]
            
            track["core"].set_segments(segs)
            track["core"].set_color(core_colors)
            track["core"].set_linewidths(core_widths)
            
            # Configure GLOW (4x thicker, 25% opacity)
            glow_colors = colors.copy()
            glow_colors[:, 3] = (decay ** track["decay_power"]) * (track["alpha_factor"] * 0.25)
            glow_widths = core_widths * 4.0
            
            track["glow"].set_segments(segs)
            track["glow"].set_color(glow_colors)
            track["glow"].set_linewidths(glow_widths)
            
        ax.view_init(elev=22, azim=frame * 0.4)
        
        return [lc_m_voc_glow, lc_m_voc_core, lc_m_gtr_glow, lc_m_gtr_core, lc_m_drm_glow, lc_m_drm_core,
                lc_s_voc_glow, lc_s_voc_core, lc_s_gtr_glow, lc_s_gtr_core, lc_s_drm_glow, lc_s_drm_core]

    print("Rendering layered frames to silent video container...")
    writer = FFMpegWriter(fps=FPS, bitrate=4000, metadata=dict(title='Orbital Topological Visualizer'))
    
    ani = FuncAnimation(fig, update, frames=total_frames, blit=False)
    ani.save(temp_silent_video, writer=writer)
    plt.close(fig)
    
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
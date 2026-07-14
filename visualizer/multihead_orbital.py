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

def separate_audio_negotiation(y, sr):
    """
    Implements an agentic competitive soft-mask separation loop where 
    three heads negotiate dynamic ownership of each STFT spectral bin.
    """
    print("Initializing multi-head competitive spectral separator...")
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    num_freqs, num_frames = magnitude.shape
    
    # Establish spectral templates (priors) for the three heads
    W_voc = np.exp(-((freqs - 1500) / 1000)**2)  # Vocal formant peak
    W_drm = np.exp(-((freqs - 60) / 40)**2) + np.exp(-((freqs - 8000) / 4000)**2) # Bass + Treble transient focus
    W_gtr = np.exp(-((freqs - 2500) / 2000)**2) * (1.0 - np.exp(-((freqs - 1500) / 500)**2)) # Melodic focus
    
    # Extract transient flux to assist drum head onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    if len(onset_env) < num_frames:
        onset_env = np.pad(onset_env, (0, num_frames - len(onset_env)), mode='edge')
    else:
        onset_env = onset_env[:num_frames]
    onset_env = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-8)
    
    M_voc = np.zeros_like(magnitude)
    M_drm = np.zeros_like(magnitude)
    M_gtr = np.zeros_like(magnitude)
    
    print("Negotiating spectral boundaries across heads...")
    for t in range(num_frames):
        a_voc = magnitude[:, t] * W_voc
        a_drm = magnitude[:, t] * W_drm * (1.0 + 3.0 * onset_env[t])
        a_gtr = magnitude[:, t] * W_gtr
        
        # Softmax competition loop (the cross-talk negotiation)
        temp = 0.1  # Negotiation sharpness
        stacked = np.vstack([a_voc, a_drm, a_gtr])
        stacked_exp = np.exp((stacked - np.max(stacked, axis=0)) / temp)
        softmax_masks = stacked_exp / (np.sum(stacked_exp, axis=0) + 1e-8)
        
        M_voc[:, t] = softmax_masks[0]
        M_drm[:, t] = softmax_masks[1]
        M_gtr[:, t] = softmax_masks[2]
        
    print("Reconstructing individual negotiated waveforms...")
    y_vocals = librosa.istft(D * M_voc, hop_length=512)
    y_drums = librosa.istft(D * M_drm, hop_length=512)
    y_guitar = librosa.istft(D * M_gtr, hop_length=512)
    
    # Keep sizes aligned
    min_len = min(len(y_vocals), len(y_drums), len(y_guitar), len(y))
    y_vocals, y_drums, y_guitar = y_vocals[:min_len], y_drums[:min_len], y_guitar[:min_len]
    
    # Calculate Dynamic Cross-Talk (spectral mask overlap) over time
    overlap_voc_drm = np.mean(M_voc * M_drm, axis=0)
    overlap_voc_gtr = np.mean(M_voc * M_gtr, axis=0)
    overlap_gtr_drm = np.mean(M_gtr * M_drm, axis=0)
    
    def smooth_and_norm(arr):
        smoothed = signal.savgol_filter(arr, window_length=151, polyorder=3)
        return (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-8)
        
    cross_voc_drm = smooth_and_norm(overlap_voc_drm[:min_len])
    cross_voc_gtr = smooth_and_norm(overlap_voc_gtr[:min_len])
    cross_gtr_drm = smooth_and_norm(overlap_gtr_drm[:min_len])
    
    return y_vocals, y_drums, y_guitar, cross_voc_drm, cross_voc_gtr, cross_gtr_drm

def extract_kinematics(y, sr, tau_samples):
    """Reconstructs 3D Phase Space with Savitzky-Golay filtering."""
    x = y[:-2 * tau_samples]
    y_delay = y[tau_samples : -tau_samples]
    z_delay = y[2 * tau_samples :]
    
    trajectory = np.vstack((x, y_delay, z_delay))
    dt = 1.0 / sr

    window_length = 101
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
    """Calculates rolling energy envelope."""
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
    
    # 2. Competitive Separation separation
    y_vocals, y_drums, y_guitar, c_voc_drm, c_voc_gtr, c_gtr_drm = separate_audio_negotiation(y_norm, sr)
    
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
    
    # Combined Master scaling
    traj_voc_v = traj_voc_v * 1.3
    traj_drm_v = traj_drm_v * 0.5
    traj_gtr_v = traj_gtr_v * 0.9
    
    num_samples_vis = traj_voc_v.shape[1]
    times_vis = np.arange(num_samples_vis) * decim / sr
    total_duration = times_vis[-1]
    
    # Interpolate STFT frames to match the continuous high-resolution visualization timestamps
    stft_frames = np.arange(len(c_voc_drm))
    stft_times = librosa.frames_to_time(stft_frames, sr=sr, hop_length=512)
    
    c_voc_drm_v = np.interp(times_vis, stft_times, c_voc_drm)
    c_voc_gtr_v = np.interp(times_vis, stft_times, c_voc_gtr)
    c_gtr_drm_v = np.interp(times_vis, stft_times, c_gtr_drm)
    
    # Calculate Spatial Offsets for the satellite components
    lim_base = max(np.max(np.abs(traj_voc_v)), np.max(np.abs(traj_drm_v)), np.max(np.abs(traj_gtr_v)))
    orbital_radius = 2.2 * lim_base
    
    # Direction offsets (120-degree intervals)
    offset_voc = np.array([[0.0], [orbital_radius], [0.0]])                                           # North (90°)
    offset_gtr = np.array([[orbital_radius * np.cos(11*np.pi/6)], [orbital_radius * np.sin(11*np.pi/6)], [0.0]]) # South-East (330°)
    offset_drm = np.array([[orbital_radius * np.cos(7*np.pi/6)], [orbital_radius * np.sin(7*np.pi/6)], [0.0]])   # South-West (210°)
    
    # Create isolated satellite paths
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
    lc_m_voc_glow = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_m_voc_core = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_m_drm_glow = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_m_drm_core = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_m_gtr_glow = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    lc_m_gtr_core = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    
    lc_s_voc_glow = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_s_voc_core = Line3DCollection(placeholder_segment, cmap=cmap_voc)
    lc_s_drm_glow = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_s_drm_core = Line3DCollection(placeholder_segment, cmap=cmap_drm)
    lc_s_gtr_glow = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    lc_s_gtr_core = Line3DCollection(placeholder_segment, cmap=cmap_gtr)
    
    # Add to scene in hierarchical order
    ax.add_collection3d(lc_m_drm_glow); ax.add_collection3d(lc_m_drm_core)
    ax.add_collection3d(lc_s_drm_glow); ax.add_collection3d(lc_s_drm_core)
    ax.add_collection3d(lc_m_gtr_glow); ax.add_collection3d(lc_m_gtr_core)
    ax.add_collection3d(lc_s_gtr_glow); ax.add_collection3d(lc_s_gtr_core)
    ax.add_collection3d(lc_m_voc_glow); ax.add_collection3d(lc_m_voc_core)
    ax.add_collection3d(lc_s_voc_glow); ax.add_collection3d(lc_s_voc_core)
    
    # Initialize the pulsating cross-talk communication links
    link_voc_drm, = ax.plot([], [], [], color='#d62728', lw=1, alpha=0.1, zorder=10) # Red/Magenta link
    link_voc_gtr, = ax.plot([], [], [], color='#9467bd', lw=1, alpha=0.1, zorder=10) # Purple link
    link_gtr_drm, = ax.plot([], [], [], color='#bcbd22', lw=1, alpha=0.1, zorder=10) # Yellow/Green link
    
    # HUD Overlay
    ax.text2D(0.04, 0.95, "Computational Acoustic Topology", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
    ax.text2D(0.04, 0.91, "● Center: Master Negotiated Knot", transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
    ax.text2D(0.04, 0.87, "● North: Vocals (Warm / Plasma)", transform=ax.transAxes, color='#f768a1', fontsize=9)
    ax.text2D(0.04, 0.84, "● South-East: Guitar/Melody (Teal / Viridis)", transform=ax.transAxes, color='#41b6c4', fontsize=9)
    ax.text2D(0.04, 0.81, "● South-West: Drums (Fire / Inferno)", transform=ax.transAxes, color='#fec44f', fontsize=9)
    ax.text2D(0.04, 0.77, "--- Pulsating Links: Real-time Spectral Negotiation ---", transform=ax.transAxes, color='#9e9e9e', fontsize=8, style='italic')
    
    total_frames = int(total_duration * FPS)
    temp_silent_video = "temp_silent_render.mp4"
    
    track_configs = [
        # --- Center ---
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
        # --- Satellites ---
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
            link_voc_drm.set_data([], [])
            link_voc_drm.set_3d_properties([])
            link_voc_gtr.set_data([], [])
            link_voc_gtr.set_3d_properties([])
            link_gtr_drm.set_data([], [])
            link_gtr_drm.set_3d_properties([])
            return [lc_m_voc_glow, lc_m_voc_core, lc_m_gtr_glow, lc_m_gtr_core, lc_m_drm_glow, lc_m_drm_core,
                    lc_s_voc_glow, lc_s_voc_core, lc_s_gtr_glow, lc_s_gtr_core, lc_s_drm_glow, lc_s_drm_core,
                    link_voc_drm, link_voc_gtr, link_gtr_drm]
        
        # Render the 12 dynamic lines
        for track in track_configs:
            coords = track["traj"][:, idx_start:idx_end].T
            segs = np.concatenate([coords[:-1, np.newaxis, :], coords[1:, np.newaxis, :]], axis=1)
            
            num_segs = len(segs)
            decay = np.linspace(0.01, 1.0, num_segs)
            
            wob_subset = track["wob"][idx_start : idx_end - 1]
            colors = track["cmap"](wob_subset)
            
            int_subset = track["intensity"][idx_start : idx_end - 1]
            core_widths = track["base_w"] + track["scale_w"] * int_subset * (decay ** 0.5)
            
            core_colors = colors.copy()
            core_colors[:, 3] = (decay ** track["decay_power"]) * track["alpha_factor"]
            
            track["core"].set_segments(segs)
            track["core"].set_color(core_colors)
            track["core"].set_linewidths(core_widths)
            
            glow_colors = colors.copy()
            glow_colors[:, 3] = (decay ** track["decay_power"]) * (track["alpha_factor"] * 0.25)
            glow_widths = core_widths * 4.0
            
            track["glow"].set_segments(segs)
            track["glow"].set_color(glow_colors)
            track["glow"].set_linewidths(glow_widths)
            
        # Draw dynamic cross-talk connection links between satellite snake heads
        head_voc = traj_voc_sat[:, idx_end - 1]
        head_gtr = traj_gtr_sat[:, idx_end - 1]
        head_drm = traj_drm_sat[:, idx_end - 1]
        
        # Pull dynamic negotiation overlaps
        val_voc_drm = c_voc_drm_v[idx_end - 1]
        val_voc_gtr = c_voc_gtr_v[idx_end - 1]
        val_gtr_drm = c_gtr_drm_v[idx_end - 1]
        
        # Vocals <-> Drums link
        link_voc_drm.set_data([head_voc[0], head_drm[0]], [head_voc[1], head_drm[1]])
        link_voc_drm.set_3d_properties([head_voc[2], head_drm[2]])
        link_voc_drm.set_alpha(0.05 + 0.65 * val_voc_drm)
        link_voc_drm.set_linewidth(0.5 + 4.5 * val_voc_drm)
        
        # Vocals <-> Guitar link
        link_voc_gtr.set_data([head_voc[0], head_gtr[0]], [head_voc[1], head_gtr[1]])
        link_voc_gtr.set_3d_properties([head_voc[2], head_gtr[2]])
        link_voc_gtr.set_alpha(0.05 + 0.65 * val_voc_gtr)
        link_voc_gtr.set_linewidth(0.5 + 4.5 * val_voc_gtr)
        
        # Guitar <-> Drums link
        link_gtr_drm.set_data([head_gtr[0], head_drm[0]], [head_gtr[1], head_drm[1]])
        link_gtr_drm.set_3d_properties([head_gtr[2], head_drm[2]])
        link_gtr_drm.set_alpha(0.05 + 0.65 * val_gtr_drm)
        link_gtr_drm.set_linewidth(0.5 + 4.5 * val_gtr_drm)
        
        ax.view_init(elev=22, azim=frame * 0.4)
        
        return [lc_m_voc_glow, lc_m_voc_core, lc_m_gtr_glow, lc_m_gtr_core, lc_m_drm_glow, lc_m_drm_core,
                lc_s_voc_glow, lc_s_voc_core, lc_s_gtr_glow, lc_s_gtr_core, lc_s_drm_glow, lc_s_drm_core,
                link_voc_drm, link_voc_gtr, link_gtr_drm]

    print("Rendering negotiated frames to silent video container...")
    writer = FFMpegWriter(fps=FPS, bitrate=4000, metadata=dict(title='Negotiated Multi-Head Visualizer'))
    
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
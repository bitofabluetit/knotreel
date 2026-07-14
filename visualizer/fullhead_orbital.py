import os
import sys
import subprocess
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- CONFIGURATION ---
AUDIO_FILE = "ell_ill_be_here.wav"                  # Replace with your audio file (WAV, MP3, etc.)
OUTPUT_VIDEO = "ella.mp4"       # Final rendered video name
X_HEADS = 4                                # Number of independent sound heads (x) to extract
FPS = 30                                   # Video frame rate
WIND_SIZE_SEC = 1.5                        # Duration of the "snake" window in seconds
VIS_SR = 300                               # Downsampled trajectory rate (Hz) for rendering performance
TAU_MS = 10                                # Delay embedding offset (10ms)

def check_dependencies():
    """Verifies ffmpeg is accessible via command line."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not in your system PATH. It is required to mux audio and video.")
        sys.exit(1)

def separate_audio_nmf(y, sr, x):
    """
    Dynamically decomposes the signal into x components using Non-negative Matrix Factorization (NMF).
    Calculates the activation dynamics to drive real-time cross-talk.
    """
    print(f"Executing competitive NMF decomposition into {x} independent heads...")
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(D)
    
    # Decompose using standard NMF
    import sklearn.decomposition
    nmf = sklearn.decomposition.NMF(n_components=x, init='random', random_state=42, max_iter=200)
    W, H = librosa.decompose.decompose(magnitude, transformer=nmf, sort=True)
    
    y_components = []
    for i in range(x):
        # Reconstruct magnitude spectrogram for component i
        S_i = np.outer(W[:, i], H[i, :])
        # Reconstruct complex spectrogram and invert to time-domain
        D_i = S_i * np.exp(1j * np.angle(D))
        y_i = librosa.istft(D_i, hop_length=512)
        
        # Ensure identical array length matching original sample count
        y_i = y_i[:len(y)]
        y_components.append(librosa.util.normalize(y_i))
        
    # Normalize the activations across heads to represent dynamic "ownership" of sound
    H_sum = np.sum(H, axis=0, keepdims=True) + 1e-8
    H_norm = H / H_sum  # Shape (x, frames)
    
    # Smooth the activations over time for clean link animations
    H_smoothed = []
    for i in range(x):
        smoothed = signal.savgol_filter(H_norm[i, :], window_length=151, polyorder=3)
        H_smoothed.append((smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-8))
        
    return y_components, H_smoothed

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

def modulate_brightness_hsv(rgb_array, factor_array):
    """
    Modulates the Value (brightness) channel of RGB colors based on 
    the dynamic factor array.
    """
    hsv = mcolors.rgb_to_hsv(rgb_array[:, :3])
    # Scale the Value (V) channel from 0.25 (dimmed) to 1.0 (bright glow)
    hsv[:, 2] = 0.25 + 0.75 * factor_array
    rgb_new = mcolors.hsv_to_rgb(hsv)
    
    output = np.zeros_like(rgb_array)
    output[:, :3] = rgb_new
    output[:, 3] = rgb_array[:, 3]  # Maintain original Alpha
    return output

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
    
    # 2. Dynamic Source Separation (X_HEADS)
    y_stems, H_smoothed = separate_audio_nmf(y_norm, sr, X_HEADS)
    
    # 3. Kinematic and Intensity Extraction per Head
    traj_list = []
    wob_list = []
    int_list = []
    
    for i in range(X_HEADS):
        traj, wob = extract_kinematics(y_stems[i], sr, tau_samples)
        intensity = compute_intensity(y_stems[i], sr, traj.shape[1])
        traj_list.append(traj)
        wob_list.append(wob)
        int_list.append(intensity)
        
    # 4. Decimation
    decim = int(sr / VIS_SR)
    
    traj_v_list, wob_v_list, int_v_list = [], [], []
    for i in range(X_HEADS):
        traj_v_list.append(traj_list[i][:, ::decim])
        wob_v_list.append(wob_list[i][::decim])
        int_v_list.append(int_list[i][::decim])
        
    # Map the STFT frames to actual timestamps (in seconds)
    stft_frames = np.arange(len(H_smoothed[0]))
    stft_times = librosa.frames_to_time(stft_frames, sr=sr, hop_length=512)
    
    num_samples_vis = traj_v_list[0].shape[1]
    times_vis = np.arange(num_samples_vis) * decim / sr
    total_duration = times_vis[-1]
    
    # Interpolate the STFT-based activations to match the visualization timeline exactly
    H_smoothed_v = []
    for i in range(X_HEADS):
        H_smoothed_v.append(np.interp(times_vis, stft_times, H_smoothed[i]))
        
    # Master scale calibration: Scale trajectories hierarchically.
    # Higher frequency elements (vocals) scale larger, while low frequency rhythmic layers sit tight.
    for i in range(X_HEADS):
        scale_factor = 0.5 + 0.8 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 1.0
        traj_v_list[i] = traj_v_list[i] * scale_factor
        
    # Calculate Spatial Offsets for Satellite Knots
    lim_base = max(np.max(np.abs(t)) for t in traj_v_list)
    orbital_radius = 2.2 * lim_base
    
    # Calculate evenly divided orbital angles starting due North (90 degrees / pi/2)
    angles = [np.pi/2 - i * (2 * np.pi / X_HEADS) for i in range(X_HEADS)]
    
    traj_sat_list = []
    for i in range(X_HEADS):
        offset = np.array([[orbital_radius * np.cos(angles[i])], 
                           [orbital_radius * np.sin(angles[i])], 
                           [0.0]])
        traj_sat_list.append(traj_v_list[i] * 0.75 + offset)
        
    # 5. Setup Plot Canvas
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
    
    # Dynamically select colormaps based on components
    cmaps_list = ['plasma', 'viridis', 'inferno', 'cool', 'magma', 'spring', 'autumn', 'winter']
    track_cmaps = [plt.get_cmap(cmaps_list[i % len(cmaps_list)]) for i in range(X_HEADS)]
    
    placeholder_segment = [np.zeros((2, 3))]
    
    # Create master and satellite collection arrays
    lc_m_glow_list, lc_m_core_list = [], []
    lc_s_glow_list, lc_s_core_list = [], []
    
    # Create collections and queue them by hierarchy (lowest centroids at bottom, highest at top)
    for i in range(X_HEADS):
        lc_m_glow = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        lc_m_core = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        lc_s_glow = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        lc_s_core = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        
        lc_m_glow_list.append(lc_m_glow)
        lc_m_core_list.append(lc_m_core)
        lc_s_glow_list.append(lc_s_glow)
        lc_s_core_list.append(lc_s_core)
        
        # Add to render stack sequentially
        ax.add_collection3d(lc_m_glow)
        ax.add_collection3d(lc_m_core)
        ax.add_collection3d(lc_s_glow)
        ax.add_collection3d(lc_s_core)
        
    # Initialize pulsating cross-talk communication links from the satellites to the center master
    links = []
    for i in range(X_HEADS):
        col = track_cmaps[i](0.6)
        link, = ax.plot([], [], [], color=col, lw=1, alpha=0.1, zorder=10)
        links.append(link)
        
    # Dynamic HUD Overlay
    ax.text2D(0.04, 0.95, "Computational Acoustic Topology", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
    ax.text2D(0.04, 0.91, f"● Center: Combined {X_HEADS}-Head Master", transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
    for i in range(X_HEADS):
        col_hex = mcolors.to_hex(track_cmaps[i](0.6))
        ax.text2D(0.04, 0.87 - i * 0.03, f"● Satellite {i+1} (Spectral Component #{i+1})", transform=ax.transAxes, color=col_hex, fontsize=9)
    ax.text2D(0.04, 0.87 - X_HEADS * 0.03, "--- Radial Links: Real-time Component Activation ---", transform=ax.transAxes, color='#9e9e9e', fontsize=8, style='italic')
    
    total_frames = int(total_duration * FPS)
    temp_silent_video = "temp_silent_render.mp4"
    
    # Structure configurations for all 2x components
    configs = []
    for i in range(X_HEADS):
        # Master Configuration
        configs.append({
            "glow": lc_m_glow_list[i], "core": lc_m_core_list[i],
            "traj": traj_v_list[i], "wob": wob_v_list[i], "intensity": int_v_list[i], "cmap": track_cmaps[i],
            "base_w": 0.5 + 1.5 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 2.0,
            "scale_w": 4.0 + 14.0 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 18.0,
            "alpha_factor": 0.4 + 0.6 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 1.0,
            "decay_power": 2.2 - 1.0 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 1.2
        })
        # Satellite Configuration
        configs.append({
            "glow": lc_s_glow_list[i], "core": lc_s_core_list[i],
            "traj": traj_sat_list[i], "wob": wob_v_list[i], "intensity": int_v_list[i], "cmap": track_cmaps[i],
            "base_w": 0.4 + 1.1 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 1.5,
            "scale_w": 3.0 + 9.0 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 12.0,
            "alpha_factor": 0.4 + 0.5 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 0.9,
            "decay_power": 2.2 - 1.0 * (i / (X_HEADS - 1)) if X_HEADS > 1 else 1.2
        })
        
    def update(frame):
        t_current = frame / FPS
        
        idx_end = np.searchsorted(times_vis, t_current)
        idx_start = np.searchsorted(times_vis, max(0.0, t_current - WIND_SIZE_SEC))
        
        if (idx_end - idx_start) < 2:
            for track in configs:
                track["glow"].set_segments([np.zeros((2, 3))])
                track["core"].set_segments([np.zeros((2, 3))])
            for link in links:
                link.set_data([], [])
                link.set_3d_properties([])
            return [t["glow"] for t in configs] + [t["core"] for t in configs] + links
        
        for track in configs:
            coords = track["traj"][:, idx_start:idx_end].T
            segs = np.concatenate([coords[:-1, np.newaxis, :], coords[1:, np.newaxis, :]], axis=1)
            
            num_segs = len(segs)
            decay = np.linspace(0.01, 1.0, num_segs)
            
            wob_subset = track["wob"][idx_start : idx_end - 1]
            base_colors = track["cmap"](wob_subset)
            
            # Brightness Modulation: Modulate color value/brightness via Wobble metric
            glowing_colors = modulate_brightness_hsv(base_colors, wob_subset)
            
            int_subset = track["intensity"][idx_start : idx_end - 1]
            core_widths = track["base_w"] + track["scale_w"] * int_subset * (decay ** 0.5)
            
            core_colors = glowing_colors.copy()
            core_colors[:, 3] = (decay ** track["decay_power"]) * track["alpha_factor"]
            
            track["core"].set_segments(segs)
            track["core"].set_color(core_colors)
            track["core"].set_linewidths(core_widths)
            
            glow_colors = glowing_colors.copy()
            glow_colors[:, 3] = (decay ** track["decay_power"]) * (track["alpha_factor"] * 0.25)
            glow_widths = core_widths * 4.0
            
            track["glow"].set_segments(segs)
            track["glow"].set_color(glow_colors)
            track["glow"].set_linewidths(glow_widths)
            
        # Draw dynamic cross-talk connection links from the center master to satellite heads
        for i in range(X_HEADS):
            head_sat = traj_sat_list[i][:, idx_end - 1]
            
            # Draw line extending from origin (0,0,0) to satellite head coordinates
            links[i].set_data([0.0, head_sat[0]], [0.0, head_sat[1]])
            links[i].set_3d_properties([0.0, head_sat[2]])
            
            # Map dynamic NMF activation levels directly to opacity and thickness
            act_val = H_smoothed_v[i][idx_end - 1]
            links[i].set_alpha(0.05 + 0.65 * act_val)
            links[i].set_linewidth(0.5 + 4.5 * act_val)
            
        ax.view_init(elev=22, azim=frame * 0.4)
        
        return [t["glow"] for t in configs] + [t["core"] for t in configs] + links

    print("Rendering negotiated frames to silent video container...")
    writer = FFMpegWriter(fps=FPS, bitrate=4000, metadata=dict(title='Dynamic X-Head Visualizer'))
    
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
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

# --- CONFIGURATION --- PUT FILE NAME HERE
AUDIO_FILE = "YOUR_FILE_NAME"                  # Replace with your audio file (WAV, MP3, etc.)
OUTPUT_VIDEO = "EXIT_FILE_NAME.mp4" # Final rendered video name
X_HEADS = 5                                # Set to ANY integer >= 2 (Bass = 1, Vocals = 2, others = rhythm/melodies)
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

def separate_audio_explicit_dynamic(y, sr, x):
    """
    Decomposes signal into x highly distinct stems using deterministic physical priors.
    Adapts dynamically to any value of x >= 2 while protecting Vocals and Bass isolation.
    """
    print(f"Executing dynamic competitive soft-mask separation for {x} heads...")
    
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    D_harm = librosa.stft(y_harmonic, n_fft=2048, hop_length=512)
    D_perc = librosa.stft(y_percussive, n_fft=2048, hop_length=512)
    D_mix = librosa.stft(y, n_fft=2048, hop_length=512)
    
    mag_harm, _ = librosa.magphase(D_harm)
    mag_perc, _ = librosa.magphase(D_perc)
    mag_mix, phase = librosa.magphase(D_mix)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    num_freqs, num_frames = mag_mix.shape
    
    priors = []
    # Configure dynamic acoustic prior masks
    for i in range(x):
        # i = 0: Low-frequency Bass & Sub-kick prior (< 150Hz)
        if i == 0:
            prior = np.exp(-(freqs / 120)**2)
            template = mag_mix * prior[:, np.newaxis]
        # i = 1: Vocals prior (Harmonic mid-frequency formant bandpass 300Hz - 2800Hz)
        elif i == 1:
            prior = np.exp(-((freqs - 1500) / 900)**2)
            template = mag_harm * prior[:, np.newaxis]
        # Remaining heads (2 to x-1): Spaced across the remaining percussive & harmonic spectrum
        else:
            if i % 2 == 0:
                # Percussive transient bands (Snare/High percussion)
                center_f = 200 + (sr/2 - 200) * ((i - 2) / max(1, x - 2))
                prior = np.exp(-((freqs - center_f) / (center_f * 0.5 + 50))**2)
                template = mag_perc * prior[:, np.newaxis]
            else:
                # Harmonic resonance bands (Guitars/Melody/Keys)
                center_f = 300 + (sr/2 - 300) * ((i - 2) / max(1, x - 2))
                prior = np.exp(-((freqs - center_f) / (center_f * 0.5 + 50))**2)
                template = mag_harm * prior[:, np.newaxis]
        priors.append(template)
        
    # Competitive Softmax negotiation loop
    print("Heads are dynamically negotiating spectral boundaries...")
    stacked = np.array(priors)
    temp = 0.1
    stacked_exp = np.exp((stacked - np.max(stacked, axis=0)) / temp)
    masks = stacked_exp / (np.sum(stacked_exp, axis=0) + 1e-8)
    
    y_components = []
    H_smoothed = []
    for i in range(x):
        D_i = (mag_mix * masks[i]) * np.exp(1j * np.angle(D_mix))
        y_i = librosa.istft(D_i, hop_length=512)
        y_i = y_i[:len(y)]
        y_components.append(librosa.util.normalize(y_i))
        
        activation = np.mean(masks[i], axis=0)
        smoothed = signal.savgol_filter(activation, window_length=151, polyorder=3)
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
    """Modulates color Value (brightness) based on the factor array."""
    hsv = mcolors.rgb_to_hsv(rgb_array[:, :3])
    hsv[:, 2] = 0.25 + 0.75 * factor_array
    rgb_new = mcolors.hsv_to_rgb(hsv)
    
    output = np.zeros_like(rgb_array)
    output[:, :3] = rgb_new
    output[:, 3] = rgb_array[:, 3]
    return output

def main():
    check_dependencies()
    
    if X_HEADS < 2:
        print("Error: Visualizer requires at least 2 heads (X_HEADS >= 2).")
        return
    
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Target audio file '{AUDIO_FILE}' not found.")
        return

    # 1. Load Audio
    print(f"Loading '{AUDIO_FILE}'...")
    y, sr = librosa.load(AUDIO_FILE, sr=22050)
    y_norm = librosa.util.normalize(y)
    
    tau_samples = int((TAU_MS / 1000.0) * sr)
    
    # 2. Advanced Prior-Based Separation
    y_stems, H_smoothed = separate_audio_explicit_dynamic(y_norm, sr, X_HEADS)
    
    # 3. Kinematic and Intensity Extraction
    traj_list, wob_list, int_list = [], [], []
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
        
    num_samples_vis = traj_v_list[0].shape[1]
    times_vis = np.arange(num_samples_vis) * decim / sr
    total_duration = times_vis[-1]
    
    # Scale center trajectories to map physical hierarchy spacing
    traj_v_list[0] = traj_v_list[0] * 0.5  # Bass / kick
    traj_v_list[1] = traj_v_list[1] * 1.3  # Vocals (Highest priority)
    for i in range(2, X_HEADS):
        scale_factor = 0.6 + 0.4 * (i / (X_HEADS - 1))
        traj_v_list[i] = traj_v_list[i] * scale_factor
    
    lim_base = max(np.max(np.abs(t)) for t in traj_v_list)
    orbital_radius = 2.2 * lim_base
    
    # Set circular satellite coordinates starting North
    angles = [np.pi/2 - i * (2 * np.pi / X_HEADS) for i in range(X_HEADS)]
    
    traj_sat_list = []
    for i in range(X_HEADS):
        offset = np.array([[orbital_radius * np.cos(angles[i])], 
                           [orbital_radius * np.sin(angles[i])], 
                           [0.0]])
        traj_sat_list.append(traj_v_list[i] * 0.75 + offset)
        
    # --- PRE-CALCULATE AND INERTIA-FILTER KINEMATIC CAMERA PATHS ---
    print("Stabilizing kinematic camera paths (applying temporal inertia)...")
    raw_cx = np.zeros(num_samples_vis)
    raw_cy = np.zeros(num_samples_vis)
    raw_cz = np.zeros(num_samples_vis)
    raw_lim = np.zeros(num_samples_vis)
    raw_elev = np.zeros(num_samples_vis)
    raw_azim = np.zeros(num_samples_vis)
    
    for k in range(num_samples_vis):
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        total_w = 0.0
        for i in range(X_HEADS):
            head_coord = traj_sat_list[i][:, k]
            weight = int_v_list[i][k] + 1e-4
            sum_x += head_coord[0] * weight
            sum_y += head_coord[1] * weight
            sum_z += head_coord[2] * weight
            total_w += weight
            
        raw_cx[k] = sum_x / total_w
        raw_cy[k] = sum_y / total_w
        raw_cz[k] = sum_z / total_w
        
        max_int = np.max([intensity[k] for intensity in int_v_list])
        raw_lim[k] = lim_base * (1.6 - 0.7 * max_int)
        
        t_curr = times_vis[k]
        raw_elev[k] = 22.0 + 10.0 * np.cos(t_curr * 1.2) * int_v_list[0][k]
        
        f_approx = k * (FPS / VIS_SR)
        raw_azim[k] = f_approx * 0.4 + 20.0 * np.sin(t_curr * 1.5) * int_v_list[1][k]
        
    # Use a large-window Savitzky-Golay filter to smooth all high-frequency camera twitching (1.0 sec window)
    filter_win = 2101  # ~1.0 second integration window at 300Hz
    if filter_win >= num_samples_vis:
        filter_win = (num_samples_vis // 2) * 2 - 1
        if filter_win < 3:
            filter_win = 3
            
    if filter_win >= 3:
        smooth_cx = signal.savgol_filter(raw_cx, filter_win, polyorder=3)
        smooth_cy = signal.savgol_filter(raw_cy, filter_win, polyorder=3)
        smooth_cz = signal.savgol_filter(raw_cz, filter_win, polyorder=3)
        smooth_lim = signal.savgol_filter(raw_lim, filter_win, polyorder=3)
        smooth_elev = signal.savgol_filter(raw_elev, filter_win, polyorder=3)
        smooth_azim = signal.savgol_filter(raw_azim, filter_win, polyorder=3)
    else:
        smooth_cx, smooth_cy, smooth_cz = raw_cx, raw_cy, raw_cz
        smooth_lim, smooth_elev, smooth_azim = raw_lim, raw_elev, raw_azim

    # 5. Setup Canvas
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Hide the axes outline frame entirely, leaving a pure void
    ax.set_axis_off()
    
    plot_lim = orbital_radius + (1.2 * lim_base)
    ax.set_xlim3d([-plot_lim, plot_lim])
    ax.set_ylim3d([-plot_lim, plot_lim])
    ax.set_zlim3d([-plot_lim, plot_lim])
    
    # Colormaps
    cmaps_list = ['plasma', 'viridis', 'inferno', 'cool', 'magma', 'spring', 'autumn', 'winter']
    track_cmaps = []
    # Direct fixed assignments for priority heads: 1 -> Bass (Viridis), 2 -> Vocals (Plasma)
    for i in range(X_HEADS):
        if i == 0:
            track_cmaps.append(plt.get_cmap('viridis'))
        elif i == 1:
            track_cmaps.append(plt.get_cmap('spring'))
        else:
            track_cmaps.append(plt.get_cmap(cmaps_list[(i+1) % len(cmaps_list)]))
            
    placeholder_segment = [np.zeros((2, 3))]
    
    # Declare 12 collections (6 Master + 6 Satellites)
    lc_m_glow_list, lc_m_core_list = [], []
    lc_s_glow_list, lc_s_core_list = [], []
    
    for i in range(X_HEADS):
        lc_m_glow = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        lc_m_core = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        lc_s_glow = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        lc_s_core = Line3DCollection(placeholder_segment, cmap=track_cmaps[i])
        
        lc_m_glow_list.append(lc_m_glow)
        lc_m_core_list.append(lc_m_core)
        lc_s_glow_list.append(lc_s_glow)
        lc_s_core_list.append(lc_s_core)
        
    # Enforce Layering Hierarchy: Render all non-vocal elements first, and add Vocals (index 1) last.
    for i in range(X_HEADS):
        if i != 1:
            ax.add_collection3d(lc_m_glow_list[i])
            ax.add_collection3d(lc_m_core_list[i])
            ax.add_collection3d(lc_s_glow_list[i])
            ax.add_collection3d(lc_s_core_list[i])
            
    ax.add_collection3d(lc_m_glow_list[1])
    ax.add_collection3d(lc_m_core_list[1])
    ax.add_collection3d(lc_s_glow_list[1])
    ax.add_collection3d(lc_s_core_list[1])
        
    # Dynamic HUD Overlay Labels
    ax.text2D(0.04, 0.95, "Computational Acoustic Topology", transform=ax.transAxes, color='white', fontsize=12, fontweight='bold')
    ax.text2D(0.04, 0.91, "● Center: Master Combined Knot", transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
    
    total_frames = int(total_duration * FPS)
    temp_silent_video = "temp_silent_render.mp4"
    
    # Structure configurations
    configs = []
    for i in range(X_HEADS):
        # Master Configuration (Sleek constant widths, opacity maps to audio intensity)
        configs.append({
            "glow": lc_m_glow_list[i], "core": lc_m_core_list[i],
            "traj": traj_v_list[i], "wob": wob_v_list[i], "intensity": int_v_list[i], "cmap": track_cmaps[i],
            "base_w": 1.1 if i != 1 else 1.8,       # Vocals slightly thicker
            "alpha_factor": 1.0 if i == 1 else 0.7, # Vocals most opaque
            "decay_power": 1.2
        })
        # Satellite Configuration
        configs.append({
            "glow": lc_s_glow_list[i], "core": lc_s_core_list[i],
            "traj": traj_sat_list[i], "wob": wob_v_list[i], "intensity": int_v_list[i], "cmap": track_cmaps[i],
            "base_w": 0.8 if i != 1 else 1.3,
            "alpha_factor": 0.9 if i == 1 else 0.6,
            "decay_power": 1.2
        })
        
    def update(frame):
        t_current = frame / FPS
        
        idx_end = np.searchsorted(times_vis, t_current)
        idx_start = np.searchsorted(times_vis, max(0.0, t_current - WIND_SIZE_SEC))
        
        if (idx_end - idx_start) < 2:
            for track in configs:
                track["glow"].set_segments([np.zeros((2, 3))])
                track["core"].set_segments([np.zeros((2, 3))])
            return [t["glow"] for t in configs] + [t["core"] for t in configs]
        
        for track in configs:
            coords = track["traj"][:, idx_start:idx_end].T
            segs = np.concatenate([coords[:-1, np.newaxis, :], coords[1:, np.newaxis, :]], axis=1)
            
            num_segs = len(segs)
            decay = np.linspace(0.01, 1.0, num_segs)
            
            wob_subset = track["wob"][idx_start : idx_end - 1]
            base_colors = track["cmap"](wob_subset)
            
            # Brightness is modulated by the Wobble factor
            glowing_colors = modulate_brightness_hsv(base_colors, wob_subset)
            
            # Opacity is modulated directly by both volume intensity and tail decay
            int_subset = track["intensity"][idx_start : idx_end - 1]
            
            # CORE layer
            core_colors = glowing_colors.copy()
            core_colors[:, 3] = (decay ** track["decay_power"]) * int_subset * track["alpha_factor"]
            
            track["core"].set_segments(segs)
            track["core"].set_color(core_colors)
            track["core"].set_linewidths(track["base_w"])  # Sleek, constant width
            
            # GLOW layer (3.5x wider, 25% opacity)
            glow_colors = glowing_colors.copy()
            glow_colors[:, 3] = (decay ** track["decay_power"]) * int_subset * (track["alpha_factor"] * 0.25)
            
            track["glow"].set_segments(segs)
            track["glow"].set_color(glow_colors)
            track["glow"].set_linewidths(track["base_w"] * 3.5)
            
        # --- SMOOTH KINEMATIC CAMERA PLACEMENT ---
        # Query pre-smoothed trajectory coordinates and zoom variables
        c_x = smooth_cx[idx_end - 1]
        c_y = smooth_cy[idx_end - 1]
        c_z = smooth_cz[idx_end - 1]
        dynamic_lim = smooth_lim[idx_end - 1]
        
        elev_val = smooth_elev[idx_end - 1]
        azim_val = smooth_azim[idx_end - 1]
        
        ax.view_init(elev=elev_val, azim=azim_val)
        
        # Center camera tracking boundaries directly on stabilized coordinates
        ax.set_xlim3d([c_x - dynamic_lim, c_x + dynamic_lim])
        ax.set_ylim3d([c_y - dynamic_lim, c_y + dynamic_lim])
        ax.set_zlim3d([c_z - dynamic_lim, c_z + dynamic_lim])
        
        return [t["glow"] for t in configs] + [t["core"] for t in configs]

    print("Rendering negotiated frames to silent video container...")
    writer = FFMpegWriter(fps=FPS, bitrate=4000, metadata=dict(title='Stabilized Kinetic Void Visualizer'))
    
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
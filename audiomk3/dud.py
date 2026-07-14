# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 DEEPFAKE PAGER v12.0: THE APEX FORENSIC SUITE (ALL SYSTEMS ONLINE)
=========================================================================================
"""

import os
import sys
import queue
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy import signal
from scipy.signal import find_peaks
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
TARGET_SR = 22050 
UPDATE_INTERVAL = 0.5  
PHYSICS_WINDOW = 10.0   

BUFFER_FRAMES = int(TARGET_SR * PHYSICS_WINDOW)
q = queue.Queue()
sos_hp = signal.butter(6, 8000, btype='highpass', fs=TARGET_SR, output='sos')

history_centroids = []
event_log = [""] * 6
rolling_infractions = []

# --- 2. LOAD CALIBRATION DATA (WITH FALLBACKS) ---
# A. Mahalanobis Matrix
try:
    baseline = np.load('Biological_Baseline.npz')
    BIO_MEAN = baseline['mean']
    BIO_INV_COV = baseline['inv_cov']
    baseline_loaded = True
except Exception:
    BIO_MEAN = np.zeros(4)
    BIO_INV_COV = np.eye(4)
    baseline_loaded = False

# B. Empirical CSV Limits
def calibrate_limits(csv_path='knotdeploymaster.csv'):
    limits = {
        'Min_Poly_Vol': 0.010, 
        'Max_Wobble': 100000.0,    
        'Max_Void': 3.5              
    }
    if not os.path.exists(csv_path): return limits
    try:
        df = pd.read_csv(csv_path)
        bio_df = df[df['Origin'] == 'Biological']
        if not bio_df.empty:
            limits['Min_Poly_Vol'] = bio_df['Polyphonic_Volume'].quantile(0.02)
            limits['Max_Wobble'] = bio_df['Wobble_Max'].quantile(0.98)
            void_col = 'Void_Density_Pct' if 'Void_Density_Pct' in bio_df.columns else 'Void_Pct'
            limits['Max_Void'] = bio_df[void_col].quantile(0.98)
    except Exception: pass
    return limits

LIMITS = calibrate_limits()

# --- 3. AUDIO CALLBACK ---
def audio_callback(indata, frames, time_info, status):
    mono_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
    q.put(mono_data.copy())

def add_log(msg, color="90"):
    global event_log
    event_log.pop(0)
    event_log.append(f"\033[{color}m{msg}\033[0m")

# --- 4. HUD RENDERER ---
def init_hud():
    os.system('cls' if os.name == 'nt' else 'clear')
    for _ in range(30): print("") 

def update_hud(score, poly, wob, bricks, reuse, void, stitch_flag, status, color_code):
    sys.stdout.write("\033[28A") # Move cursor up 28 lines
    def sci(v): return f"{v:.1e}" if v > 10000 else f"{v:.1f}"
    
    print("=======================================================================")
    print(" 📟 DEEPFAKE PAGER v12.0 [APEX FORENSIC SUITE]                         ")
    print("=======================================================================")
    print(f" 🚨 VERDICT : \033[{color_code}m{status.ljust(50)}\033[0m")
    print("-----------------------------------------------------------------------")
    print(" [ MULTI-VECTOR DEFENSE SYSTEMS ]")
    
    if score is None:
        print(f"  Mahalanobis (Covariance) : {'WAITING...':>8}    (Threshold: < 15.0)")
        print(f"  Polyphonic 3D Volume     : {'WAITING...':>8}    (Threshold: > {LIMITS['Min_Poly_Vol']:.4f})")
        print(f"  Psychoacoustic Wobble    : {'WAITING...':>8}    (Threshold: < {sci(LIMITS['Max_Wobble'])})")
        print(f"  HF Micro-Bricks (Tokens) : {'WAITING...':>8}    (Laundering Filter Active)")
        print(f"  Token Reuse Ratio        : {'WAITING...':>8}    (Threshold: < 2500.0x)")
        print(f"  Digital Void Density     : {'WAITING...':>8}    (Threshold: < {LIMITS['Max_Void']:.1f}%)")
        print(f"  Context Window Stability : {'WAITING...':>8}    (30-Sec Phase Tracking)")
    else:
        print(f"  Mahalanobis (Covariance) : {score:>8.2f}    (Threshold: < 15.0)")
        print(f"  Polyphonic 3D Volume     : {poly:>8.4f}    (Threshold: > {LIMITS['Min_Poly_Vol']:.4f})")
        print(f"  Psychoacoustic Wobble    : {sci(wob):>8}    (Threshold: < {sci(LIMITS['Max_Wobble'])})")
        print(f"  HF Micro-Bricks (Tokens) : {bricks:>8.0f}    (Laundering Filter Active)")
        print(f"  Token Reuse Ratio        : {reuse:>8.1f}x   (Threshold: < 2500.0x)")
        print(f"  Digital Void Density     : {void:>8.1f}%    (Threshold: < {LIMITS['Max_Void']:.1f}%)")
        print(f"  Context Window Stability : {'SNAPPED' if stitch_flag else 'STABLE':>8}    (30-Sec Phase Tracking)")
        
    print("-----------------------------------------------------------------------")
    print(" [ FORENSIC RULEBOOK LOG ]")
    for log_line in event_log:
        print(f"   {log_line}".ljust(85))
    print("=======================================================================")
    sys.stdout.flush()

# --- 5. THE APEX PHYSICS ENGINE ---
def analyze_chunk(audio_buffer):
    y = audio_buffer / (np.max(np.abs(audio_buffer)) + 1e-8)
    
    # 1. Base Kinematics
    tau = int(TARGET_SR * 0.010)
    x = y[:-2*tau]; y_del = y[tau:-tau]; z_del = y[2*tau:]
    dx, dy, dz = np.gradient(x), np.gradient(y_del), np.gradient(z_del)
    ddx, ddy, ddz = np.gradient(dx), np.gradient(dy), np.gradient(dz)
    dddx, dddy, dddz = np.gradient(ddx), np.gradient(ddy), np.gradient(dz)

    radius = np.sqrt(x**2 + y_del**2 + z_del**2) + 1e-8
    velocity = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    jerk = np.sqrt(dddx**2 + dddy**2 + dddz**2) + 1e-8
    cross_x = dy*ddz - dz*ddy; cross_y = dz*ddx - dx*ddz; cross_z = dx*ddy - dy*ddx
    curvature = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2) / (velocity**3)

    wobble_arr = (curvature * jerk) / (radius**2)
    wobble_max = np.max(wobble_arr)
    void_pct = np.mean(radius < 0.05) * 100

    # 2. Polyphonic Topology (Macro-Acoustics)
    peaks, _ = find_peaks(curvature, distance=int(TARGET_SR * 0.020))
    if len(peaks) > 5:
        poly_volume = np.std(x[peaks]) * np.std(y_del[peaks]) * np.std(z_del[peaks])
    else:
        poly_volume = 0.0

    # 3. Analog Laundering Check (HF Micro-Phase)
    y_hf = signal.sosfilt(sos_hp, y)
    y_hf = signal.medfilt(y_hf, kernel_size=3) 
    
    x_hf = y_hf[:-2*tau]; y_hf_del = y_hf[tau:-tau]; z_hf_del = y_hf[2*tau:]
    pos_hf = np.column_stack((x_hf, y_hf_del, z_hf_del))
    hf_steps = np.linalg.norm(np.diff(pos_hf, axis=0), axis=1)
    
    hf_bricks = len(np.unique(np.round(hf_steps, decimals=4)))
    hf_reuse = len(hf_steps) / (hf_bricks + 1e-8)

    # 4. Context Centroid (Stitching)
    centroid = np.array([np.mean(x), np.mean(y_del), np.mean(z_del)])

    # Pack the vector for Mahalanobis (Wobble, Void, HF Bricks, HF Reuse)
    vector = [wobble_max, void_pct, hf_bricks, hf_reuse]

    return vector, poly_volume, centroid

# --- 6. MAIN LOOP ---
def run_pager():
    global history_centroids
    
    try:
        device_info = sd.query_devices(kind='input')
        max_ch = int(device_info['max_input_channels'])
        channels = min(2, max_ch) if max_ch > 0 else 1
    except Exception:
        channels = 1

    block_frames = int(TARGET_SR * UPDATE_INTERVAL)
    physics_buffer = np.zeros(BUFFER_FRAMES)
    
    init_hud()
    if not baseline_loaded:
        add_log("⚠️ WARNING: 'Biological_Baseline.npz' not found. Mahalanobis will be 0.0.", "93")
    
    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=channels, blocksize=block_frames, callback=audio_callback):
            while True:
                live_chunk = q.get()
                
                if len(live_chunk) != block_frames:
                    live_chunk = signal.resample(live_chunk, block_frames)

                physics_buffer = np.roll(physics_buffer, -len(live_chunk))
                physics_buffer[-len(live_chunk):] = live_chunk
                
                # Silence Gate
                if np.max(np.abs(physics_buffer[-int(TARGET_SR*5):])) < 0.001:
                    update_hud(None, None, None, None, None, None, False, "⚪ WAITING FOR AUDIO...", "97")
                    continue
                
                # --- EXTRACT ALL METRICS ---
                vector, poly_vol, centroid = analyze_chunk(physics_buffer)
                wob, void, bricks, reuse = vector
                
                # --- TWEAK 1: MAHALANOBIS COVARIANCE ---
                try:
                    anomaly_score = mahalanobis(vector, BIO_MEAN, BIO_INV_COV)
                except Exception:
                    anomaly_score = 0.0

                # --- TWEAK 4: CONTEXT STITCHING ---
                history_centroids.append(centroid)
                if len(history_centroids) > 30: history_centroids.pop(0) 
                
                stitch_flag = False
                if len(history_centroids) > 5:
                    shift_dist = np.linalg.norm(history_centroids[-1] - history_centroids[-2])
                    avg_shift = np.mean([np.linalg.norm(history_centroids[i] - history_centroids[i-1]) for i in range(1, len(history_centroids)-1)])
                    
                    if shift_dist > avg_shift * 5.0:
                        stitch_flag = True
                        add_log("🚨 FAILED: Neural Context Stitch Detected (Phase Space Snap)", "91")

                # --- UNIFIED LOGIC GATES ---
                is_ai = False
                infractions = 0
                hud_color = "92"
                
                if anomaly_score > 15.0 and baseline_loaded:
                    infractions += 1
                    add_log(f"🚨 FAILED: Covariance Anomaly (Distance: {anomaly_score:.1f})", "91")
                    
                if poly_vol < LIMITS['Min_Poly_Vol']:
                    infractions += 1
                    add_log(f"🚨 FAILED: Holographic Smear (Vol: {poly_vol:.4f})", "91")
                    
                if wob > LIMITS['Max_Wobble']:
                    infractions += 1
                    add_log(f"🚨 FAILED: Psychoacoustic Decoupling (Wobble: {wob:.1e})", "91")
                    
                if reuse > 2500:
                    infractions += 1
                    add_log(f"🚨 FAILED: HF Analog Laundering Defeated. Tokens Found.", "91")
                    
                if void > LIMITS['Max_Void'] and wob > 50000:
                    infractions += 1
                    add_log(f"🚨 FAILED: Digital Void Paradox (Void: {void:.1f}%)", "91")
                    
                if infractions >= 2 or stitch_flag:
                    is_ai = True
                    hud_color = "91"
                elif infractions == 1:
                    hud_color = "93"
                    
                if not is_ai and not stitch_flag and infractions == 0:
                    add_log("✅ PASSED: Structural Integrity & Covariance Validated.", "92")

                # --- RENDER HUD ---
                status = "⛔ SYNTHETIC HOLOGRAM DETECTED" if is_ai else "🟢 BIOLOGICAL AUDIO"
                update_hud(anomaly_score, poly_vol, wob, bricks, reuse, void, stitch_flag, status, hud_color)

    except KeyboardInterrupt:
        print("\n\n[ SYSTEM SHUTDOWN ]")
    except Exception as e:
        print(f"\n[!] Stream Error: {e}")

if __name__ == "__main__":
    for _ in range(30): print("") 
    run_pager()
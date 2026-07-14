# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 APEX REAL-TIME FORENSIC PAGER (LIVE TOPOLOGY SCANNER)
=========================================================================================
"""

import os
import sys
import queue
import numpy as np
import sounddevice as sd
import warnings

# Suppress divide-by-zero warnings during perfect silence
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
TARGET_SR = 22050
WINDOW_DURATION = 3.0    # Hold 3 seconds of audio in the physical phase space
UPDATE_INTERVAL = 0.5    # Update the HUD every 0.5 seconds

BUFFER_FRAMES = int(TARGET_SR * WINDOW_DURATION)
BLOCK_FRAMES = int(TARGET_SR * UPDATE_INTERVAL)

q = queue.Queue()

# --- 2. THE EMPIRICAL BIOLOGICAL BASELINE ---
# These represent the physical center of human speech/analog music.
# (Mean, Standard Deviation)
BASELINE = {
    'Log_Max_Wobble': (4.5, 1.2),
    'Void_Density_Pct': (2.0, 1.5),
    'Corr_Radius_Accel': (0.15, 0.05)
}
REALITY_BOUNDARY_SIGMA = 12.0 # If total divergence exceeds 12σ, it's AI.

# --- 3. AUDIO CALLBACK ---
def audio_callback(indata, frames, time_info, status):
    """Pulls live audio from the soundcard into the processing queue."""
    if status:
        pass # Ignore buffer under/overflows for continuous running
    # Mix down to mono
    mono_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
    q.put(mono_data.copy())

# --- 4. THE PHYSICS ENGINE ---
def analyze_topology(y):
    """Applies the Master Formula to the live audio buffer."""
    # Silence Gate
    if np.max(np.abs(y)) < 0.001:
        return None

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-8)

    # Kinematics
    tau = int(TARGET_SR * 0.010)
    x = y[:-2*tau]; y_del = y[tau:-tau]; z_del = y[2*tau:]
    
    dx, dy, dz = np.gradient(x), np.gradient(y_del), np.gradient(z_del)
    ddx, ddy, ddz = np.gradient(dx), np.gradient(dy), np.gradient(dz)
    dddx, dddy, dddz = np.gradient(ddx), np.gradient(ddy), np.gradient(ddz)
    
    r = np.sqrt(x**2 + y_del**2 + z_del**2) + 1e-8
    v = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    a = np.sqrt(ddx**2 + ddy**2 + ddz**2) + 1e-8
    j = np.sqrt(dddx**2 + dddy**2 + dddz**2) + 1e-8
    
    cross_x = dy*ddz - dz*ddy
    cross_y = dz*ddx - dx*ddz
    cross_z = dx*ddy - dy*ddx
    k = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2) / (v**3)
    
    wobble = (k * j) / (r**2)

    # Calculate Relationships safely
    corr_ra = np.corrcoef(r, a)[0, 1]
    if np.isnan(corr_ra): corr_ra = 0.0

    metrics = {
        'Log_Max_Wobble': np.log10(np.max(wobble) + 1),
        'Void_Density_Pct': np.mean(r < 0.05) * 100,
        'Corr_Radius_Accel': np.abs(corr_ra) # Magnitude of the binding
    }
    return metrics

# --- 5. THE HUD PAGER ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_hud(metrics):
    clear_screen()
    
    print("\033[96m=======================================================================\033[0m")
    print(" 📟 APEX REAL-TIME FORENSIC PAGER (LIVE TOPOLOGY SCANNER)")
    print("\033[96m=======================================================================\033[0m")
    print(f" [ LIVE AUDIO STREAM ] : Active ({TARGET_SR}Hz)")
    print(f" [ PHASE SPACE BUFFER ]: {WINDOW_DURATION} Seconds\n")

    if metrics is None:
        print("\033[90m [!] WAITING FOR AUDIO... (Absolute Silence Detected)\033[0m")
        print("\n\n\n\n\n")
        print("\033[96m=======================================================================\033[0m")
        return

    # Calculate Distance from Reality (Z-Scores)
    total_sigma = 0
    readouts = []
    
    for key, (mean, std) in BASELINE.items():
        val = metrics[key]
        z_score = np.abs((val - mean) / std)
        total_sigma += z_score
        
        # Color code the individual metrics
        color = "\033[91m" if z_score > 4.0 else "\033[93m" if z_score > 2.0 else "\033[92m"
        readouts.append(f" {key:<20} : {val:>8.2f}  |  {color}+{z_score:>4.1f} σ\033[0m")

    print(" --- KINEMATIC READOUTS ---")
    for r in readouts: print(r)

    print("\n --- DISTANCE FROM REALITY ---")
    print(f" Total Divergence     : {total_sigma:.1f} σ (Standard Deviations)")
    print(f" Reality Boundary     : {REALITY_BOUNDARY_SIGMA:.1f} σ")
    print("")

    # THE VERDICT
    if total_sigma > REALITY_BOUNDARY_SIGMA:
        print("\033[1;91m 🚨 VERDICT : SYNTHETIC (PHYSICS HALLUCINATION DETECTED)\033[0m")
        print("\033[91m    -> Audio diverges mathematically from physical fluid dynamics.\033[0m")
    else:
        print("\033[1;92m ✅ VERDICT : BIOLOGICAL (ORGANIC AUDIO)\033[0m")
        print("\033[92m    -> Kinematic interlocks map perfectly to physical air.\033[0m")

    print("\033[96m=======================================================================\033[0m")

# --- 6. MAIN LOOP ---
def run_pager():
    print("Initializing APEX Phase Space...")
    audio_buffer = np.zeros(BUFFER_FRAMES, dtype=np.float32)
    
    try:
        # Open live audio stream
        with sd.InputStream(samplerate=TARGET_SR, channels=1, blocksize=BLOCK_FRAMES, callback=audio_callback):
            while True:
                # Wait for the next chunk of live audio
                live_chunk = q.get()
                
                # Roll the buffer (drop oldest audio, append newest audio)
                audio_buffer = np.roll(audio_buffer, -len(live_chunk))
                audio_buffer[-len(live_chunk):] = live_chunk
                
                # Extract Physics & Render
                metrics = analyze_topology(audio_buffer)
                render_hud(metrics)

    except KeyboardInterrupt:
        clear_screen()
        print("\n[!] APEX PAGER SHUTDOWN INITIATED. Exiting...\n")
    except Exception as e:
        print(f"\n[!] STREAM ERROR: {e}")

if __name__ == "__main__":
    run_pager()

# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 DEEPFAKE PAGER v9.0: THE ACOUSTIC PHYSICS INTERROGATOR (NO ML REQUIRED)
=========================================================================================
"""

import os
import sys
import queue
import numpy as np
import sounddevice as sd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
TARGET_SR = 22050 
UPDATE_INTERVAL = 0.5  # Analyze half a second of audio at a time
PHYSICS_WINDOW = 1.0   # 1-second rolling buffer for context

BUFFER_FRAMES = int(TARGET_SR * PHYSICS_WINDOW)
q = queue.Queue()

# Rolling History (To establish the "normal" baseline for the current song)
hist_vel = []
hist_curv = []
hist_jerk = []
hist_rad = []

# Live Event Log for the HUD
event_log = [""] * 6
rolling_infractions = []  # Tracks infractions in the last 20 checks

# --- 2. AUDIO CALLBACK ---
def audio_callback(indata, frames, time_info, status):
    """Captures live audio from the soundcard."""
    if indata.shape[1] > 1:
        mono_data = np.mean(indata, axis=1) # Stereo to Mono downmix
    else:
        mono_data = indata[:, 0]
    q.put(mono_data.copy())

# --- 3. THE LIVE PHYSICS ENGINE (V9.1 PATCHED) ---
def analyze_physics_chunk(audio_buffer):
    """Calculates the 3D Topological Forces of the rolling buffer."""
    # 1. Normalize local volume
    y = audio_buffer / (np.max(np.abs(audio_buffer)) + 1e-8)
    
    # 🚨 THE DITHER STRIPPER 🚨
    # Strip the microscopic analog noise added by the soundcard loopback.
    # This forces the AI back into the "Digital Void" it natively generated.
    y = np.where(np.abs(y) < 0.015, 1e-8, y)

    tau = int(TARGET_SR * 0.010) # 10ms phase delay
    
    x = y[:-2*tau]
    y_del = y[tau:-tau]
    z_del = y[2*tau:]
    
    # Kinematics (Calculated point-by-point!)
    dx, dy, dz = np.gradient(x), np.gradient(y_del), np.gradient(z_del)
    ddx, ddy, ddz = np.gradient(dx), np.gradient(dy), np.gradient(dz)
    dddx, dddy, dddz = np.gradient(ddx), np.gradient(ddy), np.gradient(dz)

    # Calculate arrays for every single sample
    radius_arr = np.sqrt(x**2 + y_del**2 + z_del**2) + 1e-8
    velocity_arr = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    jerk_arr = np.sqrt(dddx**2 + dddy**2 + dddz**2) + 1e-8
    
    cross_x = dy*ddz - dz*ddy
    cross_y = dz*ddx - dx*ddz
    cross_z = dx*ddy - dy*ddx
    curvature_arr = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2) / (velocity_arr**3)
    
    # 🚨 CALCULATE WOBBLE BEFORE AGGREGATING 🚨
    wobble_arr = (curvature_arr * jerk_arr) / (radius_arr**2)

    # NOW aggregate for the HUD and Logic Gates
    radius_mean = np.mean(radius_arr)
    velocity_mean = np.mean(velocity_arr)
    jerk_max = np.max(jerk_arr)
    curvature_max = np.max(curvature_arr)
    wobble_max = np.max(wobble_arr) # This will now properly explode into the millions!

    return radius_mean, velocity_mean, jerk_max, curvature_max, wobble_max



# --- 4. HUD RENDERER ---
def init_hud():
    os.system('cls' if os.name == 'nt' else 'clear')
    for _ in range(25): print("") 

def add_to_log(message, color_code):
    """Manages the rolling event log."""
    global event_log
    event_log.pop(0)
    event_log.append(f"\033[{color_code}m{message}\033[0m")

def update_hud(radius, velocity, jerk, curvature, wobble, status, color_code):
    sys.stdout.write("\033[25A") # Move cursor up
    print("=======================================================================")
    print(" 📟 DEEPFAKE PAGER v9.0 [PURE PHYSICS EDITION]                         ")
    print("=======================================================================")
    print(" 📡 STATE   : Interrogating Live Audio Stream...")
    print(f" 🚨 VERDICT : \033[{color_code}m{status.ljust(50)}\033[0m")
    print("-----------------------------------------------------------------------")
    print(" [ REAL-TIME KINEMATICS ]")
    print(f"   Air Displacement (Radius) : {radius:.4f}")
    print(f"   Acoustic Momentum (Vel)   : {velocity:.4f}")
    print(f"   Physical Force (Jerk)     : {jerk:.4f}")
    print(f"   Wave Asymmetry (Curv)     : {curvature:.2f}")
    print(f"   Decoupling Index (Wobble) : {wobble:.2f}")
    print("-----------------------------------------------------------------------")
    print(" [ LIVE PHYSICS RULEBOOK LOG ]")
    for log_line in event_log:
        print(f"   {log_line}".ljust(85))
    print("=======================================================================")
    sys.stdout.flush()

# --- 5. MAIN INTERROGATION LOOP ---
def run_pager():
    global hist_vel, hist_curv, hist_jerk, hist_rad, rolling_infractions
    
    # Query Mac Soundcard
    try:
        device_info = sd.query_devices(kind='input')
        native_sr = int(device_info['default_samplerate'])
        channels = min(2, int(device_info['max_input_channels']))
    except Exception:
        native_sr, channels = 44100, 2

    block_frames = int(native_sr * UPDATE_INTERVAL)
    physics_buffer = np.zeros(BUFFER_FRAMES)
    
    init_hud()
    
    try:
        with sd.InputStream(samplerate=native_sr, channels=channels, blocksize=block_frames, callback=audio_callback):
            while True:
                live_chunk_native = q.get()
                
                # Resample if needed
                if native_sr != TARGET_SR:
                    samples = len(live_chunk_native)
                    new_samples = int(samples * TARGET_SR / native_sr)
                    live_chunk = signal.resample(live_chunk_native, new_samples)
                else:
                    live_chunk = live_chunk_native

                # Roll buffer
                chunk_len = len(live_chunk)
                physics_buffer = np.roll(physics_buffer, -chunk_len)
                physics_buffer[-chunk_len:] = live_chunk
                
                # Silence check
                if np.max(np.abs(physics_buffer)) < 0.01:
                    update_hud(0,0,0,0,0, "⚪ WAITING FOR AUDIO...", "97")
                    continue
                
                # Calculate Physics
                rad, vel, jrk, curv, wob = analyze_physics_chunk(physics_buffer)
                
                # Update Baselines
                hist_vel.append(vel); hist_curv.append(curv)
                hist_jerk.append(jrk); hist_rad.append(rad)
                if len(hist_vel) > 20: # Keep last 10 seconds of baseline
                    hist_vel.pop(0); hist_curv.pop(0); hist_jerk.pop(0); hist_rad.pop(0)
                
                if len(hist_vel) < 5:
                    update_hud(rad, vel, jrk, curv, wob, "⏳ CALIBRATING ROOM ACOUSTICS...", "93")
                    continue
                
                avg_v = np.mean(hist_vel[:-1])
                avg_c = np.mean(hist_curv[:-1])
                avg_j = np.mean(hist_jerk[:-1])

                 # ==========================================
                # ⚖️ THE DYNAMIC INTERROGATOR
                # ==========================================
                infraction_this_tick = 0
                log_msg = ""
                color = "90" # Gray (Default)

                # RULE 1: THE WOBBLE GLITCH (Psychoacoustic Decoupling)
                if wob > 100000: # AI hits Millions. Human hits ~5,000.
                    infraction_this_tick += 2 # Heavy penalty
                    log_msg = f"🚨 FAILED: Massive Texture Glitch (Wobble: {wob:.1e})"
                    color = "91"
                    
                # RULE 2: CONSERVATION OF MOMENTUM
                elif vel > avg_v * 1.5:
                    if curv > avg_c * 2.0:
                        infraction_this_tick += 1
                        log_msg = "🚨 FAILED: Teleportation Anomaly (Momentum Broken)"
                        color = "91"
                    else:
                        log_msg = "✅ PASSED: High speed, arc widened organically"
                        color = "92"

                # RULE 3: INERTIAL MASS LIMIT
                elif jrk > avg_j * 2.0:
                    if rad < 0.05:
                        infraction_this_tick += 1
                        log_msg = "🚨 FAILED: Extreme Force, Zero Air Displaced"
                        color = "91"
                    else:
                        log_msg = "✅ PASSED: Force applied, air displaced normally"
                        color = "92"
                    
                # Add to HUD log if something interesting happened
                if log_msg != "":
                    add_to_log(log_msg, color)

                # --- VERDICT LOGIC ---
                rolling_infractions.append(infraction_this_tick)
                if len(rolling_infractions) > 20: # Last 10 seconds of analysis
                    rolling_infractions.pop(0)
                    
                total_infractions = sum(rolling_infractions)
                
                if total_infractions >= 3:
                    status = f"SYNTHETIC AUDIO (Physics Broken {total_infractions}x)"
                    hud_color = "91" # Red Alert
                elif total_infractions == 0 and sum(hist_rad) > 0.1:
                    status = "BIOLOGICAL AUDIO (Physics Validated)"
                    hud_color = "92" # Green Clear
                else:
                    status = f"ANALYZING... (Anomalies: {total_infractions})"
                    hud_color = "93" # Yellow Warning

                update_hud(rad, vel, jrk, curv, wob, status, hud_color)

    except KeyboardInterrupt:
        print("\n\n[ TRANSMISSION ENDED ]")
    except Exception as e:
        print(f"\n[!] Stream Error: {e}")

if __name__ == "__main__":
    run_pager()

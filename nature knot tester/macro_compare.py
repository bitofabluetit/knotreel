#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 APEX ADVANCED MUSICOLOGY & DYNAMIC COUPLING ENGINE
=========================================================================================
This engine executes state-space recurrence analysis, probabilistic pitch tracking,
time-lagged envelope cross-correlation, and polar pitch consonance mapping.
Outputs a cohesive two-page visual PDF report.
"""

import os
import sys
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
from scipy.ndimage import median_filter

# --- CONFIGURATION ---
TARGET_SR = 22050
NUM_HEADS = 3
TAU_MS = 10


# --- 1. AUDIO INGESTION & COUPLING SEPARATION ---
def load_and_decompose(path):
    print("⏳ [Ingestion] Loading audio and normalizing signal...")
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    
    print("🧬 [NMF] Decomposing into core structural registers...")
    stft = librosa.stft(y, n_fft=1024, hop_length=256)
    V = np.abs(stft) + 1e-9
    F, T = V.shape
    
    # Initialize templates (W) and activations (H)
    W = np.random.rand(F, NUM_HEADS) + 1e-5
    H = np.random.rand(NUM_HEADS, T) + 1e-5
    
    for _ in range(30):
        V_approx = W @ H + 1e-9
        H = H * (W.T @ (V / V_approx)) / (W.T @ np.ones((F, T)) + 1e-9)
        V_approx = W @ H + 1e-9
        W = W * ((V / V_approx) @ H.T) / (np.ones((F, T)) @ H.T + 1e-9)
        
    # Classify components by spectral centroids to find Bass and Melody registers
    freqs = np.linspace(0, TARGET_SR / 2, F)
    centroids = [np.sum(freqs * W[:, j]) / (np.sum(W[:, j]) + 1e-9) for j in range(NUM_HEADS)]
    sorted_indices = np.argsort(centroids)
    
    # Lowest centroid is Bass, Middle is Melody
    bass_idx = sorted_indices[0]
    melody_idx = sorted_indices[1]
    
    signals = {}
    for idx, label in [(bass_idx, "Bass"), (melody_idx, "Melody")]:
        mask = (W[:, [idx]] @ H[[idx], :]) / (W @ H + 1e-9)
        sig = librosa.istft(stft * mask, hop_length=256)
        
        if len(sig) < len(y):
            sig = np.pad(sig, (0, len(y) - len(sig)))
        else:
            sig = sig[:len(y)]
        signals[label] = sig
        
    return signals["Bass"], signals["Melody"], y


# --- 2. ADVANCED TOPOLOGICAL & DYNAMIC COUPLING CORE ---
def run_cross_recurrence(bass, melody, tau_samples, ds=250):
    """Computes a Cross-Recurrence Plot (CRP) between 3D delay-embeddings."""
    print("📐 [Analysis 1] Calculating 3D Cross-Recurrence phase coordinates...")
    X = np.column_stack((bass[:-2*tau_samples], bass[tau_samples:-tau_samples], bass[2*tau_samples:]))
    Y = np.column_stack((melody[:-2*tau_samples], melody[tau_samples:-tau_samples], melody[2*tau_samples:]))
    
    # Downsample coordinates to prevent giant memory allocation during cdist
    X_ds = X[::ds]
    Y_ds = Y[::ds]
    
    distances = cdist(X_ds, Y_ds, metric='euclidean')
    # Threshold recurrence at 15th percentile of distances
    threshold = np.percentile(distances, 15)
    crp_matrix = (distances < threshold).astype(int)
    return crp_matrix


def run_pitch_contours(bass, melody):
    """Extracts continuous F0 pitch contours via the YIN algorithm."""
    print("🎼 [Analysis 2] Extracting fundamental pitch contours (YIN)...")
    # Clean silence/low-level noise to avoid unstable pitch tracking
    rms_bass = librosa.feature.rms(y=bass, frame_length=1024, hop_length=256)[0]
    rms_melody = librosa.feature.rms(y=melody, frame_length=1024, hop_length=256)[0]
    
    # F0 estimation bounds adjusted for typical Bass and Melody instrument registers
    f0_bass = librosa.yin(y=bass, sr=TARGET_SR, fmin=40, fmax=180, hop_length=256)
    f0_melody = librosa.yin(y=melody, sr=TARGET_SR, fmin=150, fmax=650, hop_length=256)
    
    # Apply median filtering to smooth out octave tracking errors
    f0_bass = median_filter(f0_bass, size=3)
    f0_melody = median_filter(f0_melody, size=3)
    
    # Mask estimations where the channel amplitude drops below 5% of peak
    f0_bass[rms_bass < (np.max(rms_bass) * 0.05)] = np.nan
    f0_melody[rms_melody < (np.max(rms_melody) * 0.05)] = np.nan
    
    return f0_bass, f0_melody


def run_time_lagged_envelope(bass, melody):
    """Calculates lagging vs leading relationships via RMS envelopes."""
    print("⏱️ [Analysis 3] Calculating time-lagged envelope correlations...")
    hop_len = 256
    env_bass = librosa.feature.rms(y=bass, frame_length=1024, hop_length=hop_len)[0]
    env_melody = librosa.feature.rms(y=melody, frame_length=1024, hop_length=hop_len)[0]
    
    # Zero-center normalization
    env_bass = (env_bass - np.mean(env_bass)) / (np.std(env_bass) + 1e-9)
    env_melody = (env_melody - np.mean(env_melody)) / (np.std(env_melody) + 1e-9)
    
    # Search range: +/- 250 milliseconds
    frame_rate_ms = (hop_len / TARGET_SR) * 1000.0
    max_lag_frames = int(250 / frame_rate_ms)
    
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    correlations = []
    
    for l in lags:
        if l < 0:
            val = np.corrcoef(env_bass[-l:], env_melody[:l])[0, 1]
        elif l > 0:
            val = np.corrcoef(env_bass[:-l], env_melody[l:])[0, 1]
        else:
            val = np.corrcoef(env_bass, env_melody)[0, 1]
        correlations.append(val if not np.isnan(val) else 0.0)
        
    return lags * frame_rate_ms, np.array(correlations)


def run_chroma_consonance(bass, melody):
    """Extracts Constant-Q chromagrams to map tonal consonance profiles."""
    print("🌀 [Analysis 4] Mapping polar pitch-class profiles...")
    chroma_b = librosa.feature.chroma_stft(y=bass, sr=TARGET_SR, n_fft=2048, hop_length=512)
    chroma_m = librosa.feature.chroma_stft(y=melody, sr=TARGET_SR, n_fft=2048, hop_length=512)
    
    profile_b = np.mean(chroma_b, axis=1)
    profile_m = np.mean(chroma_m, axis=1)
    
    # Peak normalize
    profile_b /= (np.max(profile_b) + 1e-9)
    profile_m /= (np.max(profile_m) + 1e-9)
    
    return profile_b, profile_m


# --- 3. GRAPHICAL REPORT LAYOUT & GENERATION ENGINE ---
def generate_musicology_report(bass, melody, original, output_pdf="apex_advanced_musicology_report.pdf"):
    print(f"🎨 [Visualization] Formatting advanced musicology PDF: {output_pdf}")
    
    tau_samples = int(TARGET_SR * (TAU_MS / 1000.0))
    crp_matrix = run_cross_recurrence(bass, melody, tau_samples)
    f0_bass, f0_melody = run_pitch_contours(bass, melody)
    lag_times, env_corrs = run_time_lagged_envelope(bass, melody)
    chroma_bass, chroma_melody = run_chroma_consonance(bass, melody)
    
    # Find peak temporal lead/lag
    peak_idx = np.argmax(env_corrs)
    peak_lag = lag_times[peak_idx]
    peak_corr = env_corrs[peak_idx]
    
    with PdfPages(output_pdf) as pdf:
        # ---------------------------------------------------------
        # PAGE 1: RECURRENCE & MELODIC PITCH TRACKING
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle("APEX Advanced Musicology Report", fontsize=15, fontweight='bold', y=0.96, color='#1a365d')
        
        # Grid structure: Page 1 consists of two major charts
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0])
        gs1.update(left=0.12, right=0.90, top=0.88, bottom=0.08, hspace=0.35)
        
        # --- SUBPLOT 1: Cross-Recurrence Plot (CRP) ---
        ax_crp = fig1.add_subplot(gs1[0])
        ax_crp.imshow(crp_matrix, cmap='Greys', origin='lower', aspect='auto', interpolation='nearest')
        ax_crp.set_title("1. Phase Space Cross-Recurrence Plot (Phase Coupling Dynamics)", fontsize=10, fontweight='bold', color='#2b6cb0')
        ax_crp.set_xlabel("Melody Orbit Timeline (Downsampled Steps)", fontsize=8)
        ax_crp.set_ylabel("Bass Orbit Timeline (Downsampled Steps)", fontsize=8)
        ax_crp.text(0.02, 0.05, "Diagonal lines indicate phase-locking (rhythmic entrainment). Clusters indicate static states.", 
                    transform=ax_crp.transAxes, fontsize=7, color='#718096', bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.8))
        
        # --- SUBPLOT 2: Fundamental Frequency (F0) Contours ---
        ax_pitch = fig1.add_subplot(gs1[1])
        times = librosa.frames_to_time(np.arange(len(f0_bass)), sr=TARGET_SR, hop_length=256)
        
        ax_pitch.plot(times, f0_melody, label='Melody Voice (F0)', color='#ff7f0e', lw=1.2)
        ax_pitch.plot(times, f0_bass, label='Bass register (F0)', color='#1f77b4', lw=1.2)
        
        ax_pitch.set_yscale('log')
        ax_pitch.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_pitch.set_yticks([50, 100, 200, 400, 600])
        
        ax_pitch.set_title("2. Fundamental Frequency (F0) Tracking & Register Isolation", fontsize=10, fontweight='bold', color='#2b6cb0')
        ax_pitch.set_xlabel("Time (Seconds)", fontsize=8)
        ax_pitch.set_ylabel("Pitch (Hz - Log Scale)", fontsize=8)
        ax_pitch.grid(True, which='both', linestyle=':', alpha=0.5)
        ax_pitch.legend(loc='upper right', fontsize=8)
        
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # ---------------------------------------------------------
        # PAGE 2: TEMPORAL INFLUENCE & HARMONIC CLOCK
        # ---------------------------------------------------------
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("APEX Advanced Musicology Report", fontsize=15, fontweight='bold', y=0.96, color='#1a365d')
        
        gs2 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.2])
        gs2.update(left=0.12, right=0.90, top=0.88, bottom=0.08, hspace=0.35)
        
        # --- SUBPLOT 3: Lagged Envelope Cross-Correlation ---
        ax_lag = fig2.add_subplot(gs2[0])
        ax_lag.plot(lag_times, env_corrs, color='#319795', lw=1.5)
        ax_lag.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax_lag.axvline(peak_lag, color='#e53e3e', linestyle=':', lw=1.5)
        ax_lag.plot(peak_lag, peak_corr, 'ro')
        
        ax_lag.set_title("3. Time-Lagged Envelope Cross-Correlation (Who Leads vs Who Follows)", fontsize=10, fontweight='bold', color='#2b6cb0')
        ax_lag.set_xlabel("Time Lag (Milliseconds - Bass relative to Melody)", fontsize=8)
        ax_lag.set_ylabel("Correlation Strength (r)", fontsize=8)
        ax_lag.set_xlim([-250, 250])
        ax_lag.grid(True, linestyle=':', alpha=0.6)
        
        # Dynamic commentary based on lead/lag
        if peak_lag < 0:
            com_text = f"Empirical Peak: Bass leads Melody by {-peak_lag:.1f} ms (r = {peak_corr:.3f})"
        elif peak_lag > 0:
            com_text = f"Empirical Peak: Melody leads Bass by {peak_lag:.1f} ms (r = {peak_corr:.3f})"
        else:
            com_text = f"Perfect Synchrony: Peak correlation at exactly 0.0 ms (r = {peak_corr:.3f})"
        ax_lag.text(0.03, 0.85, com_text, transform=ax_lag.transAxes, fontsize=8, color='#e53e3e', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor='#feb2b2', alpha=0.9))
        
        # --- SUBPLOT 4: Polar Chroma Consonance Clock ---
        # Chroma clocks use a polar projection
        ax_polar = fig2.add_subplot(gs2[1], projection='polar')
        theta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        # Close the polar plot circle
        theta = np.append(theta, theta[0])
        val_bass = np.append(chroma_bass, chroma_bass[0])
        val_melody = np.append(chroma_melody, chroma_melody[0])
        
        ax_polar.plot(theta, val_bass, color='#1f77b4', lw=1.5, label='Bass profile')
        ax_polar.fill(theta, val_bass, color='#1f77b4', alpha=0.2)
        ax_polar.plot(theta, val_melody, color='#ff7f0e', lw=1.5, label='Melody profile')
        ax_polar.fill(theta, val_melody, color='#ff7f0e', alpha=0.2)
        
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        ax_polar.set_xticks(theta[:-1])
        ax_polar.set_xticklabels(pitch_names, fontsize=8, fontweight='semibold')
        ax_polar.set_yticklabels([])  # Hide grid circles
        
        ax_polar.set_title("4. Polar Chroma Consonance Clock (Overlapping Pitch Profiles)", fontsize=10, fontweight='bold', color='#2b6cb0', y=1.12)
        ax_polar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
        
        pdf.savefig(fig2)
        plt.close(fig2)
        
    print(f"💾 [Success] Musicology findings saved to: {os.path.abspath(output_pdf)}")


# --- 4. FALLBACK MIX GENERATOR FOR DEMO ---
def generate_fallback_wave():
    print("⚠️ [Setup] Target WAV not specified. Constructing synthesized polyphony demonstration mix...")
    t = np.arange(int(TARGET_SR * 6.0)) / TARGET_SR
    
    # Bass channel: 55Hz alternating to 65Hz
    f_bass = np.where(t < 3.0, 55.0, 65.0)
    phase_b = 2 * np.pi * np.cumsum(f_bass) / TARGET_SR
    bass_ch = 0.5 * np.sin(phase_b)
    
    # Melody channel with slight timing delay (+20ms) and shifting fifths (330Hz / 390Hz)
    t_del = t - 0.020
    f_mel = np.where(t_del < 3.0, 330.0, 390.0)
    phase_m = 2 * np.pi * np.cumsum(f_mel) / TARGET_SR
    melody_ch = 0.4 * np.sin(phase_m)
    # Silence negative time delay
    melody_ch[t < 0.020] = 0.0
    
    mix = bass_ch + melody_ch
    
    import wave
    name = "polyphonic_musicology_target.wav"
    scaled = np.clip(mix, -1.0, 1.0) * 32767.0
    with wave.open(name, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(scaled.astype(np.int16).tobytes())
        
    return name


# --- 5. PIPELINE EXECUTION ---
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("📂 Please select your target monaural (.wav) file...")
    path = filedialog.askopenfilename(title="Select Target Wave File", filetypes=[("Audio Files", "*.wav")])
    
    if not path:
        path = generate_fallback_wave()
        
    try:
        bass, melody, original = load_and_decompose(path)
        generate_musicology_report(bass, melody, original, "apex_advanced_musicology_report.pdf")
        print("\n✅ Musicology analysis report completed successfully.")
    except Exception as e:
        print(f"\n❌ Execution Error: {str(e)}")
        sys.exit(1)
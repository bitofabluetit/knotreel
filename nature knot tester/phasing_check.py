#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 APEX PHRASING DEPENDENCE & MACRO-STRUCTURAL ANALYZER
=========================================================================================
Decomposes an audio target, extracts ultra-low frequency phrasing envelopes, 
calculates continuous rolling correlation states, and compiles a PDF report.
"""

import os
import sys
import textwrap
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION ---
TARGET_SR = 22050
NUM_HEADS = 3
WINDOW_SEC = 6.0  # Macro structural window (seconds)


# --- 1. SIGNAL PRE-PROCESSING & DECOMPOSITION ---
def load_and_decompose(path):
    print("⏳ [Ingestion] Loading audio and running spectral normalization...")
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    
    print("🧬 [NMF] Extracting registration layers for phrase analysis...")
    stft = librosa.stft(y, n_fft=1024, hop_length=256)
    V = np.abs(stft) + 1e-9
    F, T = V.shape
    
    W = np.random.rand(F, NUM_HEADS) + 1e-5
    H = np.random.rand(NUM_HEADS, T) + 1e-5
    
    for _ in range(30):
        V_approx = W @ H + 1e-9
        H = H * (W.T @ (V / V_approx)) / (W.T @ np.ones((F, T)) + 1e-9)
        V_approx = W @ H + 1e-9
        W = W * ((V / V_approx) @ H.T) / (np.ones((F, T)) @ H.T + 1e-9)
        
    freqs = np.linspace(0, TARGET_SR / 2, F)
    centroids = [np.sum(freqs * W[:, j]) / (np.sum(W[:, j]) + 1e-9) for j in range(NUM_HEADS)]
    sorted_indices = np.argsort(centroids)
    
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
        
    return signals["Bass"], signals["Melody"]


# --- 2. PHRASING DEPENDENCE CALCULATION ---
def analyze_phrasing_dependence(bass, melody):
    print("📐 [Analysis] Extracting long-term phrasing arcs & envelope correlation...")
    hop_length = 512
    rms_b = librosa.feature.rms(y=bass, frame_length=2048, hop_length=hop_length)[0]
    rms_m = librosa.feature.rms(y=melody, frame_length=2048, hop_length=hop_length)[0]
    
    frames_per_sec = TARGET_SR / hop_length
    times = np.arange(len(rms_b)) / frames_per_sec
    
    # Apply a heavy Gaussian smooth to capture the slow 2-second "breathing" cycles
    sigma = int(2.0 * frames_per_sec)
    phrase_env_b = gaussian_filter1d(rms_b, sigma=sigma)
    phrase_env_m = gaussian_filter1d(rms_m, sigma=sigma)
    
    # Peak normalize
    phrase_env_b /= (np.max(phrase_env_b) + 1e-9)
    phrase_env_m /= (np.max(phrase_env_m) + 1e-9)
    
    # Calculate rolling correlation across the time windows
    window_size_frames = int(WINDOW_SEC * frames_per_sec)
    half_win = window_size_frames // 2
    rolling_corr = np.zeros_like(phrase_env_b)
    
    for i in range(len(phrase_env_b)):
        start = max(0, i - half_win)
        end = min(len(phrase_env_b), i + half_win)
        
        slice_b = phrase_env_b[start:end]
        slice_m = phrase_env_m[start:end]
        
        if len(slice_b) > 5:
            corr = np.corrcoef(slice_b, slice_m)[0, 1]
            rolling_corr[i] = corr if not np.isnan(corr) else 0.0
            
    # Calculate state percentages
    co_phrasing_pct = np.mean(rolling_corr > 0.4) * 100
    antiphonal_pct = np.mean(rolling_corr < -0.4) * 100
    independent_pct = np.mean((rolling_corr >= -0.4) & (rolling_corr <= 0.4)) * 100
    
    metrics = {
        'times': times,
        'phrase_env_b': phrase_env_b,
        'phrase_env_m': phrase_env_m,
        'rolling_corr': rolling_corr,
        'co_phrasing_pct': co_phrasing_pct,
        'antiphonal_pct': antiphonal_pct,
        'independent_pct': independent_pct
    }
    return metrics


# --- 3. TEXT WRAPPING COMPOSER ---
def write_text(ax, text, x, y, size=10, weight='normal', color='#2c3e50', wrap_width=90):
    wrapped = textwrap.wrap(text, width=wrap_width)
    current_y = y
    for line in wrapped:
        ax.text(
            x, current_y, line, fontsize=size, fontweight=weight, 
            color=color, transform=ax.transAxes, family='sans-serif'
        )
        current_y -= (size / 72.0) * 1.5
    return current_y


# --- 4. PDF REPORT COMPILATION ---
def compile_phrasing_pdf(metrics, output_pdf="phrasing_dependence_report.pdf"):
    print(f"🎨 [Visualization] Formatting PDF phrasing report: {output_pdf}")
    
    times = metrics['times']
    phrase_env_b = metrics['phrase_env_b']
    phrase_env_m = metrics['phrase_env_m']
    rolling_corr = metrics['rolling_corr']
    
    with PdfPages(output_pdf) as pdf:
        # ---------------------------------------------------------
        # PAGE 1: TEXT-BASED STRUCTURAL DISCOVERY DOCUMENT
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(8.5, 11))
        ax_text = fig1.add_subplot(111)
        ax_text.axis('off')
        
        y = 0.95
        y = write_text(ax_text, "APEX PHRASING DEPENDENCE REPORT", 0.05, y, size=16, weight='bold', color='#1a365d')
        y -= 0.01
        y = write_text(ax_text, "Macro-Envelope Modulation & Long-Term Performance Coupling", 0.05, y, size=10, weight='normal', color='#718096')
        y -= 0.04
        
        # Section 1: Introduction to Phrasing
        y = write_text(ax_text, "1. Methodological Overview", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.005
        desc_overview = (
            "While fast, note-level transients capture the instantaneous execution of a musical track, "
            "phrasing represents the slow, macro-level 'breathing' of a performance, typically unfolding "
            "over 2 to 8 seconds. By applying heavy Gaussian low-pass smoothing to the amplitude envelopes, "
            "we filter out the rapid, transient movements of individual notes. This isolates the slow "
            "structural swells, allowing us to calculate the rolling Pearson correlation of the phrasing "
            "relationship over time."
        )
        y = write_text(ax_text, desc_overview, 0.07, y, size=9, color='#2d3748')
        y -= 0.03
        
        # Section 2: Distribution Metrics
        y = write_text(ax_text, "2. Quantitative Phrasing Distribution", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.01
        
        metric_summary = (
            f"• Co-Phrasing / Unison State (r > 0.4): {metrics['co_phrasing_pct']:.2f}%\n"
            f"• Antiphonal / Call-and-Response State (r < -0.4): {metrics['antiphonal_pct']:.2f}%\n"
            f"• Contrapuntal / Independent State (-0.4 <= r <= 0.4): {metrics['independent_pct']:.2f}%"
        )
        
        ax_text.text(0.07, y, metric_summary, transform=ax_text.transAxes, fontsize=9.5, fontfamily='monospace', 
                     color='#1a202c', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7fafc', edgecolor='#e2e8f0'))
        y -= 0.12
        
        # Section 3: Diagnostic Interpretation
        y = write_text(ax_text, "3. Macro-Structural Interaction Commentary", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.005
        
        # Select the dominant state to write dynamic, customized report text
        max_state = np.argmax([metrics['co_phrasing_pct'], metrics['antiphonal_pct'], metrics['independent_pct']])
        if max_state == 0:
            commentary = (
                f"Dynamic profiling reveals that the performance is heavily dominated by Co-Phrasing "
                f"({metrics['co_phrasing_pct']:.1f}%). This indicates that the Bass and Melody layers "
                f"are structurally unified, swelling and decaying in volume together over the {WINDOW_SEC}-second "
                f"windows. This represents an intense, collective emotional drive where both instruments "
                f"complement and magnify the structural arcs of the song simultaneously."
            )
        elif max_state == 1:
            commentary = (
                f"Dynamic profiling reveals that the performance is dominated by Antiphonal (Call-and-Response) "
                f"patterns ({metrics['antiphonal_pct']:.1f}%). This indicates that the sources are actively "
                f"interlocking and swapping prominence; as the melody swells to carry a phrase, the bass rests "
                f"back into a supportive profile, and vice versa. This constitutes an active musical dialogue "
                f"where one source speaks while the other listens."
            )
        else:
            commentary = (
                f"Dynamic profiling reveals that the performance is dominated by Contrapuntal Independence "
                f"({metrics['independent_pct']:.1f}%). The two instruments maintain completely separate and "
                f"autonomous musical narratives, phrasing without reference to the other's long-term amplitude "
                f"swells. This is characteristic of complex, non-homophonic musical writing where both parts "
                f"possess structural equality."
            )
            
        y = write_text(ax_text, commentary, 0.07, y, size=9, color='#2d3748')
        
        ax_text.text(0.05, 0.04, "APEX Phrasing Diagnostic Tool • Compiled from NMF sub-envelope metrics", 
                     fontsize=8, color='#a0aec0', transform=ax_text.transAxes)
        
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # ---------------------------------------------------------
        # PAGE 2: STRUCTURAL ANALYSIS GRAPHICS
        # ---------------------------------------------------------
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("APEX Phrasing Alignment Dashboard", fontsize=14, fontweight='bold', y=0.95, color='#1a365d')
        
        gs2 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0])
        gs2.update(left=0.12, right=0.90, top=0.88, bottom=0.08, hspace=0.35)
        
        # Subplot 1: Phrase Envelopes
        ax1 = fig2.add_subplot(gs2[0])
        ax1.plot(times, phrase_env_b, label='Bass Phrasing Arc', color='#1f77b4', lw=1.5)
        ax1.plot(times, phrase_env_m, label='Melody Phrasing Arc', color='#ff7f0e', lw=1.5)
        ax1.set_title("1. Slow Envelope Modulation (Phrase-Level Tracking)", fontsize=10, fontweight='bold', color='#2b6cb0')
        ax1.set_ylabel("Normalized Energy", fontsize=8)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # Subplot 2: Rolling Alignment Correlation
        ax2 = fig2.add_subplot(gs2[1], sharex=ax1)
        ax2.plot(times, rolling_corr, color='#5a189a', lw=1.2, label='Rolling Correlation')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel("Correlation (r)", fontsize=8)
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_xlabel("Time (Seconds)", fontsize=8)
        ax2.set_title(f"2. Rolling Phrase Alignment ({WINDOW_SEC}s Window)", fontsize=10, fontweight='bold', color='#2b6cb0')
        
        # Apply color shading for states
        ax2.fill_between(times, 0, rolling_corr, where=(rolling_corr > 0.4), color='#f53b57', alpha=0.15, label='Co-Phrasing (Unison)')
        ax2.fill_between(times, 0, rolling_corr, where=(rolling_corr < -0.4), color='#3c40c6', alpha=0.15, label='Call-and-Response')
        ax2.fill_between(times, 0, rolling_corr, where=((rolling_corr >= -0.4) & (rolling_corr <= 0.4)), color='#808080', alpha=0.08, label='Independent')
        
        # Dedup legend listings
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=7, ncol=2)
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        pdf.savefig(fig2)
        plt.close(fig2)
        
    print(f"💾 [Success] Structural phrasing report saved to: {os.path.abspath(output_pdf)}")


# --- 5. DEMO MIX GENERATOR ---
def create_mock_wave():
    print("⚠️ [Setup] Input WAV file not specified. Generating synthetic phrasing mix...")
    t = np.arange(int(TARGET_SR * 15.0)) / TARGET_SR  # 15 seconds
    
    # Let's program a distinct "Call-and-Response" pattern:
    # Bass active for the first 3 seconds, rests, then active from 6-9, etc.
    gate_b = np.where(((t >= 0) & (t < 3)) | ((t >= 6) & (t < 9)) | ((t >= 12) & (t < 15)), 1.0, 0.1)
    bass = 0.5 * np.sin(2 * np.pi * 55 * t) * gate_b
    
    # Melody active when bass is resting (3-6, 9-12)
    gate_m = np.where(((t >= 3) & (t < 6)) | ((t >= 9) & (t < 12)), 1.0, 0.1)
    melody = 0.4 * np.sin(2 * np.pi * 330 * t) * gate_m
    
    mix = bass + melody
    
    import wave
    name = "phrasing_mock_target.wav"
    scaled = np.clip(mix, -1.0, 1.0) * 32767.0
    with wave.open(name, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(scaled.astype(np.int16).tobytes())
        
    return name


# --- 6. RUNTIME PIPELINE EXECUTION ---
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("📂 Please select your target monaural (.wav) file...")
    path = filedialog.askopenfilename(title="Select Target Wave File", filetypes=[("Audio Files", "*.wav")])
    
    if not path:
        path = create_mock_wave()
        
    try:
        bass, melody = load_and_decompose(path)
        metrics = analyze_phrasing_dependence(bass, melody)
        compile_phrasing_pdf(metrics, "phrasing_dependence_report.pdf")
        print("\n✅ Phrasing dependency report generated successfully.")
    except Exception as e:
        print(f"\n❌ Execution Error: {str(e)}")
        sys.exit(1)
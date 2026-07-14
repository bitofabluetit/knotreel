#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 APEX TOPOLOGICAL SOURCE SEPARATOR & KNOT ANALYZER
=========================================================================================
Decomposes a single audio track, performs 3D delay embedding on individual sources,
calculates physical kinematic invariants, and outputs a visual PDF dashboard and CSV matrix.
"""

import os
import csv
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for headless and CLI execution
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter1d

# --- CONFIGURATION ---
TARGET_SR = 22050
NUM_SOURCES = 3
TAU_MS = 10  # Coordinate latency (220 samples at 22050Hz)


# --- 1. SIGNAL PRE-PROCESSING & DECOMPOSITION ENGINE ---
def load_and_preprocess(path):
    print("⏳ [Ingestion] Loading audio and normalizing to mono float32...")
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    # Peak normalization
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


def separate_sources_nmf(y, k=NUM_SOURCES, max_iter=50):
    print(f"🧬 [Decomposition] Resolving {k} acoustic sources via STFT-NMF...")
    stft_matrix = librosa.stft(y, n_fft=1024, hop_length=256)
    V = np.abs(stft_matrix) + 1e-9
    F, T = V.shape
    
    # Initialize templates (W) and activations (H)
    W = np.random.rand(F, k) + 1e-5
    H = np.random.rand(k, T) + 1e-5
    
    # Multiplicative update rules
    for _ in range(max_iter):
        V_approx = W @ H + 1e-9
        H = H * (W.T @ (V / V_approx)) / (W.T @ np.ones((F, T)) + 1e-9)
        V_approx = W @ H + 1e-9
        W = W * ((V / V_approx) @ H.T) / (np.ones((F, T)) @ H.T + 1e-9)
        
    # Classify sources via spectral centroids
    freqs = np.linspace(0, TARGET_SR / 2, F)
    centroids = []
    for j in range(k):
        centroid_j = np.sum(freqs * W[:, j]) / (np.sum(W[:, j]) + 1e-9)
        centroids.append(centroid_j)
        
    sorted_indices = np.argsort(centroids)
    labels_map = {}
    if k == 3:
        labels_map[sorted_indices[0]] = "Bass / Kick Resonance"
        labels_map[sorted_indices[1]] = "Melodic Harmonics"
        labels_map[sorted_indices[2]] = "High Transients"
    else:
        for rank, idx in enumerate(sorted_indices):
            labels_map[idx] = f"Source {rank} ({centroids[idx]:.0f}Hz)"
            
    # Synthesize waveforms back from components
    separated_waveforms = []
    source_names = []
    for j in range(k):
        mask = (W[:, [j]] @ H[[j], :]) / (W @ H + 1e-9)
        sig = librosa.istft(stft_matrix * mask, hop_length=256)
        
        # Ensure length match
        if len(sig) < len(y):
            sig = np.pad(sig, (0, len(y) - len(sig)))
        else:
            sig = sig[:len(y)]
            
        separated_waveforms.append(sig)
        source_names.append(labels_map[j])
        
    return separated_waveforms, source_names


# --- 2. TOPOLOGICAL DELAY EMBEDDING & KINEMATICS ENGINE ---
class TopologicalAcousticKnot:
    def __init__(self, audio, sr, source_name, tau_ms=TAU_MS):
        self.source_name = source_name
        self.audio = audio
        
        # Takens' Delay Coordinate Assignment
        tau = int(sr * (tau_ms / 1000.0))
        self.x = audio[:-2*tau]
        self.y_del = audio[tau:-tau]
        self.z_del = audio[2*tau:]
        
        # Radial displacement
        self.radius = np.sqrt(self.x**2 + self.y_del**2 + self.z_del**2) + 1e-8
        
        # Kinematic Derivatives (Velocity, Acceleration, Jerk)
        dx = np.gradient(self.x)
        dy = np.gradient(self.y_del)
        dz = np.gradient(self.z_del)
        self.v_vec = np.column_stack((dx, dy, dz))
        self.velocity = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        ddz = np.gradient(dz)
        self.a_vec = np.column_stack((ddx, ddy, ddz))
        self.acceleration = np.sqrt(ddx**2 + ddy**2 + ddz**2) + 1e-8
        
        dddx = np.gradient(ddx)
        dddy = np.gradient(ddy)
        dddz = np.gradient(ddz)
        self.jerk = np.sqrt(dddx**2 + dddy**2 + dddz**2) + 1e-8
        
        # Curvature: ||v x a|| / ||v||^3
        cross_prod = np.cross(self.v_vec, self.a_vec)
        cross_mag = np.sqrt(np.sum(cross_prod**2, axis=1))
        self.curvature = cross_mag / (self.velocity**3)
        
        # Wobble Factor: (Curvature * Jerk) / Radius^2
        self.wobble = (self.curvature * self.jerk) / (self.radius**2)
        
        # Physical Invariant Metric Extraction
        self.mean_radius = np.mean(self.radius)
        self.mean_velocity = np.mean(self.velocity)
        self.mean_acceleration = np.mean(self.acceleration)
        self.mean_curvature = np.mean(self.curvature)
        
        # Decoupled Dynamics Constraint (Pearson Correlation r vs a)
        corr_matrix = np.corrcoef(self.radius, self.acceleration)
        self.r_a_correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        # Log Max Wobble
        self.log_max_wobble = np.log10(np.percentile(self.wobble, 95) + 1e-9)
        
        # Void Density (binning phase space to find unvisited spatial density)
        self.void_density = self.calculate_void_density()

    def calculate_void_density(self, grid_res=15):
        # Normalize coordinates to range [0, grid_res-1]
        coords = np.column_stack((self.x, self.y_del, self.z_del))
        min_vals = np.min(coords, axis=0)
        max_vals = np.max(coords, axis=0)
        span = max_vals - min_vals + 1e-9
        
        normalized = ((coords - min_vals) / span * (grid_res - 1)).astype(int)
        unique_bins = np.unique(normalized, axis=0)
        visited_count = len(unique_bins)
        total_bins = grid_res**3
        
        return (1.0 - (visited_count / total_bins)) * 100.0


# --- 3. EXPORT EXCEL-COMPATIBLE CSV ENGINE ---
def generate_master_csv(knots, output_csv="acoustic_relationship_matrix.csv"):
    print(f"📊 [Export] Writing physical relationships to: {output_csv}")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Table 1: Individual Source Invariant Profiles
        writer.writerow(["=== SECTION 1: TOPOLOGICAL METRIC INVARIANT PROFILES ==="])
        writer.writerow([
            "Source Name", "Mean Radius", "Mean Velocity", "Mean Acceleration", 
            "Mean Curvature", "Log Max Wobble (95th %)", "Void Density (%)", "Radius-Acceleration Correlation (C)"
        ])
        for knot in knots:
            writer.writerow([
                knot.source_name,
                f"{knot.mean_radius:.6f}",
                f"{knot.mean_velocity:.6f}",
                f"{knot.mean_acceleration:.6f}",
                f"{knot.mean_curvature:.6f}",
                f"{knot.log_max_wobble:.4f}",
                f"{knot.void_density:.2f}%",
                f"{knot.r_a_correlation:.4f}"
            ])
            
        writer.writerow([])
        
        # Table 2: Cross-Source Coupling (Radius Correlation Matrix)
        writer.writerow(["=== SECTION 2: PAIRWISE TOPOLOGICAL COUPLING MATRIX ==="])
        names = [k.source_name for k in knots]
        writer.writerow(["Source Relationship Correlation (R_a vs R_b)"] + names)
        
        for i, knot_a in enumerate(knots):
            row = [knot_a.source_name]
            for j, knot_b in enumerate(knots):
                min_len = min(len(knot_a.radius), len(knot_b.radius))
                r_a = knot_a.radius[:min_len]
                r_b = knot_b.radius[:min_len]
                corr = np.corrcoef(r_a, r_b)[0, 1]
                row.append(f"{corr:.6f}" if not np.isnan(corr) else "1.000000")
            writer.writerow(row)


# --- 4. GRAPHICAL REPORT VISUALIZATION ENGINE ---
def compile_pdf_report(knots, output_pdf="topological_knot_report.pdf"):
    print(f"🎨 [Visualization] Creating topological PDF drawing structure...")
    
    # Establish dynamic canvas size depending on source count
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Computational Acoustic Topology: Multi-Source Knot Decomposition", fontsize=16, fontweight='bold', y=0.96)
    
    # Grid structure: LEFT Column (separate knots), RIGHT Column (master overlay + metrics table)
    gs = GridSpec(len(knots), 2, width_ratios=[1.0, 1.2])
    gs.update(left=0.07, right=0.95, top=0.88, bottom=0.08, wspace=0.25, hspace=0.35)
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    ds = 15  # Trajectory downsampling factor for vector PDF optimization
    
    # --- LEFT COLUMN: Separate Knot Attractors ---
    for idx, knot in enumerate(knots):
        ax = fig.add_subplot(gs[idx, 0], projection='3d')
        ax.plot(
            knot.x[::ds], knot.y_del[::ds], knot.z_del[::ds], 
            color=colors[idx % len(colors)], alpha=0.6, lw=0.8, rasterized=True
        )
        ax.set_title(f"Knot #{idx+1}: {knot.source_name}", fontsize=10, fontweight='semibold')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        
    # --- RIGHT COLUMN: Master Overlay Knot (Top Right) ---
    ax_master = fig.add_subplot(gs[0:2, 1], projection='3d')
    for idx, knot in enumerate(knots):
        ax_master.plot(
            knot.x[::ds], knot.y_del[::ds], knot.z_del[::ds], 
            label=f"{knot.source_name}", color=colors[idx % len(colors)], alpha=0.5, lw=0.8, rasterized=True
        )
    ax_master.set_title("Master Attractor Knot Overlay", fontsize=12, fontweight='bold')
    ax_master.set_xticklabels([])
    ax_master.set_yticklabels([])
    ax_master.set_zticklabels([])
    ax_master.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_master.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_master.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_master.set_box_aspect([1, 1, 1])
    ax_master.view_init(elev=20, azim=45)
    ax_master.legend(loc='upper right', frameon=True, fontsize=8)
    
    # --- RIGHT COLUMN: Text metrics box (Bottom Right) ---
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis('off')
    
    text_content = "🔬 TOPOLOGICAL ANALYSIS INVARIANT SNAPSHOT\n" + "-"*65 + "\n"
    for idx, knot in enumerate(knots):
        text_content += f"• {knot.source_name}:\n"
        text_content += f"  - Log Max Wobble: {knot.log_max_wobble:.4f} | Void Density: {knot.void_density:.2f}%\n"
        text_content += f"  - Radius-Acceleration correlation (C): {knot.r_a_correlation:.4f} "
        if knot.r_a_correlation >= 0.20:
            text_content += "[Coupled / Biological Flow]\n"
        else:
            text_content += "[Decoupled / Synthetic Bound]\n"
            
    ax_text.text(0.0, 0.95, text_content, transform=ax_text.transAxes, 
                 fontsize=9, fontfamily='monospace', va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#f7f7f7', edgecolor='#cccccc', alpha=0.8))
    
    plt.savefig(output_pdf, format='pdf', dpi=300)
    plt.close(fig)
    print(f"💾 [Success] Topological PDF compiled and saved to: {os.path.abspath(output_pdf)}")


# --- 5. SYNTHETIC DEMO MULTI-INSTRUMENT SOURCE GENERATOR ---
def create_mock_polyphony_wav():
    print("⚠️ [Setup] File not specified. Generating synthetic polyphony mix file...")
    t = np.arange(int(TARGET_SR * 5.0)) / TARGET_SR
    
    # Bass component with heavy physical coupling (jittered)
    jitter = 0.5 * np.sin(2 * np.pi * 3.0 * t)
    bass_phase = 2 * np.pi * (55.0 * t + np.cumsum(jitter) / TARGET_SR)
    bass = 0.6 * np.sin(bass_phase)
    
    # Melodic voice with dynamic amplitude envelope
    lead = 0.45 * np.sin(2 * np.pi * 330.0 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
    
    # Transient high-frequency percussion component
    high_transient = 0.2 * np.sin(2 * np.pi * 1200.0 * t) * np.exp(-10 * (t % 0.5))
    
    mix = bass + lead + high_transient
    
    import wave
    name = "polyphonic_synthesized_target.wav"
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
    
    # Attempt simple file prompt
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("📂 Please select your target monaural (.wav) file...")
    path = filedialog.askopenfilename(title="Select Target Wave File", filetypes=[("Audio Files", "*.wav")])
    
    if not path:
        path = create_mock_polyphony_wav()
        
    # Execution steps
    y = load_and_preprocess(path)
    separated_waveforms, source_names = separate_sources_nmf(y, k=NUM_SOURCES)
    
    knots = []
    for idx, waveform in enumerate(separated_waveforms):
        print(f"📐 [Embedding] Generating 3D Attractor for: {source_names[idx]}")
        knot = TopologicalAcousticKnot(waveform, TARGET_SR, source_names[idx])
        knots.append(knot)
        
    generate_master_csv(knots, "acoustic_relationship_matrix.csv")
    compile_pdf_report(knots, "topological_knot_report.pdf")
    print("\n✅ Execution Finished successfully.")

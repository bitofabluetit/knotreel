#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
 📟 APEX TOPOLOGICAL PATTERN DISCOVERY & COUPLING ENGINE (BUGFIX EDITION)
=========================================================================================
Parses acoustic topological metrics from the APEX CSV file, automatically executes
unsupervised statistical analyses, and generates a visual multi-page PDF findings report.
"""

import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# --- 1. ROBUST CSV PARSING ENGINE ---
def parse_apex_csv(filepath="acoustic_relationship_matrix.csv"):
    """Parses mixed-section APEX CSV file into separate DataFrames."""
    print(f"📖 [Parsing] Reading CSV structure: {filepath}...")
    
    section1_rows = []
    section2_rows = []
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if "SECTION 1" in line_str:
                current_section = 1
                continue
            elif "SECTION 2" in line_str:
                current_section = 2
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if current_section == 1:
                section1_rows.append(parts)
            elif current_section == 2:
                section2_rows.append(parts)
                
    if not section1_rows or not section2_rows:
        raise ValueError("The parsed file does not contain Section 1 or Section 2 headers.")
        
    # Process Section 1 (Metrics Table)
    headers1 = section1_rows[0]
    data1 = section1_rows[1:]
    df_metrics = pd.DataFrame(data1, columns=headers1)
    
    # Clean numeric columns (using a raw string 'r' for regex to avoid syntax warnings)
    for col in df_metrics.columns[1:]:
        df_metrics[col] = (
            df_metrics[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(r'[^\d\.\-\+eE]', '', regex=True)
        )
        df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
        
    # Process Section 2 (Coupling Matrix)
    headers2 = section2_rows[0]
    source_cols = headers2[1:]
    data2 = section2_rows[1:]
    
    matrix_idx = [row[0] for row in data2]
    matrix_vals = []
    for row in data2:
        numeric_row = []
        for val in row[1:]:
            clean_val = val.replace('%', '').strip()
            numeric_row.append(float(clean_val) if clean_val else 1.0)
        matrix_vals.append(numeric_row)
        
    df_matrix = pd.DataFrame(matrix_vals, index=matrix_idx, columns=source_cols)
    
    return df_metrics, df_matrix


# --- 2. UNSUPERVISED PATTERN DISCOVERY ALGORITHMS ---
def discover_patterns(df_metrics, df_matrix):
    print("🔬 [Analysis] Running pattern discovery algorithms...")
    findings = {}
    
    # Extract numerical features
    features = df_metrics.iloc[:, 1:]
    sources = df_metrics.iloc[:, 0].tolist()
    
    # Standardize features for anomaly and distance metrics
    mean = features.mean()
    std = features.std()
    # Replace zero std to prevent division-by-zero
    std[std == 0] = 1e-9
    standardized = (features - mean) / std
    
    # 2.1 Feature-to-Feature Correlations (if enough data points exist)
    if len(df_metrics) >= 3:
        feat_corr = features.corr()
        # Find the absolute strongest relationships (using to_numpy(copy=True) to avoid read-only restrictions)
        corr_matrix = feat_corr.to_numpy(copy=True)
        np.fill_diagonal(corr_matrix, 0.0)
        abs_corr = np.abs(corr_matrix)
        
        max_idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        strongest_pair = (feat_corr.index[max_idx[0]], feat_corr.columns[max_idx[1]])
        r_val = corr_matrix[max_idx]
        findings['strongest_feature_correlation'] = {
            'pair': strongest_pair,
            'r_value': r_val
        }
    else:
        findings['strongest_feature_correlation'] = None

    # 2.2 Joint-Covariance Anomaly Score (Euclidean distance in standardized space)
    distances = np.sqrt(np.sum(standardized**2, axis=1))
    df_metrics['Anomaly_Score'] = distances
    most_anomalous_idx = distances.idxmax()
    
    findings['anomaly_profile'] = {
        'source': sources[most_anomalous_idx],
        'score': distances.max(),
        'average_score': distances.mean(),
        'all_scores': dict(zip(sources, distances))
    }
    
    # 2.3 Physical Entanglement Centrality (Section 2 Coupling Matrix)
    # Using to_numpy(copy=True) ensures write access for fill_diagonal
    matrix_vals = df_matrix.to_numpy(copy=True)
    np.fill_diagonal(matrix_vals, np.nan)
    mean_coupling = pd.Series(np.nanmean(np.abs(matrix_vals), axis=1), index=df_matrix.index)
    hub_idx = mean_coupling.idxmax()
    
    findings['coupling_centrality'] = {
        'hub_source': hub_idx,
        'centrality_value': mean_coupling.max(),
        'all_centralities': mean_coupling.to_dict()
    }
    
    # 2.4 Unsupervised Cluster Proximity
    # Find the two closest sources in the standardized topological space
    from scipy.spatial.distance import pdist, squareform
    if len(df_metrics) >= 2:
        dist_matrix = squareform(pdist(standardized))
        np.fill_diagonal(dist_matrix, np.inf)
        closest_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        findings['closest_pair'] = {
            'source_a': sources[closest_idx[0]],
            'source_b': sources[closest_idx[1]],
            'distance': dist_matrix[closest_idx]
        }
    else:
        findings['closest_pair'] = None
        
    return findings


# --- 3. PROGRAMMATIC TEXT RENDERING FOR PDF ---
def write_text(ax, text, x, y, size=10, weight='normal', color='#2c3e50', wrap_width=90):
    """Draws left-aligned wrapped text lines on ax, returning the new Y position."""
    wrapped = textwrap.wrap(text, width=wrap_width)
    current_y = y
    for line in wrapped:
        ax.text(
            x, current_y, line, fontsize=size, fontweight=weight, 
            color=color, transform=ax.transAxes, family='sans-serif'
        )
        current_y -= (size / 72.0) * 1.5
    return current_y


# --- 4. GRAPHICAL VISUALIZATION & PDF REPORT ENGINE ---
def compile_findings_pdf(df_metrics, df_matrix, findings, output_pdf="topological_pattern_discovery.pdf"):
    print(f"🎨 [Visualization] Rendering findings into PDF layout: {output_pdf}")
    
    with PdfPages(output_pdf) as pdf:
        # ---------------------------------------------------------
        # PAGE 1: EXECUTIVE PATTERN SUMMARY REPORT (Text Layout)
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(8.5, 11))
        ax_text = fig1.add_subplot(111)
        ax_text.axis('off')
        
        y = 0.95
        y = write_text(ax_text, "APEX TOPOLOGICAL DISCOVERY REPORT", 0.05, y, size=16, weight='bold', color='#1a365d')
        y -= 0.01
        y = write_text(ax_text, "Acoustic Pattern Analytics & Unsupervised Statistical Couplings", 0.05, y, size=10, weight='normal', color='#718096')
        y -= 0.03
        
        # Section A: Entanglement Coupling
        y = write_text(ax_text, "1. Acoustic Entanglement and Coupling Hubs", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.005
        hub = findings['coupling_centrality']['hub_source']
        hub_val = findings['coupling_centrality']['centrality_value']
        desc_entangle = (
            f"Unsupervised network analysis identifies the sound source '{hub}' as the primary "
            f"Physical Entanglement Hub, exhibiting a mean absolute cross-coupling correlation "
            f"of {hub_val:.4f} with all other signal profiles. This indicates that its geometric coordinate "
            f"pathway is highly integrated with the remaining sources, likely capturing the core resonance "
            f"or ambient leakage within the original sound field."
        )
        y = write_text(ax_text, desc_entangle, 0.07, y, size=9, color='#2d3748')
        y -= 0.03
        
        # Section B: Anomaly Detection
        y = write_text(ax_text, "2. Joint-Covariance Anomaly Analysis", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.005
        anomaly_src = findings['anomaly_profile']['source']
        anomaly_score = findings['anomaly_profile']['score']
        desc_anomaly = (
            f"By evaluating how the extracted metrics covary globally using a standardized joint coordinate "
            f"distance, '{anomaly_src}' was discovered to be the most anomalous source, registering an "
            f"anomaly distance score of {anomaly_score:.4f} (ensemble average: {findings['anomaly_profile']['average_score']:.2f}). "
            f"This profile indicates non-conforming physical kinematics—possessing either anomalous phase-space "
            f"volume distribution, structural rigidity, or uncoupled acceleration-radius patterns."
        )
        y = write_text(ax_text, desc_anomaly, 0.07, y, size=9, color='#2d3748')
        y -= 0.03
        
        # Section C: Feature-to-Feature Relationships
        y = write_text(ax_text, "3. Automatic Feature-to-Feature Relationships", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.005
        if findings['strongest_feature_correlation']:
            pair = findings['strongest_feature_correlation']['pair']
            r_val = findings['strongest_feature_correlation']['r_value']
            desc_corr = (
                f"Statistical scanning detected a major relationship between the physical metrics "
                f"'{pair[0]}' and '{pair[1]}', demonstrating a Pearson correlation coefficient (r) "
                f"of {r_val:.4f}. This indicates that as the trajectory structure modifies along the "
                f"first dimension, the second dimension reacts in systemic harmony, mapping a predictable "
                f"physical law governing the underlying generator's dynamics."
            )
        else:
            desc_corr = "Insufficient data instances to establish robust statistical correlation coefficients across features."
        y = write_text(ax_text, desc_corr, 0.07, y, size=9, color='#2d3748')
        y -= 0.03
        
        # Section D: Group Closures
        y = write_text(ax_text, "4. Structural Groupings & Geometric Proximity", 0.05, y, size=11, weight='bold', color='#2b6cb0')
        y -= 0.005
        if findings['closest_pair']:
            src_a = findings['closest_pair']['source_a']
            src_b = findings['closest_pair']['source_b']
            dist_val = findings['closest_pair']['distance']
            desc_cluster = (
                f"Unsupervised distance metrics map '{src_a}' and '{src_b}' as the most geometrically similar "
                f"pair, separated by a localized phase distance of {dist_val:.4f}. This cluster grouping "
                f"indicates that their underlying generator dynamics share nearly equivalent physical structures "
                f"within the reconstructed 3D phase space."
            )
        else:
            desc_cluster = "Insufficient sources to determine spatial neighborhood distance profiles."
        y = write_text(ax_text, desc_cluster, 0.07, y, size=9, color='#2d3748')
        
        # Footer
        ax_text.text(0.05, 0.04, "APEX Analytics Engine • Report compiled dynamically from source metrics", 
                     fontsize=8, color='#a0aec0', transform=ax_text.transAxes)
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # ---------------------------------------------------------
        # PAGE 2: GRAPHICAL ANALYTICS DASHBOARD
        # ---------------------------------------------------------
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("APEX Discovery Dashboard: Statistical Visualizations", fontsize=14, fontweight='bold', y=0.95, color='#1a365d')
        
        # Define 3-row grid layout
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.0])
        gs.update(left=0.12, right=0.90, top=0.88, bottom=0.08, hspace=0.35)
        
        # Subplot 1: Feature Correlations Heatmap
        ax1 = fig2.add_subplot(gs[0])
        numeric_features = df_metrics.iloc[:, 1:-1] # Exclude source names and anomaly score column
        if len(numeric_features) >= 3:
            feat_corr_matrix = numeric_features.corr()
            cax = ax1.imshow(feat_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            fig2.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_xticks(range(len(feat_corr_matrix.columns)))
            ax1.set_xticklabels(feat_corr_matrix.columns, rotation=12, ha='right', fontsize=7)
            ax1.set_yticks(range(len(feat_corr_matrix.columns)))
            ax1.set_yticklabels(feat_corr_matrix.columns, fontsize=7)
            ax1.set_title("Topological Feature Correlation Heatmap", fontsize=10, fontweight='bold', color='#2b6cb0')
        else:
            ax1.text(0.5, 0.5, "Insufficient data points to map feature correlations", ha='center', va='center', color='#718096')
            ax1.set_title("Topological Feature Correlation (Unavailable)", fontsize=10, fontweight='bold', color='#2b6cb0')
            ax1.axis('off')
            
        # Subplot 2: Joint-Covariance Anomaly Scores
        ax2 = fig2.add_subplot(gs[1])
        scores_dict = findings['anomaly_profile']['all_scores']
        bars = ax2.barh(list(scores_dict.keys()), list(scores_dict.values()), color='#3182ce', alpha=0.8, height=0.5)
        
        # Highlight anomalous source in red
        for bar, name in zip(bars, scores_dict.keys()):
            if name == findings['anomaly_profile']['source']:
                bar.set_color('#e53e3e')
                
        ax2.set_xlabel("Joint-Covariance Anomaly Distance Score", fontsize=8)
        ax2.set_title("Unsupervised Anomaly Distance Distribution", fontsize=10, fontweight='bold', color='#2b6cb0')
        ax2.grid(True, linestyle=':', alpha=0.6, axis='x')
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        # Subplot 3: Cross-Coupling Matrix Plot
        ax3 = fig2.add_subplot(gs[2])
        cax3 = ax3.imshow(df_matrix.abs(), cmap='YlGnBu', vmin=0, vmax=1)
        fig2.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_xticks(range(len(df_matrix.columns)))
        ax3.set_xticklabels(df_matrix.columns, rotation=12, ha='right', fontsize=7)
        ax3.set_yticks(range(len(df_matrix.index)))
        ax3.set_yticklabels(df_matrix.index, fontsize=7)
        ax3.set_title("Pairwise Physical Entanglement Matrix (Absolute Value)", fontsize=10, fontweight='bold', color='#2b6cb0')
        
        pdf.savefig(fig2)
        plt.close(fig2)
        
    print(f"💾 [Success] Performance findings compiled and saved to: {os.path.abspath(output_pdf)}")


# --- 5. DEMO MOCK FILE GENERATOR (FALLBACK) ---
def create_fallback_csv():
    print("⚠️ [Setup] Input CSV file not found. Generating standardized fallback demo CSV data...")
    data = [
        ["=== SECTION 1: TOPOLOGICAL METRIC INVARIANT PROFILES ==="],
        ["Source Name", "Mean Radius", "Mean Velocity", "Mean Acceleration", "Mean Curvature", "Log Max Wobble (95th %)", "Void Density (%)", "Radius-Acceleration Correlation (C)"],
        ["Bass / Kick Resonance", "0.041235", "0.003412", "0.000125", "25.412151", "-1.215400", "96.42%", "0.352400"],
        ["Melodic Harmonics", "0.184512", "0.012541", "0.000845", "12.184521", "0.451200", "85.12%", "0.412500"],
        ["High Transients", "0.012541", "0.041254", "0.010541", "124.512415", "2.845100", "99.15%", "0.012500"],
        [],
        ["=== SECTION 2: PAIRWISE TOPOLOGICAL COUPLING MATRIX ==="],
        ["Source Relationship Correlation (R_a vs R_b)", "Bass / Kick Resonance", "Melodic Harmonics", "High Transients"],
        ["Bass / Kick Resonance", "1.000000", "0.412541", "0.112451"],
        ["Melodic Harmonics", "0.412541", "1.000000", "0.052141"],
        ["High Transients", "0.112451", "0.052141", "1.000000"]
    ]
    with open("acoustic_relationship_matrix.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


# --- 6. RUNTIME PIPELINE EXECUTION ---
if __name__ == "__main__":
    import csv
    filepath = "acoustic_relationship_matrix.csv"
    
    if not os.path.exists(filepath):
        create_fallback_csv()
        
    try:
        df_metrics, df_matrix = parse_apex_csv(filepath)
        findings = discover_patterns(df_metrics, df_matrix)
        compile_findings_pdf(df_metrics, df_matrix, findings, "topological_pattern_discovery.pdf")
        print("\n✅ Analytical pattern discovery process completed successfully.")
    except Exception as e:
        print(f"\n❌ Error running pattern discovery engine: {str(e)}")
        sys.exit(1)
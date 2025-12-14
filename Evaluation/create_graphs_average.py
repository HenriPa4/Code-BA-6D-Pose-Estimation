import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
FUSION_FILE = "Evaluation_Fusion_Results/fusion_metrics.json"
SINGLE_FILE = "Evaluation_Results/evaluation_metrics.json" # Optional: For comparison
OUTPUT_FOLDER = "Evaluation_Fusion_Graphs"

def load_data(filepath, type_label):
    if not os.path.exists(filepath):
        print(f"[{type_label}] File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Normalize column names because your scripts used different keys
    if 'err_t_m' in df.columns: # Fusion Format
        df['trans_err_cm'] = df['err_t_m'] * 100
        df['rot_err_deg'] = df['err_r_deg']
    elif 'error_translation_m' in df.columns: # Single Cam Format
        df['trans_err_cm'] = df['error_translation_m'] * 100
        df['rot_err_deg'] = df['error_rotation_deg']
        
    return df

def plot_comparison_histograms(df_fused, df_single=None):
    """Overlays Single Cam (Grey) vs Fusion (Green) distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    bins = np.linspace(0, 50, 50) # Fixed bins for fair comparison
    
    # --- Translation ---
    if df_single is not None:
        axes[0].hist(df_single['trans_err_cm'], bins=bins, alpha=0.3, color='gray', label='Single Cam (Raw)', density=True)
    
    axes[0].hist(df_fused['trans_err_cm'], bins=bins, alpha=0.6, color='green', label='Multi-Cam Fusion', density=True)
    axes[0].set_title('Translation Error Distribution')
    axes[0].set_xlabel('Error (cm)')
    axes[0].set_ylabel('Density')
    
    # Add Mean Lines
    mean_f = df_fused['trans_err_cm'].mean()
    axes[0].axvline(mean_f, color='green', linestyle='--', linewidth=2, label=f'Fusion Mean: {mean_f:.1f}cm')
    if df_single is not None:
        mean_s = df_single['trans_err_cm'].mean()
        axes[0].axvline(mean_s, color='gray', linestyle='--', linewidth=1, label=f'Single Mean: {mean_s:.1f}cm')
    
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Rotation ---
    rot_bins = np.linspace(0, 180, 50)
    
    if df_single is not None:
        axes[1].hist(df_single['rot_err_deg'], bins=rot_bins, alpha=0.3, color='gray', label='Single Cam', density=True)
        
    axes[1].hist(df_fused['rot_err_deg'], bins=rot_bins, alpha=0.6, color='orange', label='Multi-Cam Fusion', density=True)
    axes[1].set_title('Rotation Error Distribution')
    axes[1].set_xlabel('Error (Degrees)')
    axes[1].set_yscale('log') # Log scale because rotation errors are usually clustered at 0 or 180

    # Add Mean Lines
    mean_r_f = df_fused['rot_err_deg'].mean()
    axes[1].axvline(mean_r_f, color='orange', linestyle='--', linewidth=2, label=f'Fusion Mean: {mean_r_f:.1f}°')
    
    if df_single is not None:
        mean_r_s = df_single['rot_err_deg'].mean()
        axes[1].axvline(mean_r_s, color='gray', linestyle='--', linewidth=1, label=f'Single Mean: {mean_r_s:.1f}°')
    
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "1_comparison_histograms.png"))
    print("-> Generated Comparison Histograms")

def plot_comparison_cdf(df_fused, df_single=None):
    """The 'Money Shot': Shows how much faster Fusion reaches 100% accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Translation CDF ---
    # Fusion
    sorted_f = np.sort(df_fused['trans_err_cm'])
    y_f = np.arange(len(sorted_f)) / float(len(sorted_f))
    axes[0].plot(sorted_f, y_f, color='green', linewidth=3, label='Multi-Cam Fusion')
    
    # Single
    if df_single is not None:
        sorted_s = np.sort(df_single['trans_err_cm'])
        y_s = np.arange(len(sorted_s)) / float(len(sorted_s))
        axes[0].plot(sorted_s, y_s, color='gray', linestyle='--', linewidth=2, label='Single Cam (Avg)')

    axes[0].set_title('Translation Accuracy (CDF)')
    axes[0].set_xlabel('Error Threshold (cm)')
    axes[0].set_ylabel('Fraction of Frames')
    axes[0].set_xlim(0, 30) # Focus on the meaningful range
    axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower right')

    # --- Rotation CDF ---
    # Fusion
    sorted_rf = np.sort(df_fused['rot_err_deg'])
    y_rf = np.arange(len(sorted_rf)) / float(len(sorted_rf))
    axes[1].plot(sorted_rf, y_rf, color='orange', linewidth=3, label='Multi-Cam Fusion')

    # Single
    if df_single is not None:
        sorted_rs = np.sort(df_single['rot_err_deg'])
        y_rs = np.arange(len(sorted_rs)) / float(len(sorted_rs))
        axes[1].plot(sorted_rs, y_rs, color='gray', linestyle='--', linewidth=2, label='Single Cam (Avg)')

    axes[1].set_title('Rotation Accuracy (CDF)')
    axes[1].set_xlabel('Error Threshold (Degrees)')
    axes[1].set_xlim(0, 180)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "2_comparison_cdf.png"))
    print("-> Generated Comparison CDFs")

def print_stats(df, label):
    print(f"\n=== {label} STATISTICS ===")
    print(f"Count: {len(df)}")
    print(f"Mean Trans Err: {df['trans_err_cm'].mean():.2f} cm")
    print(f"Median Trans:   {df['trans_err_cm'].median():.2f} cm")
    print(f"Mean Rot Err:   {df['rot_err_deg'].mean():.2f} deg")
    
    success = df[(df['trans_err_cm'] < 10.0) & (df['rot_err_deg'] < 10.0)]
    print(f"Success Rate (<10cm, <10deg): {len(success)/len(df)*100:.1f}%")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load Data
    df_fused = load_data(FUSION_FILE, "FUSION")
    df_single = load_data(SINGLE_FILE, "SINGLE") # Might be None
    
    if df_fused is None:
        print("Fatal: No fusion data found.")
        return

    # Print Stats
    if df_single is not None:
        print_stats(df_single, "SINGLE CAM (BASELINE)")
    print_stats(df_fused, "MULTI-CAM FUSION")
    
    # Calculate Improvement
    if df_single is not None:
        imp_t = df_single['trans_err_cm'].mean() - df_fused['trans_err_cm'].mean()
        print(f"\n>> IMPROVEMENT: Fusion reduced average error by {imp_t:.2f} cm per frame.")

    # Generate Plots
    plot_comparison_histograms(df_fused, df_single)
    plot_comparison_cdf(df_fused, df_single)
    
    print(f"\nGraphs saved to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
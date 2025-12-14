import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
INPUT_FILE = "Evaluation_Results/evaluation_metrics.json"
OUTPUT_FOLDER = "Evaluation_Graphs"

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert to Pandas DataFrame for easier plotting
    df = pd.DataFrame(data)
    
    # Convert Translation Error to Centimeters for better readability
    df['error_translation_cm'] = df['error_translation_m'] * 100
    
    # Extract Depth (Z) from GT Translation for "Error vs Distance" analysis
    # Assuming gt_translation is [x, y, z]
    df['gt_depth_m'] = df['gt_translation'].apply(lambda t: t[2])
    
    return df

def plot_histograms(df):
    """1. Error Distribution (How often is the model wrong?)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Translation Histogram
    sns.histplot(df['error_translation_cm'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Translation Error Distribution')
    axes[0].set_xlabel('Error (cm)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['error_translation_cm'].mean(), color='r', linestyle='--', label=f"Mean: {df['error_translation_cm'].mean():.2f}cm")
    axes[0].legend()

    # Rotation Histogram
    sns.histplot(df['error_rotation_deg'], bins=50, kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Rotation Error Distribution')
    axes[1].set_xlabel('Error (Degrees)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(df['error_rotation_deg'].mean(), color='r', linestyle='--', label=f"Mean: {df['error_rotation_deg'].mean():.2f}°")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "1_error_histograms.png"))
    print("-> Generated Histograms")

def plot_cdf(df):
    """2. Cumulative Distribution Function (Standard Metric in Papers)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Translation CDF
    sorted_err_t = np.sort(df['error_translation_cm'])
    y_vals = np.arange(len(sorted_err_t)) / float(len(sorted_err_t))
    axes[0].plot(sorted_err_t, y_vals, linewidth=2)
    axes[0].set_title('Translation Accuracy (CDF)')
    axes[0].set_xlabel('Error Threshold (cm)')
    axes[0].set_ylabel('Fraction of Data within Threshold')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].set_xlim(0, np.percentile(sorted_err_t, 95)) # Zoom in to 95% of data

    # Rotation CDF
    sorted_err_r = np.sort(df['error_rotation_deg'])
    y_vals = np.arange(len(sorted_err_r)) / float(len(sorted_err_r))
    axes[1].plot(sorted_err_r, y_vals, linewidth=2, color='orange')
    axes[1].set_title('Rotation Accuracy (CDF)')
    axes[1].set_xlabel('Error Threshold (Degrees)')
    axes[1].set_ylabel('Fraction of Data within Threshold')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].set_xlim(0, 180)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "2_cumulative_accuracy.png"))
    print("-> Generated CDF Plots")

def plot_error_vs_depth(df):
    """3. Does the model get worse further away?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter: Depth vs Trans Error
    sns.scatterplot(data=df, x='gt_depth_m', y='error_translation_cm', alpha=0.3, ax=axes[0])
    sns.regplot(data=df, x='gt_depth_m', y='error_translation_cm', scatter=False, ax=axes[0], color='red')
    axes[0].set_title('Translation Error vs. Distance')
    axes[0].set_xlabel('Object Depth (m)')
    axes[0].set_ylabel('Translation Error (cm)')

    # Scatter: Depth vs Rot Error
    sns.scatterplot(data=df, x='gt_depth_m', y='error_rotation_deg', alpha=0.3, ax=axes[1], color='orange')
    sns.regplot(data=df, x='gt_depth_m', y='error_rotation_deg', scatter=False, ax=axes[1], color='red')
    axes[1].set_title('Rotation Error vs. Distance')
    axes[1].set_xlabel('Object Depth (m)')
    axes[1].set_ylabel('Rotation Error (Degrees)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "3_error_vs_depth.png"))
    print("-> Generated Depth Analysis")

def plot_spatial_heatmap(df):
    """4. Top-Down Map of Errors (Where in the room is performance bad?)"""
    # Extract X and Z (Top down view usually uses X as lateral and Z as depth)
    df['gt_x'] = df['gt_translation'].apply(lambda t: t[0])
    df['gt_z'] = df['gt_translation'].apply(lambda t: t[2])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # We use a scatter plot where Color = Error Magnitude
    sc = ax.scatter(df['gt_x'], df['gt_z'], c=df['error_translation_cm'], 
                    cmap='viridis', alpha=0.7, s=20)
    
    plt.colorbar(sc, label='Translation Error (cm)')
    ax.set_title('Spatial Distribution of Errors (Top-Down View)')
    ax.set_xlabel('Lateral Position X (m)')
    ax.set_ylabel('Depth Position Z (m)')
    ax.grid(True, linestyle='--')
    
    # Invert Z axis usually helps visualization if Z grows away from camera
    # ax.invert_yaxis() 

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "4_spatial_heatmap.png"))
    print("-> Generated Spatial Heatmap")

def print_statistics(df):
    print("\n=== STATISTICS ===")
    print(f"Total Images: {len(df)}")
    
    print("\n-- Translation Error (cm) --")
    print(f"Mean:   {df['error_translation_cm'].mean():.2f}")
    print(f"Median: {df['error_translation_cm'].median():.2f}")
    print(f"Min:    {df['error_translation_cm'].min():.2f}")
    print(f"Max:    {df['error_translation_cm'].max():.2f}")
    
    print("\n-- Rotation Error (deg) --")
    print(f"Mean:   {df['error_rotation_deg'].mean():.2f}")
    print(f"Median: {df['error_rotation_deg'].median():.2f}")
    print(f"Max:    {df['error_rotation_deg'].max():.2f}")
    
    # Success Rate (e.g., < 5cm and < 10 deg)
    success = df[(df['error_translation_cm'] < 10.0) & (df['error_rotation_deg'] < 10.0)]
    print(f"\nSuccess Rate (<10cm and <10°): {len(success) / len(df) * 100:.2f}%")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df = load_data(INPUT_FILE)
    
    if df is not None:
        sns.set_style("whitegrid") # Make graphs pretty
        
        print_statistics(df)
        plot_histograms(df)
        plot_cdf(df)
        plot_error_vs_depth(df)
        plot_spatial_heatmap(df)
        
        print(f"\nAll graphs saved to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
INPUT_FILE = "Evaluation_Results/evaluation_metrics.json"
OUTPUT_FOLDER = "Evaluation_Graphs"

# Thresholds for "High Error" (used for the overlap analysis)
THRESH_TRANS_CM = 10.0  # cm
THRESH_ROT_DEG = 10.0   # degrees

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Extract Camera ID (Assuming '4_image.jpg' format -> '4')
    # Adjust logic if your filenames are 'camera4_...'
    df['camera_id'] = df['image'].apply(lambda x: x.split('_')[0])
    
    # Calculate Bias Components (Pred - GT)
    # We do this row by row
    def get_bias(row):
        return np.array(row['pred_translation']) - np.array(row['gt_translation'])

    bias_matrix = np.vstack(df.apply(get_bias, axis=1).values)
    df['bias_x_cm'] = bias_matrix[:, 0] * 100
    df['bias_y_cm'] = bias_matrix[:, 1] * 100
    df['bias_z_cm'] = bias_matrix[:, 2] * 100
    
    df['error_translation_cm'] = df['error_translation_m'] * 100
    
    return df

def plot_camera_bias(df):
    """
    Visualizes the X, Y, Z bias for each camera side-by-side.
    """
    # 1. Group data
    # Filter out massive rotation failures (flips) for cleaner bias calculation
    clean_df = df[df['error_rotation_deg'] < 45.0]
    grouped = clean_df.groupby('camera_id')[['bias_x_cm', 'bias_y_cm', 'bias_z_cm']].mean()
    
    cameras = grouped.index
    x = np.arange(len(cameras))
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 2. Draw Bars
    rects1 = ax.bar(x - width, grouped['bias_x_cm'], width, label='X Bias', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x,         grouped['bias_y_cm'], width, label='Y Bias', color='#66b3ff', edgecolor='black')
    rects3 = ax.bar(x + width, grouped['bias_z_cm'], width, label='Z Bias', color='#99ff99', edgecolor='black')

    # 3. Styling
    ax.set_ylabel('Bias (cm)')
    ax.set_title('Systematic Drift per Camera Axis (Mean)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cam {c}" for c in cameras])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a zero line
    ax.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "5_camera_bias_analysis.png"))
    print("-> Generated Camera Bias Chart")

def plot_error_correlation(df):
    """
    Quadrant Analysis:
    - Bottom Left: Good
    - Top Right: Catastrophic Failure (Both errors high)
    - Top Left / Bottom Right: Partial Failures
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = df['error_rotation_deg']
    y = df['error_translation_cm']
    
    # Scatter plot
    # Color code by Camera ID to see if one camera causes the overlaps
    cameras = df['camera_id'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Matplotlib defaults
    
    for i, cam in enumerate(sorted(cameras)):
        subset = df[df['camera_id'] == cam]
        ax.scatter(subset['error_rotation_deg'], subset['error_translation_cm'], 
                   alpha=0.5, label=f"Cam {cam}", s=15, c=colors[i % len(colors)])

    # Draw Threshold Lines
    ax.axvline(THRESH_ROT_DEG, color='red', linestyle='--', linewidth=2)
    ax.axhline(THRESH_TRANS_CM, color='red', linestyle='--', linewidth=2)
    
    # Label Quadrants
    # 1. Success (Bottom Left)
    ax.text(THRESH_ROT_DEG/2, THRESH_TRANS_CM/2, "Accurate Pose", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='green', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
    
    # 2. Translation Fail (Top Left)
    ax.text(THRESH_ROT_DEG/2, df['error_translation_cm'].max()*0.8, "Right Angle, Wrong Position", 
            ha='center', va='center', fontsize=10, color='orange',
            bbox=dict(facecolor='white', alpha=0.8))

    # 3. Rotation Fail (Bottom Right)
    ax.text(120, THRESH_TRANS_CM/2, "Wrong Angle, Right Position", 
            ha='center', va='center', fontsize=10, color='orange',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # 4. Total Fail (Top Right)
    ax.text(120, df['error_translation_cm'].max()*0.8, "Total Failure", 
            ha='center', va='center', fontsize=10, color='red',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.set_title(f'Correlation: Rotation vs. Translation Error\n(Thresholds: {THRESH_TRANS_CM}cm, {THRESH_ROT_DEG}°)')
    ax.set_xlabel('Rotation Error (Degrees)')
    ax.set_ylabel('Translation Error (cm)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "6_error_correlation_quadrants.png"))
    print("-> Generated Correlation Quadrant Plot")

def print_overlap_stats(df):
    """Calculates the percentages for the quadrants."""
    total = len(df)
    
    # 1. Success
    success = df[(df['error_translation_cm'] <= THRESH_TRANS_CM) & (df['error_rotation_deg'] <= THRESH_ROT_DEG)]
    
    # 2. High Rot, Low Trans (The "Flip" scenario)
    rot_fail = df[(df['error_translation_cm'] <= THRESH_TRANS_CM) & (df['error_rotation_deg'] > THRESH_ROT_DEG)]
    
    # 3. High Trans, Low Rot (The "Drift" scenario)
    trans_fail = df[(df['error_translation_cm'] > THRESH_TRANS_CM) & (df['error_rotation_deg'] <= THRESH_ROT_DEG)]
    
    # 4. Both High
    total_fail = df[(df['error_translation_cm'] > THRESH_TRANS_CM) & (df['error_rotation_deg'] > THRESH_ROT_DEG)]
    
    print("\n=== ERROR OVERLAP ANALYSIS ===")
    print(f"Thresholds: >{THRESH_TRANS_CM}cm and >{THRESH_ROT_DEG}°")
    print(f"Total Frames: {total}")
    print("-" * 40)
    print(f"1. SUCCESS ZONE:         {len(success):<5} ({len(success)/total*100:.1f}%) -> Model works perfectly.")
    print(f"2. ROTATION FLIP ONLY:   {len(rot_fail):<5} ({len(rot_fail)/total*100:.1f}%) -> Center is correct, but angle is wrong (180 flip?).")
    print(f"3. POSITION DRIFT ONLY:  {len(trans_fail):<5} ({len(trans_fail)/total*100:.1f}%) -> Angle is correct, but depth/pos is wrong.")
    print(f"4. TOTAL FAILURE:        {len(total_fail):<5} ({len(total_fail)/total*100:.1f}%) -> Model is completely lost.")
    print("-" * 40)

    # Correlation Coefficient
    corr = df['error_translation_cm'].corr(df['error_rotation_deg'])
    print(f"\nStatistical Correlation (Pearson r): {corr:.4f}")
    if corr < 0.3:
        print(">> Result: WEAK Correlation. Fixing rotation won't necessarily fix position.")
    else:
        print(">> Result: STRONG Correlation. When the model rotates wrongly, it also drifts.")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df = load_data(INPUT_FILE)
    
    if df is not None:
        plot_camera_bias(df)
        plot_error_correlation(df)
        print_overlap_stats(df)
        print(f"\nGraphs saved to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
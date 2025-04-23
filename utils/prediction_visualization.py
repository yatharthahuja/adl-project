"""
Enhanced visualization utilities for frame prediction analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import imageio

def preprocess_frame_for_display(frame, normalize=True):
    """
    Preprocess a frame for display
    """
    # Convert to float for processing
    processed = frame.astype(np.float32)
    
    # Normalize if requested
    if normalize and processed.max() > 1.0:
        processed = processed / 255.0
        
    # Handle different frame types
    if len(processed.shape) == 2:
        # Grayscale: leave as is
        return processed
    elif len(processed.shape) == 3 and processed.shape[2] == 1:
        # Single-channel image with extra dimension
        return processed[:, :, 0]
    elif len(processed.shape) == 3 and processed.shape[2] == 3:
        # RGB:convert to grayscale for consistent visualization
        return np.mean(processed, axis=2)
    else:
        # Unknown format: just use first channel or slice
        return processed.reshape(processed.shape[0], processed.shape[1])

def create_side_by_side_comparison(predicted_frames, actual_frames, output_path="plots/prediction_comparison.gif",
                                    fps=10, title="Predicted vs Actual Frames", max_frames=30):
    """Create a side-by-side GIF comparing predicted and actual frames"""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Make sure we have frames to display
    if not predicted_frames or not actual_frames:
        print("No frames provided for comparison")
        return None
        
    # Determine number of frames
    num_frames = min(len(predicted_frames), len(actual_frames), max_frames)
    
    # Select frames with even spacing if we have too many
    if len(predicted_frames) > max_frames:
        indices = np.linspace(0, len(predicted_frames)-1, max_frames, dtype=int)
        pred_subset = [predicted_frames[i] for i in indices]
        actual_subset = [actual_frames[i] for i in indices]
    else:
        pred_subset = predicted_frames[:num_frames]
        actual_subset = actual_frames[:num_frames]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    
    ax1.set_title('Predicted')
    ax2.set_title('Actual')
    ax1.axis('off')
    ax2.axis('off')
    
    # Preprocess frames for display
    processed_pred = [preprocess_frame_for_display(f) for f in pred_subset]
    processed_actual = [preprocess_frame_for_display(f) for f in actual_subset]
    
    # Initialize with first frame
    im1 = ax1.imshow(processed_pred[0], cmap='gray')
    im2 = ax2.imshow(processed_actual[0], cmap='gray')
    
    def update(frame):
        im1.set_array(processed_pred[frame])
        im2.set_array(processed_actual[frame])
        return [im1, im2]
        
    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000//fps, blit=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    print(f"Saved comparison GIF to: {output_path}")
    return output_path

def create_prediction_error_analysis(predicted_frames,actual_frames,output_path="plots/prediction_error_analysis.png",
                                        num_samples=5):
    """Create a visualization showing error analysis between predicted and actual frames"""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Make sure we have frames to display
    num_frames = min(len(predicted_frames), len(actual_frames))
    if num_frames == 0:
        print("No frames provided for analysis")
        return None
    
    # Calculate metrics for all frames
    mse_values = []
    
    for i in range(num_frames):
        # Preprocess frames
        pred = preprocess_frame_for_display(predicted_frames[i])
        actual = preprocess_frame_for_display(actual_frames[i])
        
        # Calculate MSE (Mean Squared Error)
        diff = np.abs(pred - actual)
        mse = np.mean(diff**2)
        mse_values.append(mse)
    
    avg_mse = np.mean(mse_values)
    
    # Select sample frames for visualization
    indices = np.linspace(0, num_frames-1, min(num_samples, num_frames), dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5*len(indices)))
    fig.suptitle(f'Frame Prediction Analysis (Avg MSE: {avg_mse:.4f})')
    
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Preprocess frames
        pred = preprocess_frame_for_display(predicted_frames[idx])
        actual = preprocess_frame_for_display(actual_frames[idx])
        
        # Display predicted frame
        axes[i, 0].imshow(pred, cmap='gray')
        axes[i, 0].set_title(f'Predicted #{idx}')
        axes[i, 0].axis('off')
        
        # Display actual frame
        axes[i, 1].imshow(actual, cmap='gray')
        axes[i, 1].set_title(f'Actual #{idx}')
        axes[i, 1].axis('off')
        
        # Display difference
        diff = np.abs(pred - actual)
        frame_mse = np.mean(diff**2)
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Difference (MSE: {frame_mse:.4f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Saved error analysis to: {output_path}")
    return output_path, avg_mse

def create_prediction_timeline(predicted_frames, actual_frames, output_path="plots/prediction_timeline.png", num_frames=10):
    """Create a timeline showing how predictions evolve over time"""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Make sure we have frames to display
    total_frames = min(len(predicted_frames), len(actual_frames))
    if total_frames == 0:
        print("No frames provided for timeline")
        return None
    
    # Select frames with even spacing
    indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
    fig.suptitle('Prediction Timeline')
    
    ax1.set_title('Predicted Frames')
    ax2.set_title('Actual Frames')
    
    # Process frames
    selected_pred = [preprocess_frame_for_display(predicted_frames[i]) for i in indices]
    selected_actual = [preprocess_frame_for_display(actual_frames[i]) for i in indices]
    
    # Get dimensions
    height, width = selected_pred[0].shape
    
    # Create a horizontal stack of frames
    def create_strip(frames):
        strip = np.zeros((height, width * len(frames)), dtype=np.float32)
        for i, frame in enumerate(frames):
            strip[:, i*width:(i+1)*width] = frame
        return strip
    
    # Create and display the strips
    pred_strip = create_strip(selected_pred)
    actual_strip = create_strip(selected_actual)
    
    ax1.imshow(pred_strip, cmap='gray')
    ax2.imshow(actual_strip, cmap='gray')
    
    # Add frame indices as x-ticks
    for ax in [ax1, ax2]:
        ax.set_xticks([width * (i + 0.5) for i in range(len(indices))])
        ax.set_xticklabels([f'Frame {idx}' for idx in indices])
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Saved prediction timeline to: {output_path}")
    return output_path

def create_multi_horizon_comparison(input_frames, predicted_frames_by_horizon, actual_frames_by_horizon, output_path="plots/multi_horizon_comparison.png"):
    """Create a comparison showing predictions at different horizons"""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get sorted horizons
    horizons = sorted(predicted_frames_by_horizon.keys())
    if not horizons:
        print("No horizons provided")
        return None
    
    # Select a sample index (middle of sequence)
    if input_frames:
        sample_idx = len(input_frames) // 2
        if sample_idx >= len(input_frames):
            sample_idx = 0
    else:
        sample_idx = 0
    
    # Create figure
    num_rows = len(horizons) + 1  # Input frames + one row per horizon
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4*num_rows))
    fig.suptitle('Prediction Comparison Across Different Horizons')
    
    # Plot input frames in the first row
    if input_frames:
        # Show sequence of input frames
        input_seq = np.hstack([preprocess_frame_for_display(f) for f in input_frames[-3:]])
        axes[0, 0].imshow(input_seq, cmap='gray')
        axes[0, 0].set_title(f'Input Frames')
        axes[0, 0].axis('off')
    
    # Empty plots in top row columns 1-2
    for col in range(1, 3):
        axes[0, col].axis('off')
    
    # For each horizon, show prediction vs actual
    for i, horizon in enumerate(horizons):
        row = i + 1  # Start from second row
        
        # Get frames for this horizon
        pred_frames = predicted_frames_by_horizon[horizon]
        actual_frames = actual_frames_by_horizon[horizon]
        
        if not pred_frames or not actual_frames:
            continue
            
        # Get sample frame
        pred_idx = min(sample_idx, len(pred_frames)-1)
        actual_idx = min(sample_idx, len(actual_frames)-1)
        
        # Preprocess
        pred = preprocess_frame_for_display(pred_frames[pred_idx])
        actual = preprocess_frame_for_display(actual_frames[actual_idx])
        diff = np.abs(pred - actual)
        mse = np.mean(diff**2)
        
        # Plot
        axes[row, 0].imshow(pred, cmap='gray')
        axes[row, 0].set_title(f'Horizon {horizon}: Predicted')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(actual, cmap='gray')
        axes[row, 1].set_title(f'Horizon {horizon}: Actual')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(diff, cmap='hot')
        axes[row, 2].set_title(f'Difference (MSE: {mse:.4f})')
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Saved multi-horizon comparison to: {output_path}")
    return output_path

def plot_mse_by_horizon(horizons, mse_values, output_path="plots/mse_by_horizon.png"):
    """Create a plot showing how MSE varies with prediction horizon"""
    
    if not horizons or not mse_values:
        print("Error: Empty input lists provided")
        return None
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mse_values, 'o-', linewidth=2, markersize=8)
    plt.title('Prediction Error vs. Horizon')
    plt.xlabel('Prediction Horizon (frames)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (h, m) in enumerate(zip(horizons, mse_values)):
        plt.annotate(f"{m:.1f}", (h, m), textcoords="offset points", 
                     xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved MSE vs. horizon plot to: {output_path}")
    return output_path
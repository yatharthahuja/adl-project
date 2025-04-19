"""
Visualization utilities for the simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def prepare_inference_plot_data(input_frames, output_frames, input_output_pairs):
    """Prepare data for the inference plot without creating the actual figure"""
    # Prepare data structures for plotting later
    plot_data = {
        'input_frames_list': sorted(input_frames.keys()),
        'output_frames_list': sorted(output_frames.keys()),
        'input_output_pairs': input_output_pairs
    }
    
    return plot_data

def create_inference_plot_from_data(plot_data, sim_idx=None, save_path='plots'):
    """Create a visualization of inference input-output frame relationships from prepared data"""
    print("Creating inference plot from data")
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Extract data
    input_frames_list = plot_data['input_frames_list']
    output_frames_list = plot_data['output_frames_list']
    input_output_pairs = plot_data['input_output_pairs']
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    # Plot input frames as blue dots on bottom row
    input_y = np.zeros(len(input_frames_list)) + 0.1  # Bottom row
    ax.scatter(input_frames_list, input_y, color='blue', label='Input Frames', s=20)
    
    # Plot output frames as green dots on top row
    output_y = np.zeros(len(output_frames_list)) + 0.9  # Top row
    ax.scatter(output_frames_list, output_y, color='green', label='Output Frames', s=20)
    
    # Draw connection lines between input and output pairs
    for input_frame, output_frame, latency in input_output_pairs:
        ax.plot([input_frame, output_frame], [0.1, 0.9], 'r-', alpha=0.3, linewidth=1)
    
    # Set up the plot appearance
    ax.set_yticks([0.1, 0.9])
    ax.set_yticklabels(['Input', 'Output'])
    ax.set_xlabel('Frame Number')
    ax.set_title('Inference Input-Output Frame Tracking')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Determine file name
    if sim_idx is not None:
        filename = f"inference_tracking_sim_{sim_idx}.png"
    else:
        filename = "inference_tracking.png"
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    
    return os.path.join(save_path, filename)

# For backward compatibility
def create_inference_plot(input_frames, output_frames, input_output_pairs, sim_idx=None, save_path='plots'):
    """Create a visualization of inference input-output frame relationships - wrapper for backward compatibility"""
    plot_data = prepare_inference_plot_data(input_frames, output_frames, input_output_pairs)
    return create_inference_plot_from_data(plot_data, sim_idx, save_path)
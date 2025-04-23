# 

import os
import numpy as np
import cv2
from pathlib import Path

def save_first_frame(dataset_path, video_idx=0, output_dir="debug"):
    """Save the first frame of a video without any modification."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    data = np.load(dataset_path, allow_pickle=True).item()
    videos = np.array(data['videos'])
    
    if video_idx >= len(videos):
        print(f"Error: Video index {video_idx} is out of range. Dataset has {len(videos)} videos.")
        return
    
    # Get first frame of specified video
    first_frame = videos[video_idx, 0]
    
    # Print shape and data type info
    print(f"Frame shape: {first_frame.shape}")
    print(f"Data type: {first_frame.dtype}")
    print(f"Value range: [{first_frame.min()}, {first_frame.max()}]")
    
    # Save frame to file
    output_path = os.path.join(output_dir, f"first_frame_video_{video_idx}.png")
    
    # Handle grayscale vs color images appropriately
    if len(first_frame.shape) == 2:
        # Grayscale image
        print("Frame is grayscale")
        cv2.imwrite(output_path, first_frame)
    else:
        # Color image - handle different formats
        if first_frame.shape[2] == 3:
            # OpenCV expects BGR format
            if first_frame.dtype == np.float32 or first_frame.dtype == np.float64:
                # Normalize if needed
                if first_frame.max() <= 1.0:
                    first_frame = (first_frame * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        elif first_frame.shape[2] == 4:
            # Images with alpha channel
            if first_frame.dtype == np.float32 or first_frame.dtype == np.float64:
                if first_frame.max() <= 1.0:
                    first_frame = (first_frame * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(first_frame, cv2.COLOR_RGBA2BGRA))
    
    print(f"Frame saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Define dataset path
    dataset_path = "data/genesis_videos_mnist_format.npy"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        alt_path = input("Enter the correct dataset path: ")
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            print(f"Dataset not found at {alt_path}, exiting.")
            exit(1)
    
    # Save first frame of first video
    save_first_frame(dataset_path, video_idx=0)
    
    # Save first frames of other videos
    save_first_frame(dataset_path, video_idx=1)
    save_first_frame(dataset_path, video_idx=2)
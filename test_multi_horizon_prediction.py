"""
Test script to evaluate the SwinLSTM frame prediction performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.frame_predictor import FramePredictor
from utils.prediction_visualization import (
    create_side_by_side_comparison,
    create_prediction_error_analysis,
    create_prediction_timeline,
    plot_mse_by_horizon
)

def load_test_video_frames(video_path, num_frames=30, frame_step=5):
    """
    Extract frames from a video file for testing prediction.
    Uses moviepy for video frame extraction.
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        frame_step: Step between frames
        
    Returns:
        List of frames as numpy arrays
    """
    try:
        from moviepy.editor import VideoFileClip
        
        print(f"Loading frames from video: {video_path}")
        clip = VideoFileClip(video_path)
        
        frames = []
        duration = clip.duration
        fps = clip.fps
        total_frames = int(duration * fps)
        
        print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f}s duration")
        
        # Extract frames at regular intervals
        for i in range(0, min(total_frames, num_frames * frame_step), frame_step):
            t = i / fps
            frame = clip.get_frame(t)
            frames.append(frame)
        
        clip.close()
        print(f"Loaded {len(frames)} frames")
        return frames
    
    except ImportError:
        print("Error: moviepy is required for video frame extraction.")
        print("Install with: pip install moviepy")
        return None
    except Exception as e:
        print(f"Error loading video: {e}")
        return None

def test_prediction_accuracy(frames, horizons=[1, 3, 5, 10], model_path=None):
    """
    Test the frame prediction accuracy for different prediction horizons.
    
    Args:
        frames: List of input frames
        horizons: List of prediction horizons to test
        model_path: Path to pretrained model
        
    Returns:
        Dictionary of results
    """
    if len(frames) < 20:
        print("Error: Need at least 20 frames for meaningful testing")
        return None
    
    # Create output directory
    os.makedirs('plots/test_results', exist_ok=True)
    
    # Store results
    results = {}
    mse_values = []
    
    # Test each prediction horizon
    for horizon in horizons:
        print(f"\n===== TESTING PREDICTION HORIZON {horizon} =====")
        
        # Initialize predictor with this horizon
        predictor = FramePredictor(
            model_path=model_path,
            input_frames=4,
            output_frames=horizon,
            image_size=(64, 64)
        )
        
        # We need frames for:
        # 1. Initial input (4 frames)
        # 2. Prediction targets (horizon frames)
        # So total required frames is 4 + horizon
        
        # Store actual and predicted frames
        actual_frames = []
        predicted_frames = []
        frame_mse_values = []
        
        # Use sliding window approach
        for i in range(len(frames) - (4 + horizon)):
            # Get 4 input frames
            input_frames = frames[i:i+4]
            
            # Add frames to predictor
            for frame in input_frames:
                predictor.add_frame(frame)
            
            # Generate predictions
            predictions = predictor.predict_future_frames()
            
            if predictions is None or len(predictions) < horizon:
                print(f"Warning: Prediction failed at frame {i}")
                continue
            
            # Get the target frame for this horizon
            target_frame_idx = i + 4 + (horizon - 1)
            if target_frame_idx >= len(frames):
                break
                
            target_frame = frames[target_frame_idx]
            
            # Convert to grayscale if needed
            if len(target_frame.shape) == 3 and target_frame.shape[2] == 3:
                gray_target = np.mean(target_frame, axis=2).astype(np.uint8)
            else:
                gray_target = target_frame
                
            # Resize to match prediction size
            pil_target = Image.fromarray(gray_target).resize((64, 64), Image.LANCZOS)
            processed_target = np.array(pil_target)
            
            # Use the frame predicted for this horizon
            predicted_frame = predictions[horizon-1]  # Zero-indexed
            
            # Calculate MSE
            diff = (predicted_frame.astype(float) - processed_target.astype(float)) ** 2
            mse = np.mean(diff)
            frame_mse_values.append(mse)
            
            # Store frames for visualization (only a subset to avoid too many)
            if i % 3 == 0:
                actual_frames.append(processed_target)
                predicted_frames.append(predicted_frame)
        
        # Calculate average MSE for this horizon
        avg_mse = np.mean(frame_mse_values) if frame_mse_values else float('inf')
        mse_values.append(avg_mse)
        
        # Create visualizations
        if actual_frames and predicted_frames:
            print(f"Creating visualizations for horizon {horizon}")
            
            comparison_gif = create
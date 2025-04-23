"""
Main script for running the simulation with frame prediction using SwinLSTM
"""

import os
import numpy as np
import time
import queue
from models.vision_model import load_vision_model
from models.frame_predictor import FramePredictor
from simulation.scene import setup_scene
from robot.control import setup_robot_control
from utils.inference_thread import setup_inference_thread
from simulation.runner import run_simulation
from utils.visualization import prepare_inference_plot_data, create_inference_plot_from_data
from utils.prediction_visualization import (
    create_side_by_side_comparison,
    create_prediction_error_analysis,
    create_prediction_timeline,
    create_multi_horizon_comparison,
    plot_mse_by_horizon
)
from robot.kinematics import map_openvla_to_target_pose
from utils.logging import prepare_inference_stats, write_inference_stats

def run_simulation_with_prediction(scene, franka, cam, motors_dof, fingers_dof, inference_queue, output_queue, track_queue,
                                    inference_thread, stop_event, frame_predictor=None, actual_frames=None, predicted_frames=None,
                                    num_frames=400, video_filename='video.mp4', prediction_horizon=5, 
                                    prediction_frequency=1,  # Process every nth frame for prediction
                                    store_visualization_frames=True,  # Whether to store frames for visualization
                                    visualization_frequency=5  # Store every nth frame for visualization):
    """
    Run the main simulation loop with frame prediction
    """
    # Start recording video
    cam.start_recording()
    
    # Set instruction for the robot
    instruction = "Touch the ball"
    
    # Simulation parameters
    DT = 0.01
    last_qpos = franka.get_qpos()  # Start from current pose
    
    # Tracking variables
    input_frames = {}      # {frame_idx: start_time}
    output_frames = {}     # {frame_idx: end_time}
    input_output_pairs = []  # [(input_frame, output_frame, latency)]
    
    # Frame buffer for prediction
    frame_buffer = []
    
    # Frame indices to store for visualization
    viz_frames = []
    
    # Main simulation loop
    for i in range(num_frames):
        t0 = time.time()
        
        # Step the physics simulation
        scene.step()
        
        # Apply control
        franka.control_dofs_position(last_qpos[:-2], motors_dof)
        franka.control_dofs_position(last_qpos[-2:], fingers_dof)

        # Capture current frame
        rgb = cam.render(rgb=True)[0]
        
        # Store frames for visualization
        if store_visualization_frames and i % visualization_frequency == 0:
            viz_frames.append(i)
            if actual_frames is not None:
                actual_frames.append(rgb.copy())
        
        # Process for prediction only on selected frames (to reduce computational load)
        process_prediction = (i % prediction_frequency == 0)
        
        # Update frame predictor if enabled
        if frame_predictor is not None and process_prediction:
            # Add the current frame to the predictor
            frame_predictor.add_frame(rgb)
            
            # After collecting enough frames, start predicting
            if len(frame_buffer) >= 4:  # Need at least 4 frames for SwinLSTM
                # Predict future frames
                future_frames = frame_predictor.predict_future_frames()
                
                if future_frames is not None:
                    # Determine which predicted frame to use based on prediction horizon
                    pred_idx = min(prediction_horizon - 1, len(future_frames) - 1)
                    predicted_frame = future_frames[pred_idx]
                    
                    # Store predicted frame for visualization
                    if store_visualization_frames and i % visualization_frequency == 0 and predicted_frames is not None:
                        predicted_frames.append(predicted_frame.copy())
                    
                    # Use the predicted frame for inference
                    print(f"Using predicted frame (horizon={prediction_horizon}) for inference at frame {i}")
                    inference_queue.put((i, predicted_frame, instruction))
                else:
                    # If prediction fails, use the current frame
                    print(f"Prediction failed at frame {i}, using current frame")
                    inference_queue.put((i, rgb, instruction))
            else:
                # Not enough frames for prediction yet, use current frame
                frame_buffer.append(rgb.copy())
                print(f"Not enough frames for prediction at frame {i}, using current frame")
                inference_queue.put((i, rgb, instruction))
        else:
            # Either frame prediction is disabled or this frame is skipped for prediction
            # Use current frame for inference
            if process_prediction:
                print(f"Processing frame {i} (no prediction)")
                inference_queue.put((i, rgb, instruction))

        # Process tracking info (queue updates only, no file operations)
        while True:
            try:
                track_type, frame_idx, timestamp = track_queue.get_nowait()
                if track_type == "input":
                    input_frames[frame_idx] = timestamp
                elif track_type == "output":
                    output_frames[frame_idx] = timestamp
                    if frame_idx in input_frames:
                        latency = timestamp - input_frames[frame_idx]
                        input_output_pairs.append((frame_idx, i, latency))
                track_queue.task_done()
            except queue.Empty:
                break

        # Handle any completed inferences
        while True:
            try:
                frame_idx, openvla_output, inf_time = output_queue.get_nowait()
                target_pos, target_quat, gripper_state = map_openvla_to_target_pose(openvla_output)
                ee = franka.get_link('hand')
                new_qpos = franka.inverse_kinematics(link=ee, pos=target_pos, quat=target_quat)
                new_qpos[-2:] = gripper_state
                last_qpos = new_qpos
                output_queue.task_done()
            except queue.Empty:
                break

        # Keep real‑time pacing
        elapsed = time.time() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)

    # Save the video, defer all other processing
    cam.stop_recording(save_to_filename=video_filename, fps=60)
    print(f"\n===== SIMULATION COMPLETED: {video_filename} =====")
    
    # Print minimal statistics, save detailed logging for later
    print(f"Total frames: {num_frames}")
    print(f"Input frames processed: {len(input_frames)}")
    print(f"Output frames processed: {len(output_frames)}")
    print(f"Visualization frames stored: {len(viz_frames)}")
    
    # Shut down the inference thread
    stop_event.set()              # tell worker to quit
    inference_queue.put(None)     # wake it if blocked
    inference_thread.join()       # wait for it to exit

    return input_frames, output_frames, input_output_pairs

def run_single_simulation_with_prediction(ramp_z=0.02, ball_radius=0.025, ball_color=(1.0,0.0,0.0,1), ball_z_pos_offset=0.02, ball_x_pos=0.8,
                                            processor=None, model=None, video_filename="video.mp4", num_frames=400,
                                            sim_idx=None, deferred_processing=True, 
                                            warmup_model=True,  # Parameter to control model warm-up
                                            use_frame_prediction=True,  # Parameter to enable/disable frame prediction
                                            prediction_horizon=5,  # Number of frames to predict into the future
                                            create_visualizations=True,  # Whether to create prediction visualizations
                                            prediction_frequency=1,  # Process every Nth frame for prediction
                                            visualization_frequency=5  # Store every Nth frame for visualization
                                            ):
    """Run a single simulation with specified parameters and frame prediction"""
    # Create parameter dictionary for logging
    sim_params = {
        'ramp_z': ramp_z,
        'ball_radius': ball_radius,
        'ball_color': ball_color,
        'ball_z_pos_offset': ball_z_pos_offset,
        'ball_x_pos': ball_x_pos,
        'num_frames': num_frames,
        'video_filename': video_filename,
        'use_frame_prediction': use_frame_prediction,
        'prediction_horizon': prediction_horizon,
        'prediction_frequency': prediction_frequency
    }
    
    # Set up the scene
    scene, franka, cam = setup_scene(
        ramp_z=ramp_z, ball_radius=ball_radius,
        ball_color=ball_color, ball_z_pos_offset=ball_z_pos_offset,
        ball_x_pos=ball_x_pos
    )
    motors_dof, fingers_dof = setup_robot_control(franka)
    
    # Load vision model if not provided
    if processor is None or model is None:
        processor, model = load_vision_model()
        
        # Perform model warm-up if requested
        if warmup_model:
            print("Performing model warm-up with dummy inference...")
            try:
                # Create a dummy image and run inference
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_instruction = "pick up the ball"
                
                # Import to avoid circular import
                from models.vision_model import get_openvla_output
                _ = get_openvla_output(dummy_img, dummy_instruction, processor, model)
                print("Model warm-up complete.")
            except Exception as e:
                print(f"Model warm-up failed: {e}")
    
    # Initialize frame predictor if enabled
    frame_predictor = None
    if use_frame_prediction:
        print("Initializing SwinLSTM frame predictor...")
        # Check if there's a model path in the SwinLSTM directory
        from pathlib import Path
        swinlstm_dir = Path("models/SwinLSTM")
        model_path = swinlstm_dir / "Pretrained" / "trained_model_state_dict"
        
        if model_path.exists():
            print(f"Using pretrained SwinLSTM model from: {model_path}")

        
        frame_predictor = FramePredictor(
            model_path=str(model_path) if model_path.exists() else None,
            input_frames=4,  # SwinLSTM typically uses 4 input frames
            output_frames=prediction_horizon,  # Predict N frames ahead
            image_size=(64, 64)  # SwinLSTM default input size
        )
        
    # Set up inference thread
    inference_queue, output_queue, inference_thread, track_queue, stop_event = \
        setup_inference_thread(processor, model)
    
    # Create directories for outputs
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Store actual and predicted frames for visualization
    actual_frames = [] if create_visualizations else None
    predicted_frames = [] if create_visualizations else None
    
    try:
        # Run the simulation
        input_frames, output_frames, input_output_pairs = run_simulation_with_prediction(
            scene, franka, cam,
            motors_dof, fingers_dof,
            inference_queue, output_queue, track_queue,
            inference_thread, stop_event,
            frame_predictor=frame_predictor,
            actual_frames=actual_frames,
            predicted_frames=predicted_frames,
            num_frames=num_frames,
            video_filename=video_filename,
            prediction_horizon=prediction_horizon,
            prediction_frequency=prediction_frequency,
            store_visualization_frames=create_visualizations,
            visualization_frequency=visualization_frequency
        )
        
        # Prepare data structures for deferred processing
        stats = prepare_inference_stats(input_frames, output_frames, input_output_pairs, sim_params, sim_idx)
        plot_data = prepare_inference_plot_data(input_frames, output_frames, input_output_pairs)
        
        # Create prediction visualizations
        if use_frame_prediction and create_visualizations and actual_frames and predicted_frames:
            print("Creating prediction visualizations...")
            
            # Create file paths with simulation index if provided
            if sim_idx is not None:
                gif_path = f"plots/prediction_comparison_sim_{sim_idx}.gif"
                error_path = f"plots/prediction_error_sim_{sim_idx}.png"
                timeline_path = f"plots/prediction_timeline_sim_{sim_idx}.png"
            else:
                gif_path = "plots/prediction_comparison.gif"
                error_path = "plots/prediction_error.png"
                timeline_path = "plots/prediction_timeline.png"
                
            # Create visualizations
            comparison_gif = create_side_by_side_comparison(
                predicted_frames, actual_frames, output_path=gif_path
            )
            
            error_analysis = create_prediction_error_analysis(
                predicted_frames, actual_frames, output_path=error_path
            )
            
            timeline = create_prediction_timeline(
                predicted_frames, actual_frames, output_path=timeline_path
            )
            
            print(f"Created prediction visualizations:")
            print(f"- GIF comparison: {comparison_gif}")
            print(f"- Error analysis: {error_analysis[0]} (MSE: {error_analysis[1]:.2f})")
            print(f"- Timeline: {timeline}")
        
        # If not deferring processing, write files now
        if not deferred_processing:
            # Create plot of input/output frames
            plot_file = create_inference_plot_from_data(plot_data, sim_idx=sim_idx)
            # Log data to files
            log_file = write_inference_stats(stats)
            
            print(f"Created plot: {plot_file}")
            print(f"Created log: {log_file}")
        
        return stats, plot_data
        
    finally:
        # Clean up in case of early errors
        stop_event.set()
        inference_queue.put(None)
        inference_thread.join()
        del scene, franka, cam
        time.sleep(0.5)

def run_multi_horizon_experiment(
    ramp_z=0.02, ball_radius=0.025, ball_color=(1.0,0.0,0.0,1),
    ball_z_pos_offset=0.02, ball_x_pos=0.8,
    num_frames=400, horizons=[1, 3, 5, 10],
    base_video_filename="video_horizon",
    processor=None, model=None
):
    """
    Run multiple simulations with different prediction horizons to compare results
    """
    # Create output directory
    os.makedirs('plots/multi_horizon', exist_ok=True)
    
    # Load vision model once to avoid reloading for each simulation
    if processor is None or model is None:
        processor, model = load_vision_model()
    
    # Store results for each horizon
    horizon_results = {}
    mse_values = []
    
    # Run simulations for each prediction horizon
    for horizon in horizons:
        print(f"\n===== RUNNING SIMULATION WITH HORIZON {horizon} =====")
        
        # Generate video filename
        video_filename = f"{base_video_filename}_{horizon}.mp4"
        
        # Run simulation with this horizon
        stats, plot_data = run_single_simulation_with_prediction(
            ramp_z=ramp_z,
            ball_radius=ball_radius,
            ball_color=ball_color,
            ball_z_pos_offset=ball_z_pos_offset,
            ball_x_pos=ball_x_pos,
            processor=processor,
            model=model,
            video_filename=video_filename,
            num_frames=num_frames,
            deferred_processing=True,
            use_frame_prediction=True,
            prediction_horizon=horizon,
            create_visualizations=True,
            sim_idx=horizon  # Use horizon as simulation index
        )
        
        # Get visualization files
        comparison_gif = f"plots/prediction_comparison_sim_{horizon}.gif"
        error_path = f"plots/prediction_error_sim_{horizon}.png"
        timeline_path = f"plots/prediction_timeline_sim_{horizon}.png"
        
        # Move visualization files to multi_horizon directory for organization
        import shutil
        shutil.copy(comparison_gif, f"plots/multi_horizon/horizon_{horizon}_comparison.gif")
        shutil.copy(error_path, f"plots/multi_horizon/horizon_{horizon}_error.png")
        shutil.copy(timeline_path, f"plots/multi_horizon/horizon_{horizon}_timeline.png")
        
        # Extract MSE value from error analysis (this is simplified and should be improved)
        # In a real implementation, you'd calculate this directly from the actual frames
        error_analysis = stats.get('avg_latency', 0)  # Using latency as a proxy for MSE
        mse_values.append(error_analysis)
        
        # Store results
        horizon_results[horizon] = stats
    
    # Create comparison of MSE across horizons
    plot_mse_by_horizon(
        horizons, mse_values, 
        output_path="plots/multi_horizon/mse_by_horizon.png"
    )
    
    # Create visual comparison of results across horizons
    # Note: This requires storing the actual predicted frames, which we haven't implemented yet
    # For now, this is a placeholder
    
    print("\n===== MULTI-HORIZON EXPERIMENT COMPLETED =====")
    print(f"Results saved to plots/multi_horizon/")
    
    return horizon_results

def main():
    """Main program entry point"""
    # Optional: Set random seed for reproducibility
    # np.random.seed(42)
    
    # Enable deferred processing to improve performance
    deferred_processing = True
    
    # Run with frame prediction
    use_frame_prediction = True
    
    # Use Genesis CPU backend
    import genesis as gs
    gs.init(backend=gs.cpu)
    
    # Choose experiment type
    experiment_type = "multi_horizon"  # Options: "single", "multi_horizon", "parameter_variations"
    
    if experiment_type == "single":
        # Run a single simulation with prediction
        run_single_simulation_with_prediction(
            video_filename="video_with_prediction.mp4",
            use_frame_prediction=use_frame_prediction,
            create_visualizations=True,
            prediction_horizon=5
        )
    
    elif experiment_type == "multi_horizon":
        # Run multiple simulations with different prediction horizons
        run_multi_horizon_experiment(
            horizons=[1, 3, 5, 10],
            base_video_filename="video_horizon",
            num_frames=200  # Use fewer frames for faster experimentation
        )
    
    elif experiment_type == "parameter_variations":
        # Run multiple simulations with different parameters
        # (This would use a function like run_parameter_variations_with_prediction)
        pass
        
    else:
        print(f"Unknown experiment type: {experiment_type}")

# Run the program if this script is executed directly
if __name__ == "__main__":
    main()
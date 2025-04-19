import os
import numpy as np
import time
from models.vision_model import load_vision_model
from simulation.scene import setup_scene
from robot.control import setup_robot_control
from utils.inference_thread import setup_inference_thread
from simulation.runner import run_simulation
from utils.visualization import prepare_inference_plot_data, create_inference_plot_from_data
from utils.logging import prepare_inference_stats, write_inference_stats

def run_single_simulation(
    ramp_z=0.02, ball_radius=0.025, ball_color=(1.0,0.0,0.0,1),
    ball_z_pos_offset=0.02, ball_x_pos=0.8,
    processor=None, model=None,
    video_filename="video.mp4", num_frames=400,
    sim_idx=None,
    deferred_processing=True,
    warmup_model=True  # New parameter to control model warm-up
):
    """Run a single simulation with specified parameters"""
    # Create parameter dictionary for logging
    sim_params = {
        'ramp_z': ramp_z,
        'ball_radius': ball_radius,
        'ball_color': ball_color,
        'ball_z_pos_offset': ball_z_pos_offset,
        'ball_x_pos': ball_x_pos,
        'num_frames': num_frames,
        'video_filename': video_filename
    }
    
    scene, franka, cam = setup_scene(
        ramp_z=ramp_z, ball_radius=ball_radius,
        ball_color=ball_color, ball_z_pos_offset=ball_z_pos_offset,
        ball_x_pos=ball_x_pos
    )
    motors_dof, fingers_dof = setup_robot_control(franka)
    
    if processor is None or model is None:
        processor, model = load_vision_model()
        
        # Perform an optional explicit warm-up if requested
        if warmup_model:
            print("Performing model warm-up with dummy inference...")
            try:
                # Create a dummy image and run inference
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_instruction = "pick up the ball"
                _ = get_openvla_output(dummy_img, dummy_instruction, processor, model)
                print("Model warm-up complete.")
            except Exception as e:
                print(f"Model warm-up failed: {e}")
    
    inference_queue, output_queue, inference_thread, track_queue, stop_event = \
        setup_inference_thread(processor, model)
    
    try:
        # Run the simulation without doing any file I/O during simulation
        input_frames, output_frames, input_output_pairs = run_simulation(
            scene, franka, cam,
            motors_dof, fingers_dof,
            inference_queue, output_queue, track_queue,
            inference_thread, stop_event,
            num_frames=num_frames,
            video_filename=video_filename
        )
        
        # Prepare data structures for deferred processing
        stats = prepare_inference_stats(input_frames, output_frames, input_output_pairs, sim_params, sim_idx)
        plot_data = prepare_inference_plot_data(input_frames, output_frames, input_output_pairs)
        
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

def run_parameter_variations(num_simulations=10, deferred_processing=True):
    """Run multiple simulations with random parameter combinations"""
    # Create directories for outputs
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load vision-language model once to avoid reloading for each simulation
    processor, model = load_vision_model()
    
    # Available ball colors
    ball_colors = [  # RGB values for 10 different colors
        (1.0, 0.0, 0.0, 1),  # Red
        (1.0, 0.5, 0.0, 1),  # Orange
        (1.0, 1.0, 0.0, 1),  # Yellow
        (0.0, 1.0, 0.0, 1),  # Green
        (0.0, 0.0, 1.0, 1),  # Blue
        (0.0, 1.0, 1.0, 1),  # Cyan
        (0.5, 0.0, 0.5, 1),  # Purple
        (1.0, 0.0, 1.0, 1),  # Pink
        (1.0, 1.0, 1.0, 1),  # White
        (0.0, 0.0, 0.0, 1)   # Black
    ]
    
    color_names = ["red", "orange", "yellow", "green", "blue", 
                   "cyan", "purple", "pink", "white", "black"]
    
    # Table dimensions (needed for calculating ball positions)
    table_x, table_y, table_z = 0.5, 0.8, 0.5
    table_pos_x = 0.8
    
    # Store simulation results
    all_simulation_stats = []
    all_plot_data = []
    
    # Generate random simulations
    for i in range(num_simulations):
        # Generate random parameters within specified ranges
        ramp_z = np.random.uniform(0.04, 0.1)
        ball_radius = np.random.uniform(0.025, 0.05)
        
        # Random color selection
        color_idx = np.random.randint(0, len(ball_colors))
        ball_color = ball_colors[color_idx]
        color_name = color_names[color_idx]
        
        # Random ball z-position
        ball_z_pos_offset = np.random.uniform(0, 0.1)
        
        # Random ball x-position within table bounds
        min_x = (table_pos_x - table_x/4) + ball_radius
        max_x = (table_pos_x + table_x/4) - ball_radius
        ball_x_pos = np.random.uniform(min_x, max_x)
        
        # Format filename with parameter info
        video_filename = f"video2_{i+1}_ramp{ramp_z:.2f}_ball{ball_radius:.2f}_{color_name}_z{ball_z_pos_offset:.2f}_x{ball_x_pos:.2f}.mp4"
        
        print(f"\n===== RUNNING SIMULATION {i+1}/{num_simulations} =====")
        print(f"Params: ramp_z={ramp_z:.3f}, ball_radius={ball_radius:.3f}, " 
              f"ball_color={color_name}, ball_z_offset={ball_z_pos_offset:.3f}, "
              f"ball_x={ball_x_pos:.3f}")
        
        # Run one simulation with these parameters - defer processing
        sim_stats, sim_plot_data = run_single_simulation(
            ramp_z=ramp_z,
            ball_radius=ball_radius,
            ball_color=ball_color,
            ball_z_pos_offset=ball_z_pos_offset,
            ball_x_pos=ball_x_pos,
            processor=processor,
            model=model,
            video_filename=video_filename,
            sim_idx=i+1,  # Pass simulation index
            deferred_processing=deferred_processing
        )
        
        # Store results for batch processing later
        all_simulation_stats.append(sim_stats)
        all_plot_data.append((sim_plot_data, i+1))  # Store with sim index
    
    # Now process all the data at once after simulations are done
    if deferred_processing:
        print("\n===== PROCESSING DEFERRED DATA =====")
        
        # Create all plots
        for plot_data, sim_idx in all_plot_data:
            plot_file = create_inference_plot_from_data(plot_data, sim_idx=sim_idx)
            print(f"Created plot for simulation {sim_idx}: {plot_file}")
        
        # Write all stats files
        for stats in all_simulation_stats:
            log_file = write_inference_stats(stats)
            sim_idx = stats.get('simulation_index')
            print(f"Created log for simulation {sim_idx}: {log_file}")
    
    print(f"\nCompleted {num_simulations} simulations.")
    print(f"Summary report: {summary_report}")
    
    return all_simulation_stats

def main():
    """Main program entry point"""
    # Optional: Set random seed for reproducibility
    # np.random.seed(42)
    
    # Number of simulations to run
    #num_simulations = 20  # Change this to control how many videos to generate
    
    # Enable deferred processing to improve performance
    deferred_processing = True
    
    # Run simulations with random parameter variations
    #run_parameter_variations(num_simulations, deferred_processing)
    run_single_simulation(video_filename="video1.mp4")
# Run the program if this script is executed directly
if __name__ == "__main__":
    import genesis as gs
    gs.init(backend=gs.cpu)
    main()
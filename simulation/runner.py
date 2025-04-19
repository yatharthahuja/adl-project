"""
Simulation runner functions.
"""

import time
import queue
from robot.kinematics import map_openvla_to_target_pose

def run_simulation(
    scene, franka, cam,
    motors_dof, fingers_dof,
    inference_queue, output_queue, track_queue,
    inference_thread, stop_event,       
    num_frames=400, video_filename='video.mp4'
):
    """
    Run the main simulation loop
    
    Args:
        scene: Genesis scene
        franka: Franka robot entity
        cam: Camera entity
        motors_dof: Indices of motor DOFs
        fingers_dof: Indices of finger DOFs
        inference_queue: Queue for sending frames to inference thread
        output_queue: Queue for receiving inference results
        track_queue: Queue for tracking input-output frame relationships
        inference_thread: the Thread running inference
        stop_event: threading.Event to signal the worker to stop
        num_frames: Number of simulation frames to run
        video_filename: Name of the output video file
    """
    # Start recording video
    cam.start_recording()
    
    # Set instruction for the robot
    instruction = "Grab the ball"
    
    # Simulation parameters
    DT = 0.01
    last_qpos = franka.get_qpos()  # Start from current pose
    
    # Tracking variables
    input_frames = {}      # {frame_idx: start_time}
    output_frames = {}     # {frame_idx: end_time}
    input_output_pairs = []  # [(input_frame, output_frame, latency)]
    
    # Main simulation loop
    for i in range(num_frames):
        t0 = time.time()
        
        # Step the physics simulation
        scene.step()
        
        # Apply control
        franka.control_dofs_position(last_qpos[:-2], motors_dof)
        franka.control_dofs_position(last_qpos[-2:], fingers_dof)

        # Capture and queue for inference
        rgb = cam.render(rgb=True)[0]
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

    # Tear down - only save the video, defer all other processing
    cam.stop_recording(save_to_filename=video_filename, fps=60)
    print(f"\n===== SIMULATION COMPLETED: {video_filename} =====")
    
    # Just print minimal statistics, save detailed logging for later
    print(f"Total frames: {num_frames}")
    print(f"Input frames processed: {len(input_frames)}")
    print(f"Output frames processed: {len(output_frames)}")
    
    # ── cleanly shut down the inference thread ──
    stop_event.set()              # tell worker to quit
    inference_queue.put(None)     # wake it if blocked
    inference_thread.join()       # wait for it to exit

    return input_frames, output_frames, input_output_pairs
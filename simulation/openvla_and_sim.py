import genesis as gs
import numpy as np
import cv2
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from torchvision import transforms
import io

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer = False,
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

table = scene.add_entity(
    gs.morphs.Box(
        size=(0.5, 0.8, 0.5),  # Width, depth, height (meters)
        pos=(0.8, 0.0, 0.25),  # 0.5m in front (x), centered (y), half-height z-offset
    ),
    surface=gs.surfaces.Aluminium(color=(0.3, 0.2, 0.1, 1))  # Wood-like color
)

ramp = scene.add_entity(
    gs.morphs.Mesh(
        file="ramp.stl",
        scale=(0.5, 0.1, 0.1),
        pos=(0.8, -0.35, 0.55)
    ),
    surface=gs.surfaces.Aluminium(color=(0.3, 0.2, 0.1, 1))
)

ball = scene.add_entity(
    gs.morphs.Sphere(
        radius =(0.025),
        pos=(0.8, -0.35, 0.65)
    ),
    surface=gs.surfaces.Aluminium(color=(1.0,0.0,0.0,1))
)

# ball = scene.add_entity(
#     gs.morphs.Sphere(
#         radius =(0.025),
#         pos=(0.8, 0.0, 0.55)
#     ),
#     surface=gs.surfaces.Aluminium(color=(1.0,0.0,0.0,1))
# )

# ball = scene.add_entity(
#     gs.morphs.Box(
#         size=(0.05, 0.05, 0.05),  # Width, depth, height (meters)
#         pos=(0.8, 0.0, 0.55),  # 0.5m in front (x), centered (y), half-height z-offset
#     ),
#     surface=gs.surfaces.Aluminium(color=(1.0,0.0,0.0,1))  # Wood-like color
# )

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (1.8, 0.5, 1.5),
    lookat = (0.8, 0.0, 0.5),
    fov    = 30,
    GUI    = False,
)

scene.build()

# Define joint names and get their dof indices
jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]

dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

# Set control gains (if necessary)
franka.set_dofs_kp(
    kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local=dofs_idx,
)

franka.set_dofs_kv(
    kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local=dofs_idx,
)

desired_qpos = np.array([
    0.0,         # Joint 1
    -0.785398,   # Joint 2 (≈ -45°)
    0.0,         # Joint 3
    -2.0943951,     # Joint 4 (≈ -120°)
    0.0,         # Joint 5
    2.0943951,      # Joint 6 (≈  120°)
    0.0,         # Joint 7
    0.03,        # Finger Joint 1 (example value; adjust as needed)
    0.03         # Finger Joint 2 (example value; adjust as needed)
])

# Send the joint configuration to the robot
franka.set_qpos(desired_qpos)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")

instruction = "Touch red sphere"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# Define joint indices
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# Set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local=np.arange(9),
)

franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local=np.arange(9),
)

# Function to get OpenVLA output


def get_openvla_output(image_input, instruction):
    # Step 1: Load the image from a file path or numpy array
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise ValueError("Unsupported input type. Must be file path or numpy array.")

    # Step 2: Resize the image to 224x224 (if needed)
    image = image.resize((224, 224))
    
    # At this point, 'image' is still a PIL Image.
    # No need to manually convert it to a tensor; let the processor handle preprocessing.

    # Step 3: Format the prompt
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Step 4: Pass the PIL image to the processor (wrap in a list if the processor expects an iterable)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    return action

import numpy as np
from scipy.spatial.transform import Rotation as R

def map_openvla_to_target_pose(openvla_output):
    # Unpack the OpenVLA output
    delta_xyz = openvla_output[:3]       # Translation delta
    delta_rpy = openvla_output[3:6]        # Rotation delta (assumed to be in radians)
    gripper_state = openvla_output[6]      # Gripper state

    # Current state (example values)
    current_pose = np.array([0.65, 0.0, 0.25])
    # Define current orientation as a quaternion.
    # The quaternion format here is [x, y, z, w].
    current_quat = np.array([0, 1, 0, 0])
    
    # Calculate new target position
    target_pos = current_pose + delta_xyz

    # Convert current quaternion to a Rotation object
    current_rot = R.from_quat(current_quat)
    
    # Convert the delta in Euler angles to a Rotation object.
    # Here, we assume `delta_rpy` is given in radians.
    delta_rot = R.from_euler('xyz', delta_rpy, degrees=False)
    
    # Combine rotations: the new orientation is the delta rotation applied to the current rotation.
    target_rot = delta_rot * current_rot  # Note: order matters depending on your convention.
    
    # Convert the result back to a quaternion [x, y, z, w]
    target_quat = target_rot.as_quat()

    return target_pos, target_quat, gripper_state


import threading
import queue
import time

# Queue for sending camera frames to inference thread
inference_queue = queue.Queue()
output_queue = queue.Queue()

# Inference thread function
def inference_worker():

    while True:
        item = inference_queue.get()
        if item is None:
            inference_queue.task_done()
            break

        frame_idx, rgb, instruction = item
        try:
            t0 = time.time()
            openvla_output = get_openvla_output(rgb, instruction)
            t1 = time.time()
            inf_time = t1 - t0
            output_queue.put((frame_idx, openvla_output, inf_time))
        except Exception as e:
            print(f"[Inference] ERROR on frame {frame_idx}: {e!r}")
        finally:
            inference_queue.task_done()


# Start the inference thread
inference_thread = threading.Thread(target=inference_worker)
inference_thread.start()

cam.start_recording()
openvla_update_interval = 8
instruction = "Grab red ball"
import time

step_times = []
output_queue_counter = 0
DT = 0.01

last_qpos = franka.get_qpos()  # start from current pose

for i in range(600):
    t0 = time.time()
    scene.step()

    # always send the last known target to keep the controller alive
    franka.control_dofs_position(last_qpos[:-2], motors_dof)
    franka.control_dofs_position(last_qpos[-2:], fingers_dof)

    # fire off inference request every step
    rgb = cam.render(rgb=True)[0]
    inference_queue.put((i, rgb, instruction))

    # whenever an inference arrives, update last_qpos
    while True:
        try:
            frame_idx, openvla_output, _ = output_queue.get_nowait()
        except queue.Empty:
            break
        target_pos, target_quat, gripper_state = map_openvla_to_target_pose(openvla_output)
        ee = franka.get_link('hand')
        new_qpos = franka.inverse_kinematics(link=ee, pos=target_pos, quat=target_quat)
        new_qpos[-2:] = gripper_state
        last_qpos = new_qpos   # ← store it

    # throttle to DT
    elapsed = time.time() - t0
    if elapsed < DT:
        time.sleep(DT - elapsed)

# Signal the inference thread to terminate
inference_queue.put(None)
inference_thread.join()


cam.stop_recording(save_to_filename='video_threading2.mp4', fps=60)
print(f"Out of 400 frames, OpenVLA was only able to do inference for: {output_queue_counter}")
"""
Robot kinematics functions for pose mapping.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def map_openvla_to_target_pose(openvla_output):
    """
    Map OpenVLA output to robot target pose
    
    Args:
        openvla_output: Vector of action parameters from OpenVLA
        
    Returns:
        target_pos: 3D target position
        target_quat: Quaternion target orientation
        gripper_state: Gripper opening value
    """
    # Unpack the OpenVLA output
    delta_xyz = openvla_output[:3]      # Translation delta
    delta_rpy = openvla_output[3:6]     # Rotation delta (in radians)
    gripper_state = openvla_output[6]   # Gripper state

    # Current state (example values - should be replaced with actual robot state)
    current_pose = np.array([0.65, 0.0, 0.25])
    current_quat = np.array([0, 1, 0, 0])  # Quaternion format [x, y, z, w]
    
    # Calculate new target position
    target_pos = current_pose + delta_xyz

    # Convert current quaternion to a Rotation object
    current_rot = R.from_quat(current_quat)
    
    # Convert the delta in Euler angles to a Rotation object
    delta_rot = R.from_euler('xyz', delta_rpy, degrees=False)
    
    # Combine rotations: the new orientation is the delta rotation applied to the current rotation
    target_rot = delta_rot * current_rot
    
    # Convert the result back to a quaternion [x, y, z, w]
    target_quat = target_rot.as_quat()

    return target_pos, target_quat, gripper_state
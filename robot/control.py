"""
Robot control functions.
"""

import numpy as np
from config import ROBOT_KP, ROBOT_KV, HOME_QPOS

def setup_robot_control(franka):
    """Configure robot control parameters and initial pose"""
    # Define joint names and get their DOF indices
    jnt_names = [
        'joint1', 'joint2', 'joint3', 'joint4', 
        'joint5', 'joint6', 'joint7',
        'finger_joint1', 'finger_joint2',
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]
    
    # Set control gains for position control
    franka.set_dofs_kp(
        kp=np.array(ROBOT_KP),
        dofs_idx_local=dofs_idx,
    )
    franka.set_dofs_kv(
        kv=np.array(ROBOT_KV),
        dofs_idx_local=dofs_idx,
    )

    # Apply initial pose from config
    franka.set_qpos(np.array(HOME_QPOS))
    
    # Define motor and finger DOF indices for control
    motors_dof = np.arange(7)  # First 7 DOFs for arm joints
    fingers_dof = np.arange(7, 9)  # Last 2 DOFs for gripper

    return motors_dof, fingers_dof
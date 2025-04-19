"""
Configuration parameters for the simulation environment.
"""

# Simulation parameters
SIM_DT = 0.01  # Simulation time step (seconds)
GRAVITY = (0, 0, -9.81)  # Gravity vector (x, y, z)

# Camera settings
CAMERA_RES = (640, 480)
CAMERA_FOV = 30
CAMERA_POS = (1.8, 0.5, 1.5)
CAMERA_LOOKAT = (0.8, 0.0, 0.5)

# Viewer settings
VIEWER_RES = (1280, 960)
VIEWER_CAMERA_POS = (3.5, 0.0, 2.5)
VIEWER_CAMERA_LOOKAT = (0.0, 0.0, 0.5)
VIEWER_CAMERA_FOV = 40
VIEWER_MAX_FPS = 60

# Robot control parameters
ROBOT_KP = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]  # Position control gains
ROBOT_KV = [450, 450, 350, 350, 200, 200, 200, 10, 10]          # Velocity control gains

# Initial joint configuration (home position)
HOME_QPOS = [
    0.0,         # Joint 1
    -0.785398,   # Joint 2 (≈ -45°)
    0.0,         # Joint 3
    -2.0943951,  # Joint 4 (≈ -120°)
    0.0,         # Joint 5
    2.0943951,   # Joint 6 (≈ 120°)
    0.0,         # Joint 7
    0.03,        # Finger Joint 1 (gripper)
    0.03         # Finger Joint 2 (gripper)
]

# Paths
LOG_DIR = 'logs'
PLOT_DIR = 'plots'
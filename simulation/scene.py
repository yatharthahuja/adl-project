"""
Scene setup functions for the simulation environment.
"""

import genesis as gs
import numpy as np
from config import (
    SIM_DT, GRAVITY, CAMERA_RES, CAMERA_FOV, CAMERA_POS, CAMERA_LOOKAT,
    VIEWER_RES, VIEWER_CAMERA_POS, VIEWER_CAMERA_LOOKAT, VIEWER_CAMERA_FOV, VIEWER_MAX_FPS
)

def setup_scene(ramp_z=0.02, ball_radius=0.025, ball_color=(1.0, 0.0, 0.0, 1),
                ball_z_pos_offset=0.02, ball_x_pos=0.8):
    """Create and configure the simulation scene with all objects"""
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=SIM_DT, gravity=GRAVITY),
        viewer_options=gs.options.ViewerOptions(
            res=VIEWER_RES,
            camera_pos=VIEWER_CAMERA_POS,
            camera_lookat=VIEWER_CAMERA_LOOKAT,
            camera_fov=VIEWER_CAMERA_FOV,
            max_FPS=VIEWER_MAX_FPS,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            ambient_light=(0.1, 0.1, 0.1),
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add table
    table_x, table_y, table_z = 0.5, 0.8, 0.5
    table_pos_x, table_pos_y, table_pos_z = 0.8, 0.0, table_z/2
    table = scene.add_entity(
        gs.morphs.Box(
            size=(table_x, table_y, table_z),  # Width, depth, height (meters)
            pos=(table_pos_x, table_pos_y, table_pos_z), 
        ),
        surface=gs.surfaces.Aluminium(color=(0.3, 0.2, 0.1, 1))  # Wood-like color
    )

    # Add ramp on table with parametrized z-height
    ramp_x, ramp_y = table_x, 0.1  # Fixed x and y dimensions
    ramp_x_pos = table_pos_x
    ramp_y_pos = -((table_y/2) - (ramp_y/2))
    ramp_z_pos = table_z + (ramp_z/2)
    ramp = scene.add_entity(
        gs.morphs.Mesh(
            file="ramp.stl",
            scale=(ramp_x, ramp_y, ramp_z),
            pos=(ramp_x_pos, ramp_y_pos, ramp_z_pos)
        ),
        surface=gs.surfaces.Aluminium(color=(0.3, 0.2, 0.1, 1))
    )

    # Add ball with parametrized properties
    # Y position based on table and ball radius
    ball_y_pos = -((table_y/2) - ball_radius)
    # Z position based on table height, ramp height, offset and ball radius
    ball_z_pos = table_z + ramp_z + ball_z_pos_offset + ball_radius
    
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=ball_radius,
            pos=(ball_x_pos, ball_y_pos, ball_z_pos)
        ),
        surface=gs.surfaces.Aluminium(color=ball_color)
    )

    # Add Franka robot
    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )

    # Add camera for perception
    cam = scene.add_camera(
        res=CAMERA_RES,
        pos=CAMERA_POS,
        lookat=CAMERA_LOOKAT,
        fov=CAMERA_FOV,
        GUI=False,
    )

    scene.build()
    return scene, franka, cam
import genesis as gs
import time

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer = False,
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
        pos=(0.5, 0.0, 0.25),  # 0.5m in front (x), centered (y), half-height z-offset
    ),
    surface=gs.surfaces.Aluminium(color=(0.3, 0.2, 0.1, 1))  # Wood-like color
)

ramp = scene.add_entity(
    gs.morphs.Mesh(
        file="ramp.stl",
        scale=(0.5, 0.1, 0.01),
        pos=(0.5, -0.35, 0.55)
    ),
    surface=gs.surfaces.Aluminium(color=(0.3, 0.2, 0.1, 1))
)

ball = scene.add_entity(
    gs.morphs.Sphere(
        radius =(0.05),
        pos=(0.5, -0.35, 0.65)
    ),
    surface=gs.surfaces.Aluminium(color=(1.0,0.0,0.0,1))
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (1.5, 0.5, 1.5),
    lookat = (0.5, 0.0, 0.5),
    fov    = 30,
    GUI    = False,
)

scene.build()

# render rgb, depth, segmentation, and normal
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording()
import numpy as np

for i in range(5):
    scene.step()
    cam.render()
cam.stop_recording(save_to_filename='video_box_ramp2.mp4', fps=60)
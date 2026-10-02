"""Renders the film's 3D chapter with Blender Cycles (soft shadows, bounced light).

    blender -b -P render_models.py -- hero  [first_frame last_frame]   # the plan standing up, then turning
    blender -b -P render_models.py -- grid                            # six apartments in one still
    blender -b -P render_models.py -- cover                           # the two cover stills (landscape and portrait)

Reads renders/models.json (written by export_models.js); writes PNG frames into renders/.
One Blender unit is one 55 cm cell.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy

FILM_DIRECTORY = Path(__file__).resolve().parent
RENDER_DIRECTORY = FILM_DIRECTORY / "renders"
FRAMES_PER_SECOND = 30
HERO_FRAME_COUNT = 330
RISE_START_SECONDS = 0.6
RISE_DURATION_SECONDS = 3.2
FINAL_ELEVATION_RADIANS = 0.88
FINAL_YAW_RADIANS = 0.7
TURN_RADIANS_PER_SECOND = 0.22
GROUND_COLOUR = "#20262e"
FLOOR_THICKNESS = 0.32


def linear_channel(value):
    value = value / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_colour(hex_colour):
    return tuple(linear_channel(int(hex_colour[index:index + 2], 16)) for index in (1, 3, 5)) + (1.0,)


def smooth_step(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


materials = {}


def material_for(hex_colour, alpha):
    key = (hex_colour, alpha < 1)
    if key not in materials:
        material = bpy.data.materials.new(name=f"{hex_colour}{'-glass' if alpha < 1 else ''}")
        material.use_nodes = True
        shader = material.node_tree.nodes["Principled BSDF"]
        shader.inputs["Base Color"].default_value = linear_colour(hex_colour)
        if alpha < 1:
            shader.inputs["Transmission Weight"].default_value = 1.0
            shader.inputs["Roughness"].default_value = 0.08
            shader.inputs["IOR"].default_value = 1.2
        else:
            shader.inputs["Roughness"].default_value = 0.8
        materials[key] = material
    return materials[key]


def build_model(model, name):
    """One mesh per material, all parented to an empty at the middle of the plan."""
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    middle_x, middle_y = model["columnCount"] / 2, model["rowCount"] / 2
    meshes = {}
    for solid in model["solids"]:
        key = (solid["colour"], solid["alpha"] < 1)
        if key not in meshes:
            meshes[key] = bmesh.new()
        mesh = meshes[key]
        height = solid["z1"] - solid["z0"]
        if solid.get("cylinder"):
            created = bmesh.ops.create_cone(mesh, cap_ends=True, cap_tris=False, segments=48, radius1=solid["radius"], radius2=solid["radius"], depth=height)
            centre = (solid["centreX"] - middle_x, -(solid["centreY"] - middle_y), (solid["z0"] + solid["z1"]) / 2)
            bmesh.ops.translate(mesh, verts=created["verts"], vec=centre)
        else:
            created = bmesh.ops.create_cube(mesh, size=1.0)
            bmesh.ops.scale(mesh, verts=created["verts"], vec=(solid["x1"] - solid["x0"], solid["y1"] - solid["y0"], height))
            centre = ((solid["x0"] + solid["x1"]) / 2 - middle_x, -((solid["y0"] + solid["y1"]) / 2 - middle_y), (solid["z0"] + solid["z1"]) / 2)
            bmesh.ops.translate(mesh, verts=created["verts"], vec=centre)
    for (hex_colour, is_glass), mesh in meshes.items():
        data = bpy.data.meshes.new(f"{name}-{hex_colour}")
        mesh.to_mesh(data)
        mesh.free()
        part = bpy.data.objects.new(f"{name}-{hex_colour}", data)
        part.data.materials.append(material_for(hex_colour, 0.5 if is_glass else 1.0))
        part.parent = root
        bpy.context.collection.objects.link(part)
    return root


def prepare_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = device_type
            preferences.get_devices()
            for device in preferences.devices:
                device.use = device.type != "CPU"
            scene.cycles.device = "GPU"
            break
        except TypeError:
            continue
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = True
    scene.render.use_persistent_data = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 4
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"

    # a dim cool sky fills the shadows; one broad sun makes them soft-edged
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.68, 0.9, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22
    scene.world = world
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.6
    sun_data.angle = math.radians(11)
    sun_data.color = (1.0, 0.96, 0.9)
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(58), 0, math.radians(-42))
    scene.collection.objects.link(sun)

    ground_mesh = bpy.data.meshes.new("ground")
    ground_bmesh = bmesh.new()
    bmesh.ops.create_grid(ground_bmesh, x_segments=1, y_segments=1, size=400)
    ground_bmesh.to_mesh(ground_mesh)
    ground_bmesh.free()
    ground = bpy.data.objects.new("ground", ground_mesh)
    ground.location = (0, 0, -FLOOR_THICKNESS - 0.001)
    ground.data.materials.append(material_for(GROUND_COLOUR, 1.0))
    scene.collection.objects.link(ground)

    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def aim_camera(camera, elevation_radians, ortho_scale, shift_x, shift_y):
    distance = 200
    camera.location = (0, -distance * math.cos(elevation_radians), distance * math.sin(elevation_radians))
    camera.rotation_euler = (math.pi / 2 - elevation_radians, 0, 0)
    camera.data.ortho_scale = ortho_scale
    camera.data.shift_x = shift_x
    camera.data.shift_y = shift_y
    camera.data.clip_end = 1000


def render_hero(models, first_frame, last_frame):
    scene, camera = prepare_scene()
    root = build_model(models[0], "hero")
    diagonal = math.hypot(models[0]["columnCount"], models[0]["rowCount"])
    output_directory = RENDER_DIRECTORY / "hero"
    output_directory.mkdir(parents=True, exist_ok=True)
    for frame in range(first_frame, last_frame + 1):
        seconds = frame / FRAMES_PER_SECOND
        rise = smooth_step((seconds - RISE_START_SECONDS) / RISE_DURATION_SECONDS)
        yaw = FINAL_YAW_RADIANS * rise + max(0.0, seconds - RISE_START_SECONDS - RISE_DURATION_SECONDS) * TURN_RADIANS_PER_SECOND
        elevation = math.pi / 2 + (FINAL_ELEVATION_RADIANS - math.pi / 2) * rise
        root.rotation_euler = (0, 0, -yaw)
        root.scale = (1, 1, 0.02 + 0.98 * rise)
        # the model sits left of centre, leaving the right of the frame for captions
        aim_camera(camera, elevation, diagonal * 2.05, 0.15, 0.03)
        scene.render.filepath = str(output_directory / f"{frame:04d}.png")
        bpy.ops.render.render(write_still=True)


def render_grid(models):
    scene, camera = prepare_scene()
    scene.cycles.samples = 256
    spacing_x, spacing_y = 32, 23
    for index, model in enumerate(models[1:7]):
        root = build_model(model, f"model-{index}")
        root.location = ((index % 3 - 1) * spacing_x, (0.5 - index // 3) * spacing_y, 0)
        root.rotation_euler = (0, 0, -(0.6 + index * 0.9))
    aim_camera(camera, FINAL_ELEVATION_RADIANS, spacing_x * 3.25, 0, -0.02)
    scene.render.filepath = str(RENDER_DIRECTORY / "grid.png")
    bpy.ops.render.render(write_still=True)


def render_cover(models):
    scene, camera = prepare_scene()
    scene.cycles.samples = 256
    root = build_model(models[0], "cover")
    root.rotation_euler = (0, 0, -0.95)
    diagonal = math.hypot(models[0]["columnCount"], models[0]["rowCount"])
    # landscape: the model on the right; portrait: the model in the lower half
    for name, width, height, ortho_scale, shift_x, shift_y in (("landscape", 1920, 1080, diagonal * 2.3, -0.275, 0.0), ("portrait", 1080, 1920, diagonal * 1.95, 0.0, 0.2)):
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        aim_camera(camera, FINAL_ELEVATION_RADIANS, ortho_scale, shift_x, shift_y)
        scene.render.filepath = str(RENDER_DIRECTORY / f"cover-{name}.png")
        bpy.ops.render.render(write_still=True)


arguments = sys.argv[sys.argv.index("--") + 1:]
all_models = json.loads((RENDER_DIRECTORY / "models.json").read_text(encoding="utf-8"))
if arguments[0] == "hero":
    first = int(arguments[1]) if len(arguments) > 1 else 0
    last = int(arguments[2]) if len(arguments) > 2 else HERO_FRAME_COUNT - 1
    render_hero(all_models, first, last)
elif arguments[0] == "cover":
    render_cover(all_models)
else:
    render_grid(all_models)

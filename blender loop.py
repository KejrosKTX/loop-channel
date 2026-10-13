"""
blender_loop.py - renders ONE seamless 3D loop. Runs INSIDE Blender (headless):

    blender -b -P blender_loop.py -- package.json out_dir width height fps frames samples

Every animation is a periodic function of t in [0, 1). Frame number `frames` would be
identical to frame 0, so the clip loops with no visible seam.

The maths (scene layout, motion, camera framing) uses no bpy, so pipeline.py imports it
for validation and it can be unit-tested outside Blender.
"""
import json
import math
import os
import re
import sys

try:
    import bpy
    from mathutils import Vector
except ImportError:  # normal Python (pipeline.py / tests)
    bpy = None

TAU = 2 * math.pi
SCENE_TYPES = ["wave_grid", "cube_flip_wave", "pendulum_wave", "ring_gyro", "spiral_ripple"]
SHAPES = ["cube", "sphere", "cylinder"]  # wave_grid only
MATERIALS = ["glossy", "matte", "metallic", "neon"]
CAMERAS = ["front_iso", "top_down", "low_angle", "close"]
DEFAULT_PALETTE = ["#ff5ea8", "#ffb347", "#4cc9f0"]

# ----------------------------------------------------------------------------- params


def _hex(v, default):
    if isinstance(v, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", v.strip()):
        v = v.strip()
        return v if v.startswith("#") else "#" + v
    return default


def _int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def _luma(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _scale_hex(h, f):
    h = h.lstrip("#")
    return "#" + "".join("%02x" % max(0, min(255, int(int(h[i:i + 2], 16) * f))) for i in (0, 2, 4))


def clean_scene_params(s):
    """Clamp anything the AI returned to values the renderer can handle."""
    s = s if isinstance(s, dict) else {}
    pal = [_hex(c, None) for c in (s.get("palette") if isinstance(s.get("palette"), list) else [])]
    pal = [c for c in pal if c][:4]
    while len(pal) < 3:
        pal.append(DEFAULT_PALETTE[len(pal)])
    bg = _hex(s.get("background"), "#0b0d14")
    if _luma(bg) > 0.25:  # keep the stage dark so colours pop
        bg = _scale_hex(bg, 0.2)
    pick = lambda key, options, default: s.get(key) if s.get(key) in options else default
    return {
        "type": pick("type", SCENE_TYPES, "wave_grid"),
        "shape": pick("shape", SHAPES, "cube"),
        "material": pick("material", MATERIALS, "glossy"),
        "camera": pick("camera", CAMERAS, "front_iso"),
        "palette": pal,
        "background": bg,
        "grid": _int(s.get("grid"), 6, 21, 13),
        "speed": _int(s.get("speed"), 1, 3, 1),
        "seed": _int(s.get("seed"), 0, 99999, 0),
    }


# ----------------------------------------------------------------------------- math


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def rot_xyz(a, b, c, v):
    """Blender 'XYZ' euler = Rz @ Ry @ Rx applied to v."""
    x, y, z = v
    y, z = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
    x, z = x * math.cos(b) + z * math.sin(b), -x * math.sin(b) + z * math.cos(b)
    x, y = x * math.cos(c) - y * math.sin(c), x * math.sin(c) + y * math.cos(c)
    return (x, y, z)


# Each scene returns {"items": [...], "pose": f(i, t) -> (loc, rot, scale), "target_z", "floor_z"}
# item = {"shape": cube|sphere|cylinder|torus, "cval": 0..1 palette position,
#         "mat": "palette"|"neutral", optional "bevel", "major", "minor", "rad"}


def scene_wave_grid(p):
    n, k, shape = p["grid"], p["speed"], p["shape"]
    half = (n - 1) / 2
    rmax = math.hypot(half, half) or 1.0
    items, pos = [], []
    for ix in range(n):
        for iy in range(n):
            x, y = ix - half, iy - half
            r = math.hypot(x, y)
            items.append({"shape": shape, "cval": r / rmax})
            pos.append((x, y, r))

    def pose(i, t):
        x, y, r = pos[i]
        h = 0.25 + 1.7 * (0.5 + 0.5 * math.sin(TAU * (k * t - r * 0.125)))
        if shape == "sphere":
            return (x, y, 0.45 + h * 0.9), (0, 0, 0), (0.84, 0.84, 0.84)
        return (x, y, h / 2), (0, 0, 0), (0.84, 0.84, h)

    return {"items": items, "pose": pose, "target_z": 0.8, "floor_z": 0.0}


def scene_cube_flip(p):
    n, k = max(5, min(14, p["grid"])), p["speed"]
    half = (n - 1) / 2
    items, meta = [], []
    for ix in range(n):
        for iy in range(n):
            c = (ix + iy) / (2 * (n - 1))
            items.append({"shape": "cube", "cval": c, "bevel": 0.07})
            meta.append((ix - half, iy - half, c * 0.7))

    def pose(i, t):
        x, y, ph = meta[i]
        u = k * t - ph
        kk = math.floor(u)
        s = smoothstep((u - kk) / 0.45)
        ang = (math.pi / 2) * (kk + s)  # whole quarter-turns per loop -> cube looks identical
        lift = 0.45 * math.sin(math.pi * s)
        return (x, y, 0.4 + lift), (ang, 0, 0), (0.8, 0.8, 0.8)

    return {"items": items, "pose": pose, "target_z": 0.5, "floor_z": 0.0}


def scene_pendulum(p):
    count = 12 + int((p["grid"] - 6) / 15 * 8)  # 12..20
    L, spacing, amp = 4.0, 0.62, 0.55
    pivot_z = L + 0.9
    base = 4 + 2 * p["speed"]  # integer swings per loop; bob i does base+i
    width = (count - 1) * spacing
    items = [{"shape": "sphere", "cval": i / (count - 1)} for i in range(count)]
    items += [{"shape": "cylinder", "cval": 0, "mat": "neutral"} for _ in range(count)]
    items.append({"shape": "cylinder", "cval": 0, "mat": "neutral"})  # top bar

    def pose(i, t):
        if i == 2 * count:
            return (0, 0, pivot_z), (0, math.pi / 2, 0), (0.08, 0.08, width + 0.8)
        b = i % count
        x = (b - (count - 1) / 2) * spacing
        th = amp * math.sin(TAU * (base + b) * t)
        if i < count:
            return (x, L * math.sin(th), pivot_z - L * math.cos(th)), (0, 0, 0), (0.6, 0.6, 0.6)
        mid = (x, L * math.sin(th) / 2, pivot_z - L * math.cos(th) / 2)
        return mid, (th, 0, 0), (0.04, 0.04, L)

    return {"items": items, "pose": pose, "target_z": 2.5, "floor_z": 0.0}


def scene_ring_gyro(p):
    n = max(4, min(9, p["grid"] // 2))
    k = p["speed"]
    items, ring_cfg = [], []
    for i in range(n):
        R = 1.0 + i * 0.45
        ring_cfg.append((R, (1 if i % 2 == 0 else -1) * (1 + i % 2) * k, i * math.pi / (2 * n)))
        items.append({"shape": "torus", "cval": i / (n - 1), "major": R, "minor": 0.07, "rad": R + 0.1})
    for i in range(n):
        items.append({"shape": "sphere", "cval": i / (n - 1), "rad": 0.2})
    rmax = 1.0 + (n - 1) * 0.45

    def ring_rot(i, t):
        _, m, tilt = ring_cfg[i]
        return (TAU * m * t, tilt, 0.0)

    def pose(i, t):
        if i < n:
            return (0, 0, 0), ring_rot(i, t), (1, 1, 1)
        b = i - n
        R = ring_cfg[b][0]
        phi = TAU * ((1 + b % 3) * t + b / n)
        loc = rot_xyz(*ring_rot(b, t), (R * math.cos(phi), R * math.sin(phi), 0.0))
        return loc, (0, 0, 0), (0.34, 0.34, 0.34)

    return {"items": items, "pose": pose, "target_z": 0.0, "floor_z": -(rmax + 0.9)}


def scene_spiral(p):
    n = 120 + p["grid"] * 12
    k = p["speed"]
    ga = math.pi * (3 - math.sqrt(5))
    c = 0.33
    items = [{"shape": "sphere", "cval": i / (n - 1)} for i in range(n)]

    def pose(i, t):
        r = c * math.sqrt(i + 0.5)
        a = i * ga + TAU * t
        ph = TAU * k * t - r * 1.1
        s = 0.2 + 0.2 * (0.5 + 0.5 * math.sin(ph))
        return (r * math.cos(a), r * math.sin(a), 0.7 * math.sin(ph)), (0, 0, 0), (s, s, s)

    return {"items": items, "pose": pose, "target_z": 0.0, "floor_z": -1.3}


SCENES = {
    "wave_grid": scene_wave_grid,
    "cube_flip_wave": scene_cube_flip,
    "pendulum_wave": scene_pendulum,
    "ring_gyro": scene_ring_gyro,
    "spiral_ripple": scene_spiral,
}


def build_scene(p):
    return SCENES[p["type"]](p)


# ----------------------------------------------------------------------------- camera

LENS = 50.0
# azimuth deg, elevation deg, how far outside the frame objects may reach (NDC)
CAMERA_PRESETS = {
    "front_iso": (35, 30, 0.88),
    "top_down": (20, 80, 0.88),
    "low_angle": (25, 12, 0.88),
    "close": (40, 26, 1.3),
}


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def project(cam, target, pt, rad, w, h):
    """NDC extent (worst axis) of a sphere (pt, rad) seen by a Blender camera, lens 50, sensor 36 AUTO."""
    f = _unit(_sub(target, cam))
    right = _unit(_cross(f, (0, 0, 1)))
    up = _cross(right, f)
    d = _sub(pt, cam)
    depth = _dot(d, f)
    if depth <= 0.01:
        return 9e9
    k = LENS / 18.0
    ax, ay = (1.0, w / h) if w >= h else (h / w, 1.0)
    x = abs(_dot(d, right)) / depth * k * ax
    y = abs(_dot(d, up)) / depth * k * ay
    r = rad / depth * k * max(ax, ay)
    return max(x, y) + r


def sample_points(sc, n=12):
    pts = []
    for k in range(n):
        t = k / n
        for i, it in enumerate(sc["items"]):
            loc, _, scl = sc["pose"](i, t)
            pts.append((loc, it.get("rad") or 0.5 * math.sqrt(_dot(scl, scl))))
    return pts


def fit_camera(sc, p, w, h):
    az, el, limit = CAMERA_PRESETS[p["camera"]]
    az, el = math.radians(az), math.radians(el)
    target = (0.0, 0.0, sc["target_z"])
    d = (math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el))
    pts = sample_points(sc)

    def cam_at(dist):
        return (target[0] + d[0] * dist, target[1] + d[1] * dist, target[2] + d[2] * dist)

    lo, hi = 1.0, 400.0
    for _ in range(40):
        mid = (lo + hi) / 2
        cam = cam_at(mid)
        if all(project(cam, target, q, r, w, h) <= limit for q, r in pts):
            hi = mid
        else:
            lo = mid
    return cam_at(hi), target


# ----------------------------------------------------------------------------- Blender side


def hex_to_linear(h, mul=1.0):
    h = h.lstrip("#")

    def c(v):
        v = int(v, 16) / 255
        v = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        return min(1.0, v * mul)

    return (c(h[0:2]), c(h[2:4]), c(h[4:6]), 1.0)


def palette_at(pal, x):
    x = max(0.0, min(1.0, x)) * (len(pal) - 1)
    i = min(int(x), len(pal) - 2)
    f = x - i
    a, b = pal[i], pal[i + 1]
    return tuple(a[j] + (b[j] - a[j]) * f for j in range(4))


def _set(node, names, value):
    for n in names:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except Exception:
                pass
    return False


def make_material(name, rgba, style):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    _set(b, ["Base Color"], rgba)
    if style == "glossy":
        _set(b, ["Roughness"], 0.15)
        _set(b, ["Coat Weight", "Clearcoat"], 0.6)
        _set(b, ["Coat Roughness", "Clearcoat Roughness"], 0.05)
    elif style == "matte":
        _set(b, ["Roughness"], 0.65)
    elif style == "metallic":
        _set(b, ["Metallic"], 0.9)
        _set(b, ["Roughness"], 0.28)
    elif style == "neon":
        _set(b, ["Roughness"], 0.4)
        _set(b, ["Emission Color", "Emission"], rgba)
        _set(b, ["Emission Strength"], 2.5)
    elif style == "neutral":
        _set(b, ["Base Color"], (0.55, 0.55, 0.58, 1.0))
        _set(b, ["Metallic"], 0.8)
        _set(b, ["Roughness"], 0.3)
    return m


def make_template(it):
    s = it["shape"]
    if s == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1)
    elif s == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=32, ring_count=16)
    elif s == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, vertices=48)
    else:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=it["major"], minor_radius=it["minor"], major_segments=96, minor_segments=24)
    o = bpy.context.active_object
    for poly in o.data.polygons:
        poly.use_smooth = s != "cube"
    return o


def set_render(scn, w, h, fps, samples):
    scn.render.engine = "CYCLES"
    scn.render.resolution_x, scn.render.resolution_y = w, h
    scn.render.resolution_percentage = 100
    scn.render.fps = fps
    im = scn.render.image_settings
    im.file_format, im.color_mode, im.compression = "PNG", "RGB", 10
    c = scn.cycles
    c.device = "CPU"
    c.samples = samples
    c.use_adaptive_sampling = True
    for attr, val in (("use_denoising", True), ("denoiser", "OPENIMAGEDENOISE"), ("max_bounces", 6),
                      ("diffuse_bounces", 2), ("glossy_bounces", 4), ("transmission_bounces", 2),
                      ("sample_clamp_indirect", 8.0)):
        try:
            setattr(c, attr, val)
        except Exception:
            pass
    for vt in ("Filmic", "Standard"):
        try:
            scn.view_settings.view_transform = vt
            break
        except Exception:
            continue


def main(argv):
    pkg_path, out_dir, w, h, fps, frames, samples = argv[:7]
    w, h, fps, frames, samples = int(w), int(h), int(fps), int(frames), int(samples)
    with open(pkg_path, encoding="utf-8") as f:
        p = clean_scene_params(json.load(f).get("scene", {}))
    sc = build_scene(p)
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    set_render(scn, w, h, fps, samples)

    pal = [hex_to_linear(c) for c in p["palette"]]
    bg = hex_to_linear(p["background"])

    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    bgn = world.node_tree.nodes["Background"]
    bgn.inputs[0].default_value = bg
    bgn.inputs[1].default_value = 0.8

    for name, energy, rot, col in (("Key", 3.5, (55, 0, 35), (1, 1, 1)),
                                   ("Rim", 1.4, (62, 0, -140), pal[1][:3])):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy, ld.angle, ld.color = energy, 0.15, col
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = tuple(math.radians(a) for a in rot)
        scn.collection.objects.link(lo)

    bpy.ops.mesh.primitive_plane_add(size=400)
    floor = bpy.context.active_object
    floor.location = (0, 0, sc["floor_z"])
    fm = make_material("Floor", tuple(min(1.0, c * 1.6) for c in bg[:3]) + (1.0), "glossy")
    floor.data.materials.append(fm)

    nb = 10
    mats = [make_material("m%d" % i, palette_at(pal, i / (nb - 1)), p["material"]) for i in range(nb)]
    neutral = make_material("neutral", (0, 0, 0, 1), "neutral")

    templates, objs = {}, []
    for i, it in enumerate(sc["items"]):
        key = (it["shape"], it.get("major"), it.get("minor"))
        if key not in templates:
            templates[key] = make_template(it)
        o = bpy.data.objects.new("item%d" % i, templates[key].data)
        scn.collection.objects.link(o)
        o.data.materials.clear()
        o.data.materials.append(neutral if it.get("mat") == "neutral" else mats[round(it["cval"] * (nb - 1))])
        if it.get("bevel"):
            bm = o.modifiers.new("bevel", "BEVEL")
            bm.width, bm.segments = it["bevel"], 3
        objs.append(o)
    for tpl in templates.values():
        bpy.data.objects.remove(tpl, do_unlink=True)

    cam_loc, target = fit_camera(sc, p, w, h)
    cd = bpy.data.cameras.new("Cam")
    cd.lens = LENS
    cam = bpy.data.objects.new("Cam", cd)
    scn.collection.objects.link(cam)
    cam.location = cam_loc
    cam.rotation_euler = (Vector(target) - Vector(cam_loc)).to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam

    for f in range(frames):
        t = f / frames
        for i, o in enumerate(objs):
            loc, rot, scl = sc["pose"](i, t)
            o.location, o.rotation_euler, o.scale = loc, rot, scl
        scn.render.filepath = os.path.join(out_dir, "frame_%04d" % (f + 1))
        bpy.ops.render.render(write_still=True)
        print("FRAME %d/%d" % (f + 1, frames), flush=True)


if __name__ == "__main__" and bpy is not None and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1:])

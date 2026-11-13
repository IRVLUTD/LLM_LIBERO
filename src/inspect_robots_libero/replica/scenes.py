"""Import ordered VLA Replica instructions and compile editable MJCF scenes.

Photos supply provenance, not inferred calibration. Every generated layout starts
as an approximation. JSON objects and/or the emitted MJCF can be edited locally.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).with_name("assets")
TASK_NAMES = {1: "bread_on_plate", 2: "bowl_on_coaster", 3: "stack_blocks",
              4: "fold_towel", 5: "open_oven", 6: "clean_whiteboard",
              7: "pepper_shakes", 8: "lift_bowl", 9: "press_button", 10: "collect_blocks"}
PAPER_TASK_NUMBERS = {1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 4}
COLORS = {"red": [.8, .04, .08, 1], "blue": [.04, .25, .8, 1],
          "green": [.18, .6, .12, 1], "yellow": [.95, .8, .08, 1],
          "orange": [1, .35, .02, 1], "purple": [.5, .12, .65, 1],
          "pink": [.95, .15, .5, 1], "white": [.92, .92, .92, 1],
          "brown": [.65, .35, .12, 1], "black": [.1, .1, .1, 1]}
HOME = [0, -1.57, 1.57, 1.57, -1.57, .5]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scene(path):
    path = Path(path).resolve()
    scene = json.loads(path.read_text())
    if scene.get("schema_version") != 1:
        raise ValueError(f"Unsupported scene schema: {path}")
    return scene


def _object(name, kind, color, pos, size, **kwargs):
    return dict(name=name, kind=kind, color=color, pos=pos, size=size, **kwargs)


def default_scene(task_id, subset, variation, instruction):
    """Deterministic starting layouts. Coordinates are explicit, never photo-calibrated."""
    if task_id not in TASK_NAMES or subset not in ("ID", "OOD", "AB"):
        raise ValueError("Unknown task or subset")
    colors = re.findall(r"\b(" + "|".join(COLORS) + r")\b", instruction.lower())
    number_words = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9)
    count_match = re.search(r"\b([1-9][0-9]*|" + "|".join(number_words) + r")\b", instruction.lower())
    token = count_match[1] if count_match else "1"
    count = number_words[token] if token in number_words else int(token)
    offset = (variation - 3) * .008
    src = [.23 + offset, -.065, .025]
    dst = [.26 - offset, .075, .008]
    objects = []
    score = dict(kind=TASK_NAMES[task_id], dwell_seconds=.5, count=count,
                 xy_tolerance_m=.02, linear_velocity_m_s=.025, angular_velocity_rad_s=.4)
    fidelity = "primitive_geometry_and_manual_layout"
    if task_id in (1, 2, 3):
        source_kind = {1: "bread", 2: "bowl", 3: "block"}[task_id]
        target_kind = {1: "plate", 2: "coaster", 3: "block"}[task_id]
        c1 = "brown" if task_id == 1 else colors[0]
        c2 = colors[-1]
        size = {1: [.028, .021, .016], 2: [.035, .028], 3: [.018, .018, .018]}[task_id]
        src[2] = {1: .017, 2: .0025, 3: .0185}[task_id]
        dst[2] = .0185 if task_id == 3 else .004
        objects += [_object("source", source_kind, c1, src, size),
                    _object("target", target_kind, c2, dst,
                            [.018]*3 if task_id == 3 else [.055 if task_id == 1 else .042, .004])]
        score.update(source="source", target="target")
    elif task_id == 4:
        objects += [_object("towel", "towel", colors[0], [.26 + offset, -.05, .004], [.04, .07, .002], segments=4)]
        score.update(source="towel", fold_gap_m=.035)
        fidelity = "articulated_strip_towel_proxy"
    elif task_id == 5:
        objects += [_object("oven", "oven", "black", [.32 + offset, .04, .002], [.065, .055, .065])]
        score.update(joint="oven_hinge", open_angle_rad=1.0)
    elif task_id == 6:
        objects += [_object("board", "board", "white", [.29 + offset, .02, .009], [.065, .06, .008]),
                    _object("eraser", "block", "black", [.19, -.08, .013], [.025, .012, .012])]
        score.update(source="eraser", target="board", coverage_fraction=.9, wipe_radius_m=.024)
        fidelity = "contact_coverage_whiteboard_proxy"
    elif task_id == 7:
        objects += [_object("pepper", "bottle", "white", [.21 + offset, -.07, .035], [.014, .035]),
                    _object("target", "plate", colors[-1], dst, [.06, .004])]
        score.update(source="pepper", target="target", inverted_axis_z=-.2, reset_axis_z=.5)
        fidelity = "tilt_cycles_over_plate_proxy_no_grains"
    elif task_id == 8:
        src[2] = .0025
        objects += [_object("source", "bowl", colors[0], src, [.035, .028])]
        score.update(source="source", lift_height_m=.05, return_height_m=.015)
    elif task_id == 9:
        objects += [_object("button", "button", "red", [.25 + offset, 0, .012], [.027, .012])]
        score.update(joint="button_slide", press_depth_m=.007, release_depth_m=.002)
    elif task_id == 10:
        objects += [_object("target", "box", colors[-1], [.29 - offset, .075, .004], [.055, .055, .032])]
        for i in range(count):
            objects += [_object(f"block{i}", "block", ["red", "blue", "green", "yellow", "orange"][i % 5],
                                [.17 + (i % 3)*.043, -.075 - (i // 3)*.045, .0155], [.015]*3)]
        score.update(target="target", sources=[f"block{i}" for i in range(count)])
    # Difficulty varies in the paper via distractors as well as positions. These
    # provisional distractors are editable, not reconstructions of photo contents.
    if variation >= 4 and task_id in (1, 2, 3, 4):
        for i in range(variation - 3):
            objects += [_object(f"distractor{i}", "block", ["orange", "purple"][i],
                                [.18 + .05*i, .14, .016] if task_id == 4 else [.34, -.09 + .05*i, .016], [.015]*3)]
    return dict(schema_version=1, id=f"{subset}_task{task_id:02d}_v{variation:02d}",
                task_id=task_id, paper_task_number=PAPER_TASK_NUMBERS[task_id],
                task_name=TASK_NAMES[task_id], subset=subset, variation=variation,
                instruction=instruction, seed=142 + variation, fidelity=fidelity,
                calibration=dict(status="approximate", notes="Manual provisional layout; no photo-to-world calibration performed."),
                objects=objects, scoring=score,
                cameras={"top_camera": dict(pos=[.46, 0, .46], xyaxes=[0, 1, 0, -.85, 0, .52], fovy=58)},
                robot=dict(base_pos=[0, 0, 0], base_euler=[0, 0, math.pi/2], home=HOME))


def prepare_scenes(tasks_root, output):
    """Read the external repository; write a new local catalog. Never overwrite edits."""
    tasks_root, output = Path(tasks_root).resolve(), Path(output).resolve()
    paths = sorted(tasks_root.glob("*/task_variants/task[0-9][0-9].json"))
    paths = [p for p in paths if p.parts[-3] in ("ID", "OOD", "AB")]
    if not paths:
        raise ValueError(f"No task variants found under {tasks_root}")
    # Validate every input before creating an output directory.
    scenes = []
    for path in paths:
        task_id, subset = int(path.stem[4:]), path.parts[-3]
        variants = json.loads(path.read_text())
        variants = variants.get("tasks") if isinstance(variants, dict) else variants
        if not isinstance(variants, list) or len(variants) != 5:
            raise ValueError(f"Expected five ordered variants: {path}")
        for variation, item in enumerate(variants, 1):
            instruction = item.get("name", item.get("task")) if isinstance(item, dict) else item
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Invalid variant {variation}: {path}")
            scene = default_scene(task_id, subset, variation, instruction)
            refs = tasks_root/subset/"referencePics"/path.stem
            reference = next((refs/f"pic{variation}.{ext}" for ext in ("jpg", "png", "jpeg")
                              if (refs/f"pic{variation}.{ext}").is_file()), None)
            scene["provenance"] = dict(instructions_file=str(path), instructions_sha256=sha256(path),
                                       reference_image=str(reference) if reference else None,
                                       reference_sha256=sha256(reference) if reference else None)
            scenes.append(scene)
    output.mkdir(parents=True, exist_ok=False)
    for scene in scenes:
        folder = output/scene["id"]
        folder.mkdir()
        config = folder/"scene.json"
        config.write_text(json.dumps(scene, indent=2) + "\n")
        compile_scene(config)
    catalog = dict(schema_version=1, source_tasks_root=str(tasks_root),
                   scenes=[dict(id=s["id"], config=f'{s["id"]}/scene.json',
                                subset=s["subset"], task_id=s["task_id"], variation=s["variation"])
                           for s in scenes])
    (output/"catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


def _text(values):
    return " ".join(str(v) for v in values)


def compile_scene(config_path):
    """Compile JSON to sibling scene.xml; raw edits are preserved until explicitly rebuilt.

    custom_mjcf replaces a generated object's body with an editable MJCF body file.
    Keep the same body/joint names used by scoring, or update scoring as well.
    """
    config_path = Path(config_path).resolve()
    scene = read_scene(config_path)
    root = ET.parse(ASSETS/"robot.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSETS/"meshes"))
    for mesh in scene.get("meshes", []):
        path = Path(mesh["file"])
        if not path.is_absolute():
            path = config_path.parent/path
        ET.SubElement(root.find("asset"), "mesh", name=mesh["name"], file=str(path.resolve()),
                      scale=_text(mesh.get("scale", [1, 1, 1])))
    world = root.find("worldbody")
    base = world.find("body[@name='Base']")
    base.set("pos", _text(scene["robot"]["base_pos"]))
    base.set("euler", _text(scene["robot"]["base_euler"]))
    for name, camera in scene.get("cameras", {}).items():
        element = root.find(f".//camera[@name='{name}']")
        if element is None:
            raise ValueError(f"Unknown camera {name}")
        for key, value in camera.items():
            element.set(key, _text(value) if isinstance(value, list) else str(value))
    for obj in scene["objects"]:
        if obj.get("custom_mjcf"):
            custom = Path(obj["custom_mjcf"])
            if not custom.is_absolute():
                custom = config_path.parent/custom
            body = ET.parse(custom).getroot()
            if body.tag != "body" or body.get("name") != obj["name"]:
                raise ValueError("custom_mjcf must contain one body with the object's name")
            world.append(body)
        else:
            _add_object(world, obj)
    # Model defaults establish object free-joint poses and articulated zeroes.
    # Seed only robot coordinates; do not construct fragile full-length keyframes.
    import mujoco
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = model.qpos0.copy()
    home = scene["robot"]["home"]
    if len(home) != 6:
        raise ValueError("robot.home must contain six joint positions")
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    for name, value in zip(joints, home):
        qpos[model.joint(name).qposadr[0]] = value
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=_text(qpos), ctrl=_text(home))
    ET.indent(root)
    target = config_path.with_suffix(".xml")
    ET.ElementTree(root).write(target, encoding="unicode")
    return target


def _add_object(world, obj):
    name, kind = obj["name"], obj["kind"]
    body = ET.SubElement(world, "body", name=name, pos=_text(obj["pos"]))
    if "quat" in obj:
        body.set("quat", _text(obj["quat"]))
    elif "euler" in obj:
        body.set("euler", _text(obj["euler"]))
    size = obj["size"]
    color = obj.get("rgba", COLORS[obj["color"]])
    mass = obj.get("mass", .045)

    def geom(parent=body, **kwargs):
        values = {"rgba": _text(color), "friction": "1.2 .01 .001", **kwargs}
        return ET.SubElement(parent, "geom", **{k: str(v) for k, v in values.items()})

    fixed = obj.get("fixed", kind in ("plate", "coaster", "box", "board", "oven", "button"))
    if not fixed:
        ET.SubElement(body, "freejoint", name=name+"_free")
    if kind == "mesh":
        geom(type="mesh", mesh=obj["mesh"], mass=mass)
    elif kind in ("block", "bread", "board"):
        geom(type="ellipsoid" if kind == "bread" else "box", size=_text(size), mass=mass)
        if kind == "board":
            for i in range(4):
                for j in range(4):
                    geom(name=f"ink_{i}_{j}", type="box", size=".007 .007 .0003",
                         pos=_text([(i-1.5)*.028, (j-1.5)*.026, size[2]+.0004]),
                         rgba=".05 .05 .05 1", contype="0", conaffinity="0", mass="0")
    elif kind in ("plate", "coaster", "bottle"):
        geom(type="cylinder", size=_text(size), mass=mass)
        if kind == "bottle":
            geom(type="cylinder", size=_text([size[0]*1.05, .006]), pos=_text([0, 0, size[1]]),
                 rgba=".08 .08 .08 1", mass=.005)
    elif kind == "bowl":
        radius, height = size
        geom(type="cylinder", size=_text([radius*.62, .002]), mass=mass*.4)
        for i in range(20):
            angle = 2*math.pi*i/20
            geom(type="capsule", fromto=_text([radius*math.cos(angle), radius*math.sin(angle), height*.45,
                                              radius*math.cos(angle+.32), radius*math.sin(angle+.32), height*.45]),
                 size=".0035", mass=mass*.6/20)
            geom(type="capsule", fromto=_text([radius*.6*math.cos(angle), radius*.6*math.sin(angle), .002,
                                              radius*math.cos(angle), radius*math.sin(angle), height*.45]),
                 size=".002", mass=.0005)
    elif kind == "box":
        x, y, h = size
        geom(type="box", size=_text([x, y, .004]))
        for sign in (-1, 1):
            geom(type="box", size=_text([.004, y, h]), pos=_text([sign*x, 0, h]))
            geom(type="box", size=_text([x, .004, h]), pos=_text([0, sign*y, h]))
    elif kind == "oven":
        x, y, z = size
        geom(type="box", size=_text([x, .004, z]), pos=_text([0, y, z]))
        geom(type="box", size=_text([x, y, .004]), pos=_text([0, 0, 2*z]))
        for sign in (-1, 1):
            geom(type="box", size=_text([.004, y, z]), pos=_text([sign*x, 0, z]))
        door = ET.SubElement(body, "body", name="oven_door", pos=_text([0, -y, 0]))
        ET.SubElement(door, "joint", name="oven_hinge", axis="1 0 0", range="0 1.8", damping=".15", frictionloss=".08")
        geom(door, type="box", size=_text([x, .004, z]), pos=_text([0, 0, z]), mass=.08)
        geom(door, type="capsule", fromto=_text([-x*.7, -.018, z*1.7, x*.7, -.018, z*1.7]), size=".006", mass=.01)
    elif kind == "button":
        geom(type="cylinder", size=_text(size), rgba=".12 .12 .12 1")
        cap = ET.SubElement(body, "body", name="button_cap", pos=_text([0, 0, size[1]+.008]))
        ET.SubElement(cap, "joint", name="button_slide", type="slide", axis="0 0 1", range="-.012 0", stiffness="65", damping=".7", frictionloss="0", armature=".001")
        geom(cap, type="cylinder", size=_text([size[0]*.75, .007]), mass=.02)
    elif kind == "towel":
        # A physical articulated strip, not a rigid rectangle labeled 'cloth'.
        x, y, thickness = size
        geom(type="box", size=_text([x/2, y, thickness]), mass=.012)
        current = body
        for i in range(1, obj.get("segments", 4)):
            segment = ET.SubElement(current, "body", name=f"towel_segment{i}", pos=_text([x/2 if i == 1 else x, 0, 0]))
            ET.SubElement(segment, "joint", name=f"towel_fold{i}", axis="0 1 0", range="-3.14 3.14", damping=".0002", frictionloss=".00001", armature=".00001")
            geom(segment, type="box", size=_text([x/2, y, thickness]), pos=_text([x/2, 0, 0]), mass=.012)
            current = segment
    else:
        raise ValueError(f"Unknown geometry kind: {kind}")

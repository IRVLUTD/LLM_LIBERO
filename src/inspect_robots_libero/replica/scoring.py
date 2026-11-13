"""Private, stateful physical/proxy scorers. Never expose these to the policy.

Counted tasks use hysteresis and require complete cycles. Current success can
be lost by overshooting; it is not an irreversible 'ever succeeded' flag.
"""
from __future__ import annotations

import numpy as np


class ReplicaScorer:
    def __init__(self, sim, scene):
        self.sim, self.scene, self.spec = sim, scene, scene["scoring"]
        self.count, self.active, self.dwell = 0, False, 0.
        self.success = False
        self.wiped = set()
        self.interacted = False
        self.initial_z = {obj["name"]: float(sim.data.body(obj["name"]).xpos[2])
                          for obj in scene["objects"]}

    def touching(self, first, second):
        """Include descendants (oven door and button cap) in contact membership."""
        model, data = self.sim.model, self.sim.data

        def descendants(name):
            root = model.body(name).id
            found = {root}
            for i in range(root + 1, model.nbody):
                if int(model.body_parentid[i]) in found:
                    found.add(i)
            return found

        left, right = descendants(first), descendants(second)
        for c in data.contact:
            if c.dist > .001:
                continue
            a, b = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if (a in left and b in right) or (a in right and b in left):
                return True
        return False

    def released(self, source):
        return self.sim.gripper_command < .5 and not self.touching(source, "Base")

    def still(self, source):
        # cvel is world-oriented rotational then translational body velocity.
        v = self.sim.data.cvel[self.sim.model.body(source).id]
        return (np.linalg.norm(v[3:]) < self.spec["linear_velocity_m_s"]
                and np.linalg.norm(v[:3]) < self.spec["angular_velocity_rad_s"])

    def supported(self, source, target):
        data = self.sim.data
        src, dst = data.body(source), data.body(target)
        return (self.touching(source, target) and self.released(source)
                and self.still(source) and src.xmat.reshape(3, 3)[2, 2] > .94
                and np.linalg.norm(src.xpos[:2]-dst.xpos[:2]) < self.spec["xy_tolerance_m"]
                and src.xpos[2] > dst.xpos[2])

    def update(self):
        s, data = self.spec, self.sim.data
        kind = s["kind"]
        valid = False
        interaction_body = s.get("source", "oven_door" if kind == "open_oven" else s.get("target"))
        if interaction_body:
            self.interacted |= self.touching(interaction_body, "Base")
        if kind in ("bread_on_plate", "bowl_on_coaster", "stack_blocks"):
            valid = self.supported(s["source"], s["target"])
        elif kind == "collect_blocks":
            target = data.body(s["target"])
            geometry = next(o for o in self.scene["objects"] if o["name"] == s["target"])
            half_x, half_y, half_z = geometry["size"]
            # Transform into the box frame, rather than assuming axis-aligned boxes.
            local = [target.xmat.reshape(3, 3).T @ (data.body(name).xpos-target.xpos)
                     for name in s["sources"]]
            inside = [abs(p[0]) < half_x-.015 and abs(p[1]) < half_y-.015
                      and 0 < p[2] < 2*half_z for p in local]
            valid = sum(inside) == s["count"] and all(
                self.touching(name, s["target"]) and self.released(name) and self.still(name)
                for name in s["sources"])
        elif kind == "open_oven":
            valid = self.interacted and float(data.joint(s["joint"]).qpos[0]) >= s["open_angle_rad"]
        elif kind == "lift_bowl":
            name = s["source"]
            height = data.body(name).xpos[2] - self.initial_z[name]
            if not self.active and height >= s["lift_height_m"] and self.touching(name, "Base"):
                self.active = True
            if self.active and height <= s["return_height_m"]:
                self.count += 1
                self.active = False
            valid = self.count == s["count"] and not self.active and height <= s["return_height_m"] and self.still(name)
        elif kind == "press_button":
            depth = -float(data.joint(s["joint"]).qpos[0])
            if not self.active and depth >= s["press_depth_m"] and self.touching("button", "Base"):
                self.active = True
            if self.active and depth <= s["release_depth_m"]:
                self.count += 1
                self.active = False
            valid = self.count == s["count"] and not self.active
        elif kind == "pepper_shakes":
            source, target = data.body(s["source"]), data.body(s["target"])
            axis_z = source.xmat.reshape(3, 3)[2, 2]
            target_geometry = next(o for o in self.scene["objects"] if o["name"] == s["target"])
            over = (np.linalg.norm(source.xpos[:2]-target.xpos[:2]) < target_geometry["size"][0]
                    and .03 < source.xpos[2]-target.xpos[2] < .22)
            if not self.active and axis_z < s["inverted_axis_z"] and over and self.touching(s["source"], "Base"):
                self.active = True
            if self.active and axis_z > s["reset_axis_z"]:
                self.count += 1
                self.active = False
            valid = self.count == s["count"] and not self.active
        elif kind == "clean_whiteboard":
            if self.touching(s["source"], s["target"]):
                eraser = data.body(s["source"]).xpos
                for i in range(4):
                    for j in range(4):
                        name = f"ink_{i}_{j}"
                        if np.linalg.norm(eraser-data.geom(name).xpos) < s["wipe_radius_m"]:
                            self.wiped.add(name)
                            self.sim.model.geom_rgba[self.sim.model.geom(name).id] = [.92, .92, .92, 1]
            valid = len(self.wiped)/16 >= s["coverage_fraction"] and self.released(s["source"])
        elif kind == "fold_towel":
            n = next(o for o in self.scene["objects"] if o["name"] == s["source"])["segments"]
            first, last = data.body("towel"), data.body(f"towel_segment{n-1}")
            valid = (np.linalg.norm(first.xpos-last.xpos) < s["fold_gap_m"]
                     and self.interacted and self.released("towel") and self.still("towel"))
        else:
            raise ValueError(f"No scorer for {kind}")
        self.dwell = self.dwell + self.sim.control_dt if valid else 0.
        self.success = self.dwell + 1e-9 >= s["dwell_seconds"]
        return self.success

    def diagnostics(self):
        return dict(success=self.success, event_count=self.count, cycle_active=self.active,
                    interacted=self.interacted,
                    dwell_seconds=self.dwell, wiped_fraction=len(self.wiped)/16,
                    scorer_kind=self.spec["kind"], fidelity=self.scene["fidelity"])

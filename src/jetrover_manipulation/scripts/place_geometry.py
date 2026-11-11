"""Nominal fixed-base USD collision-vertex floor guard (metres/radians).

Geometry comes from the project's USD export, not a TCP point approximation.
This does not model dynamic base motion, deformation, or other obstacles.
"""

import json
import math
from pathlib import Path

import numpy as np

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "r_joint")


def transform(position, quaternion):
    x, y, z, w = np.asarray(quaternion, float) / np.linalg.norm(quaternion)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = position
    return result


class FloorViolation(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__(
            f"Floor guard: {details['body']} clearance="
            f"{details['clearance_m']*1000:.2f}mm < "
            f"{details['minimum_m']*1000:.2f}mm ({details['segment']})"
        )


class FloorGuard:
    def __init__(self, geometry=None, *, minimum=0.005, floor_z=0.0, base_z_offset=0.0):
        if not np.isfinite([minimum, floor_z, base_z_offset]).all() or minimum <= 0:
            raise ValueError(
                "Floor parameters must be finite; minimum must be positive"
            )
        path = (
            Path(geometry)
            if geometry
            else Path(__file__).with_name("floor_geometry.json")
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.minimum, self.floor_z = minimum, floor_z
        self.base = np.array(
            data["bodies"]["base_link"]["world_zero_matrix"], dtype=float
        )
        self.base[2, 3] += base_z_offset
        self.points = {
            name: np.asarray(body["collision_vertices"], dtype=float)
            for name, body in data["bodies"].items()
            if name != "base_link"
        }
        self.frames = [
            (
                j,
                transform(j["pos0"], j["rot0"]),
                np.linalg.inv(transform(j["pos1"], j["rot1"])),
            )
            for j in data["joints"]
        ]
        if not all(name in {j["name"] for j, _, _ in self.frames} for name in JOINTS):
            raise ValueError("Incomplete geometry: six commanded joints required")
        self.source = data["source_name"]

    def clearance(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint positions")
        values = dict(zip(JOINTS, q))
        poses, pending = {"base_link": self.base}, list(self.frames)
        minimum, body = float("inf"), None
        while pending:
            count = len(pending)
            for item in pending[:]:
                j, a, b = item
                if j["parent"] not in poses:
                    continue
                rotation = np.eye(4)
                if j["axis"]:
                    if j["name"] in values:
                        angle = values[j["name"]]
                    elif "gripper_multiplier" in j:
                        angle = q[5] * j["gripper_multiplier"]
                    else:
                        raise ValueError(f"Unknown moving joint: {j['name']}")
                    c, s = math.cos(angle), math.sin(angle)
                    axis = j["axis"]
                    rotation[:3, :3] = {
                        "X": [[1, 0, 0], [0, c, -s], [0, s, c]],
                        "Y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
                        "Z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
                    }[axis]
                pose = poses[j["parent"]] @ a @ rotation @ b
                poses[j["child"]] = pose
                points = self.points.get(j["child"])
                if points is not None and points.size:
                    clearance = float(
                        np.min(points @ pose[2, :3] + pose[2, 3]) - self.floor_z
                    )
                    if clearance < minimum:
                        minimum, body = clearance, j["child"]
                pending.remove(item)
            if len(pending) == count:
                raise ValueError("Disconnected geometry joint tree")
        if body is None:
            raise ValueError("No arm collision vertices")
        return minimum, body

    def check_motion(self, actual, previous, candidate):
        """Check endpoints and sampled joint paths from actual AND previous target.

        Samples are at most .01 rad apart in every joint. Not continuous CCD.
        """
        lowest = float("inf")
        for name, start in [
            ("actual_to_target", actual),
            ("previous_to_target", previous),
        ]:
            start, end = np.asarray(start), np.asarray(candidate)
            steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / 0.01)))
            for fraction in np.linspace(0.0, 1.0, steps + 1):
                clearance, body = self.clearance(start + (end - start) * fraction)
                lowest = min(lowest, clearance)
                if clearance < self.minimum:
                    raise FloorViolation(
                        dict(
                            body=body,
                            clearance_m=clearance,
                            minimum_m=self.minimum,
                            segment=name,
                            fraction=float(fraction),
                        )
                    )
        return lowest

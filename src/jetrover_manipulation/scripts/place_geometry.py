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

    def _evaluate(self, q, *, gripper_pose=False):
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
        return (
            minimum,
            body,
            poses["gripper_link"] if gripper_pose else poses["gripper_link"][:3, 3],
        )

    def gripper_pose(self, q):
        return self._evaluate(q, gripper_pose=True)[2]

    def clearance(self, q):
        minimum, body, _ = self._evaluate(q)
        return minimum, body

    def features(self, q):
        clearance, _, wrist = self._evaluate(q)
        radial = np.linalg.norm(wrist[:2] - self.base[:2, 3])
        return np.array([clearance, wrist[2], np.sum(q[1:4]), radial])

    def correct_motion(
        self, actual, previous, candidate, max_step, *, max_wrist_z, min_pitch
    ):
        """Project shoulder/elbow/wrist targets inside floor/height/pitch bounds.

        Keep yaw, wrist roll and gripper unchanged when projection succeeds.
        Respect the existing per-tick speed box. Preserve tool radial position
        and pitch where possible. Fall back to a verified partial step/hold
        when a full correction is infeasible; never silently send unsafe q.
        """
        actual, previous, candidate = [
            np.asarray(q, float) for q in (actual, previous, candidate)
        ]
        limits = np.asarray(max_step, float)
        if limits.shape != (6,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("Expected six positive speed steps")
        if not np.isfinite([max_wrist_z, min_pitch]).all():
            raise ValueError("Workspace limits must be finite")
        original = candidate.copy()
        initial = self.features(actual)
        previous_features = self.features(previous)
        # A small measured tracking overshoot into the soft margin may recover
        # without another abort, but an actual nominal floor penetration stops.
        if initial[0] <= 0:
            raise FloorViolation(
                dict(
                    body=self.clearance(actual)[1],
                    clearance_m=initial[0],
                    minimum_m=self.minimum,
                    segment="actual_below_floor",
                )
            )

        def check(q):
            if np.any(np.abs(q - previous) > limits + 1e-9):
                return None
            lowest = float("inf")
            for start, metrics in ((actual, initial), (previous, previous_features)):
                steps = max(1, int(np.ceil(np.max(np.abs(q - start)) / 0.01)))
                # Never worsen an existing soft-bound violation while recovering.
                floor_bound = min(self.minimum, metrics[0])
                upper_z = max(max_wrist_z, metrics[1])
                lower_pitch = min(min_pitch, metrics[2])
                for fraction in np.linspace(0.0, 1.0, steps + 1):
                    m = self.features(start + (q - start) * fraction)
                    lowest = min(lowest, m[0])
                    if (
                        m[0] < floor_bound - 1e-9
                        or m[1] > upper_z + 1e-9
                        or m[2] < lower_pitch - 1e-9
                    ):
                        return None
            endpoint = self.features(q)
            if initial[0] < self.minimum and endpoint[0] < initial[0] + 1e-6:
                return None
            if initial[1] > max_wrist_z and endpoint[1] > initial[1] - 1e-6:
                return None
            if initial[2] < min_pitch and endpoint[2] < initial[2] + 1e-6:
                return None
            return lowest

        # A 2 mm buffer avoids chattering on the 5 mm soft boundary.
        desired_floor = self.minimum + 0.002
        desired_z = max_wrist_z - 0.002
        desired_pitch = min_pitch + 0.01
        reference = self.features(original)
        q = original.copy()
        lower, upper = previous - limits, previous + limits
        lower[1:4] = np.maximum(lower[1:4], -2.08)
        upper[1:4] = np.minimum(upper[1:4], 2.08)
        for iteration in range(13):
            m = self.features(q)
            result = check(q)
            if result is not None:
                return q, dict(
                    mode="unchanged" if iteration == 0 else "projected",
                    minimum_path_clearance_m=result,
                    correction_rad=(q - original).tolist(),
                    wrist_z_m=float(m[1]),
                    pitch_rad=float(m[2]),
                )
            jacobian = np.empty((4, 3))
            for k, joint in enumerate((1, 2, 3)):
                plus, minus = q.copy(), q.copy()
                plus[joint] += 1e-4
                minus[joint] -= 1e-4
                jacobian[:, k] = (self.features(plus) - self.features(minus)) / (2e-4)
            rows, residuals = [], []
            if m[0] < desired_floor:
                rows.append(jacobian[0])
                residuals.append(desired_floor - m[0])
            if m[1] > desired_z:
                rows.append(jacobian[1])
                residuals.append(desired_z - m[1])
            pitch_target = max(reference[2], desired_pitch)
            rows.append(jacobian[2] * 0.05)
            residuals.append((pitch_target - m[2]) * 0.05)
            rows.append(jacobian[3])
            residuals.append(reference[3] - m[3])
            matrix = np.asarray(rows)
            delta = np.linalg.solve(
                matrix.T @ matrix + np.eye(3) * 1e-7, matrix.T @ np.asarray(residuals)
            )
            q[1:4] = np.clip(
                q[1:4] + np.clip(delta, -0.08, 0.08), lower[1:4], upper[1:4]
            )

        # If constrained IK cannot fit this tick, continue with a verified
        # fraction of the original move, rather than terminating the trial.
        for fraction in (0.75, 0.5, 0.25, 0.0):
            partial = previous + (original - previous) * fraction
            result = check(partial)
            if result is not None:
                m = self.features(partial)
                return partial, dict(
                    mode="partial" if fraction else "hold",
                    minimum_path_clearance_m=result,
                    correction_rad=(partial - original).tolist(),
                    wrist_z_m=float(m[1]),
                    pitch_rad=float(m[2]),
                )
        clearance, body = self.clearance(actual)
        raise FloorViolation(
            dict(
                body=body,
                clearance_m=clearance,
                minimum_m=self.minimum,
                segment="no_safe_rate_limited_correction",
            )
        )

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

"""Baseline50 Hermite return reversed for lowering; no ROS imports."""

import numpy as np


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (10 + u * (-15 + 6 * u))


class Path:
    def __init__(self, knots):
        self.points = np.asarray(knots, dtype=float)
        if self.points.shape != (7, 5) or not np.isfinite(self.points).all():
            raise ValueError("Expected seven finite calibrated five-joint poses")
        self.slopes = np.zeros_like(self.points)
        # Derivatives with respect to normalized segment coordinate.
        for i in range(1, 6):
            for j in range(5):
                a = self.points[i, j] - self.points[i - 1, j]
                b = self.points[i + 1, j] - self.points[i, j]
                self.slopes[i, j] = 2 * a * b / (a + b) if a * b > 0 else 0.0

    def sample(self, u):
        t = min(1.0, max(0.0, u)) * 6
        i = min(5, int(t))
        s = t - i
        a, b = self.points[i : i + 2]
        va, vb = self.slopes[i : i + 2]
        return (
            (2 * s**3 - 3 * s * s + 1) * a
            + (s**3 - 2 * s * s + s) * va
            + (-2 * s**3 + 3 * s * s) * b
            + (s**3 - s * s) * vb
        )


def segments(config, actual):
    path = Path(config["return_knots_rad"])
    actual = np.asarray(actual, dtype=float)
    opened = float(config["open_rad"])
    held = float(actual[5])
    if not 0 <= held <= 1.5 or not 0 <= opened <= 1.5:
        raise ValueError("Gripper out of range")

    def pose(q, g):
        return np.r_[q, g]

    home = pose(path.sample(1.0), held)
    low = pose(path.sample(0.0), held)
    # Quintic ease has maximum derivative 1.875. Allow a slow alignment
    # from the measured policy end pose instead of assuming exact HOME.
    speed = float(config.get("align_max_speed_rad_s", 0.25))
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Invalid carry alignment speed")
    align_time = max(
        float(config["align_seconds"]),
        1.875 * float(np.max(np.abs(home[:5] - actual[:5]))) / speed,
    )
    if config.get("carry_only", False):
        return [
            (
                "align carry at HOME",
                align_time,
                lambda u: actual + (home - actual) * ease(u),
            )
        ]
    if config.get("reset_only", False):
        return [
            (
                "align carry at HOME",
                align_time,
                lambda u: actual + (home - actual) * ease(u),
            ),
            (
                "open empty gripper at HOME",
                config["open_seconds"],
                lambda u: pose(home[:5], held + (opened - held) * ease(u)),
            ),
        ]
    return [
        (
            "align carry at HOME",
            align_time,
            lambda u: actual + (home - actual) * ease(u),
        ),
        (
            "lower at HOME release position",
            config["lower_seconds"],
            lambda u: pose(path.sample(1 - u), held),
        ),
        (
            "open gripper to release",
            config["open_seconds"],
            lambda u: pose(low[:5], held + (opened - held) * ease(u)),
        ),
        (
            "release pause",
            config["release_pause_seconds"],
            lambda u: pose(low[:5], opened),
        ),
        (
            "return empty arm HOME",
            config["return_seconds"],
            lambda u: pose(path.sample(u), opened),
        ),
    ]

"""Rate-limit approach velocity using simulation time."""

import math


def stopping_speed(remaining, maximum, acceleration):
    """Start braking before target X; reserve margin for observation delay."""
    return min(maximum, 0.7 * math.sqrt(2 * acceleration * max(0.0, remaining)))


def ramp_velocity(current, target, dt, linear_accel=0.2, angular_accel=3.0):
    if not all(
        math.isfinite(v) for v in (*current, *target, dt, linear_accel, angular_accel)
    ):
        raise ValueError("Velocity/rate must be finite")
    if dt < 0 or linear_accel <= 0 or angular_accel <= 0:
        raise ValueError("Invalid time or acceleration")
    dt = min(dt, 0.05)
    return tuple(
        c + max(-a * dt, min(a * dt, t - c))
        for c, t, a in zip(current, target, (linear_accel, angular_accel))
    )

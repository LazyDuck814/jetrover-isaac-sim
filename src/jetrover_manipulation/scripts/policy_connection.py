"""HTTP startup handshake only; does not publish ROS commands."""

import json
from pathlib import Path
import time
import urllib.request

MODEL_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]


def validate_health(info, checkpoint=None):
    if (
        info.get("model") != "smolvla"
        or info.get("joint_names") != MODEL_JOINTS
        or info.get("action_shape") != [50, 6]
        or info.get("fps") != 30
        or info.get("action_semantics") != "absolute_joint_position_target_rad"
    ):
        raise RuntimeError(f"Incompatible policy server: {info}")
    if (
        checkpoint
        and Path(info.get("checkpoint", "")).resolve() != Path(checkpoint).resolve()
    ):
        raise RuntimeError(
            f'Wrong checkpoint: server={info.get("checkpoint")}, expected={checkpoint}'
        )
    return info


def wait_for_policy(host, port, timeout, checkpoint=None):
    deadline = time.monotonic() + timeout
    print(f"[SERVER WAIT] http://{host}:{port} | timeout={timeout:g}s", flush=True)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "Policy server readiness timed out; no arm commands sent"
            )
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/health", timeout=min(2.0, remaining)
            ) as response:
                info = json.load(response)
        except OSError:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
            continue
        return validate_health(info, checkpoint)

#!/usr/bin/env python3
"""SmolVLA ROS Jazzy client. Default: one inference, NO arm commands.

--execute enables one bounded trial from HOME. Run with simulation playing.
This is an experimental controller, not a collision avoidance system.
"""
import argparse
import base64
from datetime import datetime, timezone
from functools import partial
import io
import json
from pathlib import Path
import threading
import time
import urllib.request
import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from rosgraph_msgs.msg import Clock
from policy_connection import wait_for_policy

INSTRUCTION = "Pick up the red cube, return to the initial arm pose, and hold it."

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
# Conservative target-rate limits inherited from the existing live controller.
MAX_TARGET_SPEED = np.array([0.5, 1.3, 0.9, 0.7, 0.7, 1.5])


class ObservationPending(RuntimeError):
    """A transient input gap; reset/nonfinite-state errors must not be retried."""


def wait_for_arm_subscriber(publisher, timeout=10.0):
    deadline = time.monotonic() + timeout
    announced = False
    while publisher.get_subscription_count() == 0:
        if not announced:
            print(
                "[ARM WAIT] Waiting for /arm_joint_commands subscriber (up to 10s).",
                flush=True,
            )
            announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "No arm command subscriber after waiting 10 seconds; check Isaac arm command graph"
            )
        time.sleep(0.05)
    print("[ARM READY] /arm_joint_commands subscriber connected.", flush=True)


def limit_target(target, previous, dt, *, gripper_max=1.5):
    bounded = np.clip(target, [-2.08] * 5 + [0.0], [2.08] * 5 + [gripper_max])
    rates = MAX_TARGET_SPEED * min(dt, 0.05)
    return previous + np.clip(bounded - previous, -rates, rates)


class Bridge(Node):
    def __init__(self, args):
        super().__init__("smolvla_live_grasp")
        self.environment = getattr(args, "environment", "isaac")
        self.ros_joint_names = list(args.joint_names)
        self.model_names = dict(zip(self.ros_joint_names, JOINTS))
        self.lock = threading.Lock()
        self.rgb = None
        self.positions = {}
        self.received = {}
        self.clock_ns = None
        self.clock_wall = 0.0
        self.reset = False
        self.image_wall = 0.0
        self.image_ns = 0
        self.joint_stamps = {}
        self.sync_timeout = 10.0
        self.recovery_id = 0
        self.pub = self.create_publisher(JointState, args.command_topic, 10)
        self.subs = [
            self.create_subscription(
                Image, args.image_topic, self.image, qos_profile_sensor_data
            ),
            self.create_subscription(
                JointState, args.joint_topic, self.joints, qos_profile_sensor_data
            ),
            self.create_subscription(
                Clock, args.clock_topic, self.clock, qos_profile_sensor_data
            ),
        ]

    def image(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        if msg.height <= 0 or msg.width <= 0 or msg.step < msg.width * 3:
            return
        if self.environment == "isaac" and (msg.height != 360 or msg.width != 640):
            return
        if len(msg.data) != msg.height * msg.step:
            return
        rgb = (
            np.frombuffer(msg.data, dtype=np.uint8)
            .reshape(msg.height, msg.step)[:, : msg.width * 3]
            .reshape(msg.height, msg.width, 3)
        )
        if msg.encoding == "bgr8":
            rgb = rgb[:, :, ::-1]
        if (msg.height, msg.width) != (360, 640):
            rgb = np.asarray(
                PILImage.fromarray(rgb).resize((640, 360), PILImage.Resampling.BILINEAR)
            )
        with self.lock:
            self.rgb = rgb.copy()
            self.image_wall = time.monotonic()
            self.image_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

    def joints(self, msg):
        with self.lock:
            for name, value in zip(msg.name, msg.position):
                name = self.model_names.get(name)
                if name is None:
                    continue
                self.positions[name] = value
                self.received[name] = time.monotonic()
                self.joint_stamps[name] = (
                    msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                )

    def clock(self, msg):
        stamp = msg.clock.sec * 10**9 + msg.clock.nanosec
        with self.lock:
            if self.clock_ns is not None and stamp < self.clock_ns:
                self.reset = True
            if stamp != self.clock_ns:
                self.clock_wall = time.monotonic()
            self.clock_ns = stamp

    def snapshot(self):
        deadline = time.monotonic() + self.sync_timeout
        waiting = False
        while True:
            try:
                result = self._snapshot()
                if waiting:
                    print(
                        "[SYNC READY] Fresh synchronized input; discard old prediction and re-infer.",
                        flush=True,
                    )
                return result
            except ObservationPending as exc:
                if not waiting:
                    self.recovery_id += 1
                    waiting = True
                    print(
                        f"[SYNC WAIT] {exc}; no new targets sent while waiting.",
                        flush=True,
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Input did not recover within {self.sync_timeout:.1f}s: {exc}"
                    ) from exc
                time.sleep(0.01)

    def _snapshot(self):
        with self.lock:
            now = time.monotonic()
            if self.reset:
                raise RuntimeError("Simulation reset; restart trial")
            if (
                self.rgb is None
                or self.clock_ns is None
                or now - self.clock_wall > 2
                or now - self.image_wall > 2
            ):
                raise ObservationPending("Fresh RGB and advancing clock required")
            if any(now - self.received.get(j, 0) > 2 for j in JOINTS):
                raise ObservationPending("Joint states stale")
            stamps = [self.image_ns, *[self.joint_stamps[j] for j in JOINTS]]
            if max(stamps) - min(stamps) > 750_000_000:
                raise ObservationPending(
                    "RGB/joint timestamps differ by over 0.75 simulation seconds; waiting for new inputs"
                )
            q = np.array([self.positions[j] for j in JOINTS])
            if not np.isfinite(q).all():
                raise RuntimeError("Nonfinite joints")
            return self.rgb.copy(), q, self.clock_ns

    def command(self, q, stamp):
        msg = JointState()
        msg.name = self.ros_joint_names
        msg.position = [float(x) for x in q]
        msg.header.stamp.sec = int(stamp // 10**9)
        msg.header.stamp.nanosec = int(stamp % 10**9)
        self.pub.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["isaac", "real"], default="isaac")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-timeout", type=float, default=180.0)
    parser.add_argument("--expected-checkpoint", type=Path)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--clock-topic", default="/clock")
    parser.add_argument("--command-topic", default="/arm_joint_commands")
    parser.add_argument("--joint-names", nargs=6, default=JOINTS.copy())
    parser.add_argument("--gripper-max", type=float, default=1.5)
    parser.add_argument(
        "--prompt", default=INSTRUCTION, help="Language instruction sent to the policy"
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=10,
        help="Targets executed at 30Hz before observing again (1-50)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path.home() / ".ros/jetrover_manipulation/live"
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=10.0,
        help="Wall seconds to wait for synchronized input without sending new targets",
    )
    args = parser.parse_args()
    args.prompt = args.prompt.strip()
    if not args.prompt:
        parser.error("prompt must not be empty")
    if args.seed < 0 or args.seed > 2**31 - 1:
        parser.error("seed must be in [0, 2**31-1]")
    if not 0 < args.seconds <= 120:
        parser.error("seconds must be in (0,120]")
    if not 1 <= args.chunk_steps <= 50:
        parser.error("chunk-steps must be in [1,50]")
    if not np.isfinite(args.sync_timeout) or not 0 < args.sync_timeout <= 30:
        parser.error("sync-timeout must be in (0,30]")
    if not np.isfinite(args.server_timeout) or not 0 < args.server_timeout <= 600:
        parser.error("server-timeout must be in (0,600]")
    if len(set(args.joint_names)) != 6:
        parser.error("joint-names must be unique in model vector order")
    if not np.isfinite(args.gripper_max) or not 0 < args.gripper_max <= 1.5:
        parser.error("gripper-max must be in (0,1.5]")
    return args


def connect_policy(args):
    server_info = wait_for_policy(
        args.server_host, args.port, args.server_timeout, args.expected_checkpoint
    )
    if (
        server_info.get("model") != "smolvla"
        or server_info.get("joint_names") != JOINTS
        or server_info.get("action_shape") != [50, 6]
        or server_info.get("fps") != 30
    ):
        raise RuntimeError(f"Incompatible policy server: {server_info}")
    print("[MODEL] " + server_info["checkpoint"], flush=True)
    return server_info


def predict_actions(args, rgb, q, iteration, image_path):
    """Record the input frame and request a validated action chunk."""
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="PNG")
    encoded_image = buf.getvalue()
    image_path.write_bytes(encoded_image)
    request = json.dumps(
        {
            "image": base64.b64encode(encoded_image).decode(),
            "state": q.tolist(),
            "prompt": args.prompt,
            "seed": args.seed + iteration,
        }
    ).encode()
    request_start = time.monotonic()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"http://{args.server_host}:{args.port}",
            data=request,
            headers={"Content-Type": "application/json"},
        ),
        timeout=30,
    ) as response:
        reply = json.load(response)
        if reply.get("prompt") != args.prompt:
            raise RuntimeError(
                "Policy prompt mismatch; update and restart the model server"
            )
        actions = np.asarray(reply["actions"], dtype=float)
    if actions.shape != (50, 6) or not np.isfinite(actions).all():
        raise RuntimeError("Invalid action chunk")
    return actions, request_start


def initial_observation(node, execute):
    deadline = time.monotonic() + 30
    while True:
        try:
            rgb, q, stamp = node.snapshot()
            break
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    if execute:
        wait_for_arm_subscriber(node.pub)
        rgb, q, stamp = node.snapshot()
    return rgb, q, stamp


def wait_for_action_tick(node, previous_tick, observation_epoch):
    """Wait for a 30 Hz action tick, dropping the chunk if input recovers."""
    while True:
        _, current, now = node.snapshot()
        if node.recovery_id != observation_epoch:
            return None
        if now - previous_tick >= 1e9 / 30:
            return current, now
        time.sleep(0.002)


def main():
    args = parse_args()
    limit = partial(limit_target, gripper_max=args.gripper_max)
    server_info = connect_policy(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = args.output / ("trial_" + run_id + ".jsonl")
    image_dir = args.output.parent / (args.output.stem + "_images")
    image_dir.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Bridge(args)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    node.sync_timeout = args.sync_timeout
    last = None
    started = False
    completed = False
    try:
        _, q, stamp = initial_observation(node, args.execute)
        last = q.copy()
        start = stamp
        wall_start = time.monotonic()
        iteration = 0
        tracking_since = None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as log:

            def record(data):
                log.write(json.dumps({"run_id": run_id, **data}) + "\n")
                log.flush()

            record(
                {
                    "event": "start",
                    "execute": args.execute,
                    "joint_names": JOINTS,
                    "prompt": args.prompt,
                    "seed": args.seed,
                    "server": server_info,
                    "initial_state": q.tolist(),
                    "max_target_speed_rad_s": MAX_TARGET_SPEED.tolist(),
                    "ros_joint_names": node.ros_joint_names,
                    "gripper_max_rad": args.gripper_max,
                    "topics": dict(
                        image=args.image_topic,
                        joints=args.joint_topic,
                        clock=args.clock_topic,
                        command=args.command_topic,
                    ),
                    "chunk_steps": args.chunk_steps,
                    "sync_timeout_s": args.sync_timeout,
                    "image_dir": str(image_dir),
                }
            )
            print(f"[LOG] {args.output}\n[IMAGES] {image_dir}", flush=True)
            print(f"[PROMPT] {args.prompt}", flush=True)
            if abs(q[-1]) > 0.05:
                print(
                    f"[WARN] Initial gripper={q[-1]:.4f}; recorded demos started at 0.0 rad.",
                    flush=True,
                )
            while True:
                rgb, q, stamp = node.snapshot()
                observation_epoch = node.recovery_id
                if (
                    stamp - start
                ) / 1e9 >= args.seconds or time.monotonic() - wall_start > 600:
                    break
                image_path = image_dir / f"{iteration:04d}.png"
                actions, request_start = predict_actions(
                    args, rgb, q, iteration, image_path
                )
                _, current, now = node.snapshot()
                if node.recovery_id != observation_epoch:
                    record(
                        {
                            "event": "discard_prediction",
                            "reason": "input_recovered_during_inference",
                        }
                    )
                    iteration += 1
                    continue
                latency = time.monotonic() - request_start
                record(
                    {
                        "event": "prediction",
                        "iteration": iteration,
                        "execute": args.execute,
                        "sim_ns": stamp,
                        "state": q.tolist(),
                        "state_after_inference": current.tolist(),
                        "inference_wall_s": latency,
                        "inference_sim_s": (now - stamp) / 1e9,
                        "image": str(image_path),
                        "predicted": actions.tolist(),
                    }
                )
                print(
                    f"[PREDICT] {iteration} | wait={latency:.3f}s wall/{(now-stamp)/1e9:.3f}s sim"
                    f" | first={actions[0].round(4).tolist()} | last={actions[args.chunk_steps-1].round(4).tolist()}",
                    flush=True,
                )
                if np.max(np.abs(current - q)) > 0.1:
                    # The arm can still converge to the last published target
                    # while synchronous inference runs. Observation delta alone
                    # is not a tracking failure; compare against last below.
                    record(
                        {
                            "event": "inference_motion",
                            "delta_rad": (current - q).tolist(),
                            "last_sent": last.tolist(),
                            "tracking_error_rad": (current[:5] - last[:5]).tolist(),
                        }
                    )
                iteration += 1
                if not args.execute:
                    print(
                        "[CHECK PASS] Live RGB/state inference. No commands sent.",
                        flush=True,
                    )
                    break
                # Receding-horizon execution. Inference remains synchronous; log its hold gaps.
                tick = now
                for action_index, target in enumerate(actions[: args.chunk_steps]):
                    observation = wait_for_action_tick(node, tick, observation_epoch)
                    if observation is None:
                        record(
                            {
                                "event": "cancel_chunk",
                                "reason": "input_recovered",
                                "iteration": iteration - 1,
                                "action_index": action_index,
                            }
                        )
                        break
                    current, now = observation
                    if (now - start) / 1e9 >= args.seconds:
                        break
                    dt = min((now - tick) / 1e9, 0.05)
                    tick = now
                    error = current[:5] - last[:5]
                    worst = int(np.argmax(np.abs(error)))
                    tracking_since = (
                        (now if tracking_since is None else tracking_since)
                        if abs(error[worst]) > 0.3
                        else None
                    )
                    if (
                        tracking_since is not None
                        and now - tracking_since >= 1_000_000_000
                    ):
                        record(
                            {
                                "event": "error",
                                "reason": "tracking",
                                "joint": JOINTS[worst],
                                "actual": current.tolist(),
                                "last_sent": last.tolist(),
                                "error_rad": error.tolist(),
                            }
                        )
                        raise RuntimeError(
                            f"Arm tracking: {JOINTS[worst]} actual={current[worst]:.6f}, "
                            f"command={last[worst]:.6f}, error={error[worst]:+.6f} rad >0.30 for 1 sim second"
                        )
                    last = limit(target, last, dt)
                    node.command(last, now)
                    started = True
                    record(
                        {
                            "event": "command",
                            "iteration": iteration - 1,
                            "action_index": action_index,
                            "sim_ns": now,
                            "dt_s": dt,
                            "actual": current.tolist(),
                            "target": target.tolist(),
                            "sent": last.tolist(),
                            "target_modified_rad": (last - target).tolist(),
                        }
                    )
            record(
                {
                    "event": "end",
                    "execute": args.execute,
                    "predictions": iteration,
                }
            )
        completed = True
        print(
            (
                "[DONE] Trial ended; grasp success NOT established."
                if args.execute
                else "[DONE] Inference check only; no motion trial performed."
            ),
            flush=True,
        )
    finally:
        if started:
            try:
                _, current, stamp = node.snapshot()
                # Normal completion retains all six final policy targets while
                # the base returns. On error, stop arm motion at measured pose.
                if completed:
                    current = last.copy()
                else:
                    current[-1] = last[-1]
                node.command(current, stamp)
            except RuntimeError:
                pass
        rclpy.shutdown()
        thread.join(timeout=2)
        node.destroy_node()


if __name__ == "__main__":
    main()

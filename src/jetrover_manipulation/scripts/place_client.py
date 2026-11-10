#!/usr/bin/env python3
"""Isaac placement after navigation HOME; deterministic, no policy server required."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml
from place_geometry import FloorGuard
from place_plan import segments

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]


def audit(config, actual=None):
    home = list(config["return_knots_rad"][-1]) + [0.45] if actual is None else actual
    guard = FloorGuard(minimum=config["minimum_floor_clearance_m"])
    minimum = float("inf")
    previous = None
    for name, duration, sample in segments(config, home):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid placement duration")
        first = sample(0.0)
        if previous is not None:
            np.testing.assert_allclose(first, previous, atol=1e-10)
        for u in np.linspace(0, 1, 201):
            q = sample(float(u))
            if not np.isfinite(q).all() or np.max(np.abs(q[:5])) > 2.08:
                raise ValueError("Placement joint range")
            clearance, _ = guard.clearance(q)
            minimum = min(minimum, clearance)
            if clearance < guard.minimum:
                raise ValueError("Nominal placement touches floor")
        previous = sample(1.0)
    return dict(
        status="nominal_geometry_pass",
        minimum_floor_clearance_m=minimum,
        release_position_m=config["release_position_m"],
        physical_release_tested=False,
    )


def execute(config, log):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from rosgraph_msgs.msg import Clock
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    class Arm(Node):
        def __init__(self):
            super().__init__("jetrover_place")
            self.values = {}
            self.received = {}
            self.stamp = None
            self.clock_wall = 0.0
            self.reset = False
            self.names = config["ros_joint_names"]
            self.mapping = dict(zip(self.names, JOINTS))
            self.pub = self.create_publisher(JointState, config["command_topic"], 10)
            self.create_subscription(
                JointState, config["joint_topic"], self.joints, qos_profile_sensor_data
            )
            self.create_subscription(
                Clock, config["clock_topic"], self.clock, qos_profile_sensor_data
            )

        def joints(self, msg):
            for name, value in zip(msg.name, msg.position):
                name = self.mapping.get(name)
                if name is not None:
                    self.values[name] = value
                    self.received[name] = time.monotonic()

        def clock(self, msg):
            stamp = msg.clock.sec * 10**9 + msg.clock.nanosec
            if self.stamp is not None and stamp < self.stamp:
                self.reset = True
            if self.stamp != stamp:
                self.clock_wall = time.monotonic()
            self.stamp = stamp

        def state(self):
            if self.reset:
                raise RuntimeError("Simulation reset during placement")
            now = time.monotonic()
            if (
                self.stamp is None
                or now - self.clock_wall > 2
                or any(now - self.received.get(j, 0) > 2 for j in JOINTS)
            ):
                raise RuntimeError("Placement needs fresh joints and advancing clock")
            q = np.array([self.values[j] for j in JOINTS])
            if not np.isfinite(q).all():
                raise RuntimeError("Nonfinite placement joints")
            return q

        def send(self, q):
            msg = JointState()
            msg.name = self.names
            msg.position = [float(v) for v in q]
            msg.header.stamp.sec = int(self.stamp // 10**9)
            msg.header.stamp.nanosec = int(self.stamp % 10**9)
            self.pub.publish(msg)

    node = Arm()
    last = None
    started = False
    try:
        deadline = time.monotonic() + 30
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                q = node.state()
                ready = node.pub.get_subscription_count() > 0
            except RuntimeError:
                if node.reset:
                    raise
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Place input/subscriber timeout")
        offsets = q[:5] - np.array(config["return_knots_rad"][-1])
        print(
            "[PLACE START] "
            + json.dumps(
                dict(actual_rad=q.tolist(), carry_home_error_rad=offsets.tolist())
            ),
            flush=True,
        )
        # Validate the actual alignment, not only the nominal HOME trajectory.
        # Failure here sends no arm target. Runtime floor/tracking checks remain.
        actual_audit = (
            audit(config, q)
            if config["environment"] == "isaac"
            else dict(status="real_carry_model_pose", physical_path_tested=False)
        )
        print("[PLACE ALIGN CHECK] " + json.dumps(actual_audit), flush=True)
        log.write(
            json.dumps(
                dict(
                    event="actual_alignment_audit",
                    actual=q.tolist(),
                    report=actual_audit,
                )
            )
            + "\n"
        )
        log.flush()
        guard = (
            FloorGuard(minimum=config["minimum_floor_clearance_m"])
            if config["environment"] == "isaac"
            else None
        )
        last = q.copy()
        wall_deadline = time.monotonic() + 600
        for name, duration, sample in segments(config, q):
            print(f"[PLACE] {name} | {duration:g} simulation seconds", flush=True)
            start = node.stamp
            tick = start
            tracking_since = None
            while True:
                rclpy.spin_once(node, timeout_sec=0.005)
                current = node.state()
                if time.monotonic() > wall_deadline:
                    raise RuntimeError("Placement wall timeout")
                if node.stamp - tick < 1e9 / 30:
                    continue
                tick = node.stamp
                elapsed = (tick - start) / 1e9
                target = sample(min(1.0, elapsed / duration))
                tracking_since = (
                    (tick if tracking_since is None else tracking_since)
                    if np.max(np.abs(current[:5] - last[:5]))
                    > config["tracking_limit_rad"]
                    else None
                )
                if (
                    tracking_since is not None
                    and tick - tracking_since >= 1_000_000_000
                ):
                    raise RuntimeError("Placement arm tracking error")
                if guard is not None:
                    guard.check_motion(current, last, target)
                node.send(target)
                last = target
                started = True
                log.write(
                    json.dumps(
                        dict(
                            event="command",
                            stage=name,
                            sim_ns=tick,
                            actual=current.tolist(),
                            target=target.tolist(),
                        )
                    )
                    + "\n"
                )
                log.flush()
                if elapsed >= duration:
                    error = np.max(np.abs(current - target))
                    if error < config["settle_tolerance_rad"]:
                        break
                    if elapsed > duration + config["settle_timeout_seconds"]:
                        raise RuntimeError(
                            f"Placement settling timeout: {name}, error={error:.6f}rad"
                        )
        print(
            (
                "[CARRY DONE] Arm HOME measured; gripper position preserved."
                if config.get("carry_only", False)
                else (
                    "[RESET DONE] Arm HOME and gripper open."
                    if config.get("reset_only", False)
                    else "[PLACE DONE] Release sequence and empty-arm HOME completed; object release requires visual review."
                )
            ),
            flush=True,
        )
    finally:
        if started:
            try:
                q = node.state()
                q[5] = last[5]
                node.send(q)
            except RuntimeError:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reset-only", action="store_true")
    parser.add_argument("--carry-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".ros/jetrover_manipulation/place.jsonl",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["reset_only"] = args.reset_only
    config["carry_only"] = args.carry_only
    if args.reset_only and args.carry_only:
        parser.error("Select either reset-only or carry-only")
    if config["environment"] != "isaac" and not args.carry_only:
        raise RuntimeError(
            "Full placement calibration is Isaac-only; use real_place_client for hardware placement"
        )
    report = (
        audit(config)
        if config["environment"] == "isaac"
        else dict(status="real_carry_model_pose", physical_path_tested=False)
    )
    print(
        ("[CARRY CHECK] " if args.carry_only else "[PLACE CHECK] ")
        + json.dumps(report),
        flush=True,
    )
    if not args.execute:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as log:
        log.write(json.dumps(dict(event="start", config=config, audit=report)) + "\n")
        log.flush()
        try:
            execute(config, log)
        except BaseException as exc:
            log.write(json.dumps(dict(event="error", reason=str(exc))) + "\n")
            raise
        log.write(
            json.dumps(dict(event="end", physical_release_automatically_verified=False))
            + "\n"
        )


if __name__ == "__main__":
    main()

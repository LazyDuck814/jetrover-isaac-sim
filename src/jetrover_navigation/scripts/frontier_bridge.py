#!/usr/bin/env python3
"""Coordinate frontier exploration and return to the saved start pose."""
import copy
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose, Spin
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener, TransformException
from frontier_exploration_ros2.srv import ControlExploration
from costmap_projection import project_grid


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class FrontierBridge(Node):
    def __init__(self):
        super().__init__("frontier_bridge")
        defaults = dict(
            auto_start=True,
            map_frame="map",
            base_frame="base_link",
            home_file="~/.ros/jetrover_navigation/home.json",
            initial_spin=False,
        )
        self.p = {k: self.declare_parameter(k, v).value for k, v in defaults.items()}
        self.tf = Buffer(node=self)
        self.listener = TransformListener(self.tf, self)
        self.control = self.create_client(ControlExploration, "/control_exploration")
        self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.spin = ActionClient(self, Spin, "/spin")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status = self.create_publisher(String, "/navigation/state", qos)
        self.home_pub = self.create_publisher(PoseStamped, "/navigation/home", qos)
        self.local_pub = self.create_publisher(
            OccupancyGrid, "/navigation/frontier/local_costmap", qos
        )
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, qos)
        self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self.local_cb, 10
        )
        for name, callback in (
            ("start_exploration", self.start_cb),
            ("stop_exploration", self.stop_cb),
            ("return_home", self.return_cb),
        ):
            self.create_service(Trigger, "/navigation/" + name, callback)
        self.mode = "EXPLORE" if self.p["auto_start"] else "IDLE"
        self.pending_local = []
        self.projection_warning_wall = 0.0
        self.home = None
        self.grid = None
        self.local_wall = None
        self.last_map_wall = None
        self.external_idle = False
        self.external_running = False
        self.control_future = None
        self.control_started = 0.0
        self.last_control = 0.0
        self.goal_handle = None
        self.pending = False
        self.canceling = False
        self.last_clock = None
        self.reset = False
        self.last_report = None
        self.spin_attempted = False
        self.started = 0.0
        self.goal_kind = None
        self.goal_timed_out = False
        self.create_timer(0.5, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def report(self, text):
        self.status.publish(String(data=text))
        if text != self.last_report:
            self.get_logger().info(text)
            self.last_report = text

    def map_cb(self, msg):
        self.grid = msg
        self.last_map_wall = time.monotonic()

    def pose(self):
        try:
            t = self.tf.lookup_transform(
                self.p["map_frame"], self.p["base_frame"], Time()
            )
            return (
                t.transform.translation.x,
                t.transform.translation.y,
                yaw(t.transform.rotation),
            )
        except TransformException:
            return None

    def message(self, pose):
        m = PoseStamped()
        m.header.frame_id = self.p["map_frame"]
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = pose[0]
        m.pose.position.y = pose[1]
        m.pose.orientation.z = math.sin(pose[2] / 2)
        m.pose.orientation.w = math.cos(pose[2] / 2)
        return m

    def local_cb(self, msg):
        self.pending_local.append(msg)
        self.pending_local = self.pending_local[-10:]
        self.retry_local()

    def retry_local(self):
        # Retry TF at the original costmap timestamp.
        for msg in list(self.pending_local):
            if self.project_local(msg):
                self.pending_local.remove(msg)

    def project_local(self, msg):
        try:
            transform = (0.0, 0.0, 0.0)
            if msg.header.frame_id != self.p["map_frame"]:
                t = self.tf.lookup_transform(
                    self.p["map_frame"],
                    msg.header.frame_id,
                    Time.from_msg(msg.header.stamp),
                )
                transform = (
                    t.transform.translation.x,
                    t.transform.translation.y,
                    yaw(t.transform.rotation),
                )
            i = msg.info
            if abs(i.origin.orientation.x) + abs(i.origin.orientation.y) > 1e-6:
                raise ValueError("Non-planar costmap")
            data, w, h, origin = project_grid(
                msg.data,
                i.width,
                i.height,
                i.resolution,
                (i.origin.position.x, i.origin.position.y, yaw(i.origin.orientation)),
                transform,
            )
            out = copy.deepcopy(msg)
            out.header.frame_id = self.p["map_frame"]
            out.info.width = w
            out.info.height = h
            out.info.origin.position.x = origin[0]
            out.info.origin.position.y = origin[1]
            out.info.origin.orientation.x = out.info.origin.orientation.y = (
                out.info.origin.orientation.z
            ) = 0.0
            out.info.origin.orientation.w = 1.0
            out.data = data
            self.local_pub.publish(out)
            self.local_wall = time.monotonic()
            return True
        except TransformException as exc:
            age = (
                self.get_clock().now().nanoseconds
                - msg.header.stamp.sec * 10**9
                - msg.header.stamp.nanosec
            ) / 1e9
            if age > 1.0:
                if time.monotonic() - self.projection_warning_wall > 5.0:
                    self.get_logger().warning(
                        "Costmap TF missing for >1 simulation second; dropping stale projection: "
                        + str(exc)
                    )
                    self.projection_warning_wall = time.monotonic()
                return True
            return False
        except ValueError as exc:
            self.get_logger().error("Invalid costmap: " + str(exc))
            return True

    def cancel_own(self):
        if self.goal_handle and not self.canceling:
            self.canceling = True
            self.goal_handle.cancel_goal_async()

    def start_cb(self, request, response):
        response.success = not self.reset
        response.message = (
            "Exploration requested"
            if response.success
            else "Restart required after clock reset"
        )
        if response.success:
            self.mode = "EXPLORE"
            self.cancel_own()
        return response

    def stop_cb(self, request, response):
        self.mode = "IDLE"
        self.external_idle = False
        self.cancel_own()
        # Clear the previous status so this stop request gets a fresh IDLE.
        self.last_report = None
        response.success = True
        response.message = "Stop requested; wait for /navigation/state IDLE"
        return response

    def return_cb(self, request, response):
        response.success = self.home is not None and not self.reset
        response.message = (
            "Return requested; wait for HOME_REACHED"
            if response.success
            else "HOME unavailable"
        )
        if response.success:
            self.mode = "RETURN"
            self.external_idle = False
            self.cancel_own()
        return response

    def command(self, action):
        if self.control_future or not self.control.service_is_ready():
            return
        if time.monotonic() - self.last_control < 1.0:
            return
        req = ControlExploration.Request()
        req.action = action
        req.delay_seconds = 0.0
        req.quit_after_stop = False
        self.control_started = self.last_control = time.monotonic()
        self.control_future = self.control.call_async(req)
        self.control_future.add_done_callback(
            lambda f, a=action: self.control_reply(f, a)
        )

    def control_reply(self, future, action):
        self.control_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.report("FRONTIER_SERVICE_ERROR: " + str(exc))
            return
        self.external_idle = (
            result.accepted and result.state == ControlExploration.Request.STATE_IDLE
        )
        self.external_running = (
            result.accepted and result.state == ControlExploration.Request.STATE_RUNNING
        )
        if not result.accepted:
            self.report("FRONTIER_CONTROL_REJECTED: " + result.message)

    def send(self, kind):
        self.pending = True
        self.goal_kind = kind
        self.started = self.now()
        self.goal_timed_out = False
        if kind == "HOME":
            req = NavigateToPose.Goal()
            req.pose = self.message(self.home)
            client = self.nav
        else:
            req = Spin.Goal()
            req.target_yaw = 2 * math.pi
            req.time_allowance.sec = 60
            client = self.spin
        client.send_goal_async(req).add_done_callback(self.accepted)

    def accepted(self, future):
        self.pending = False
        try:
            self.goal_handle = future.result()
        except Exception as exc:
            self.goal_handle = None
            self.mode = "IDLE"
            self.report("GOAL_ERROR: " + str(exc))
            return
        if not self.goal_handle.accepted:
            self.goal_handle = None
            if self.goal_kind == "HOME":
                self.mode = "IDLE"
            self.report("GOAL_REJECTED")
            return
        self.goal_handle.get_result_async().add_done_callback(self.finished)
        if (self.goal_kind == "HOME" and self.mode != "RETURN") or (
            self.goal_kind == "SPIN" and self.mode != "EXPLORE"
        ):
            self.cancel_own()

    def finished(self, future):
        try:
            ok = future.result().status == 4
        except Exception:
            ok = False
        kind = self.goal_kind
        was_cancel = self.canceling
        self.goal_handle = None
        self.canceling = False
        if kind == "HOME" and (not was_cancel or self.goal_timed_out):
            self.mode = "IDLE"
            self.report("HOME_REACHED" if ok else "HOME_FAILED")
        elif kind == "SPIN":
            self.report(
                "INITIAL_OBSERVATION_DONE" if ok else "INITIAL_OBSERVATION_FAILED"
            )

    def save_home(self, pose, now):
        self.home = pose
        self.home_pub.publish(self.message(pose))
        path = Path(self.p["home_file"]).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    dict(
                        frame=self.p["map_frame"],
                        x=pose[0],
                        y=pose[1],
                        yaw=pose[2],
                        sim_time=now,
                    ),
                    indent=2,
                )
            )
        except OSError as exc:
            self.report("HOME_FILE_WRITE_ERROR: " + str(exc))
        self.report("HOME_SAVED")

    def tick(self):
        if getattr(self, "pending_local", None):
            self.retry_local()
        now = self.now()
        if self.last_clock is not None and now < self.last_clock:
            self.reset = True
            self.mode = "IDLE"
            self.home = None
            self.external_idle = False
        advancing = self.last_clock is None or now > self.last_clock
        self.last_clock = now
        if self.reset:
            self.cancel_own()
            self.command(ControlExploration.Request.ACTION_STOP)
            self.report("CLOCK_RESET_RESTART_REQUIRED")
            return
        if self.control_future:
            if time.monotonic() - self.control_started > 10:
                self.report("WAITING_FRONTIER_CONTROL_NO_HANDOVER")
            return
        if self.pending:
            return
        if self.goal_handle:
            if now - self.started > 120:
                self.goal_timed_out = True
                if self.goal_kind == "HOME":
                    self.mode = "IDLE"
                self.cancel_own()
                self.report("WAITING_OWN_GOAL_CANCEL")
            return
        if not self.control.service_is_ready():
            self.report("WAITING_FRONTIER_SERVICE")
            return
        if self.mode != "EXPLORE":
            if not self.external_idle:
                self.command(ControlExploration.Request.ACTION_STOP)
                self.report("WAITING_FRONTIER_STOP")
                return
            if self.mode == "RETURN":
                if self.nav.server_is_ready():
                    self.send("HOME")
                    self.report("RETURNING_HOME")
                return
            if self.last_report not in ("HOME_REACHED", "HOME_FAILED"):
                self.report("IDLE")
            # Capture initial HOME below even in inactive mode.
            if self.home is not None:
                return
        if (
            not advancing
            or self.grid is None
            or self.last_map_wall is None
            or time.monotonic() - self.last_map_wall > 15
        ):
            self.report("WAITING_MAP_CLOCK")
            return
        pose = self.pose()
        if pose is None:
            self.report("WAITING_MAP_TF")
            return
        if self.grid.header.frame_id != self.p["map_frame"]:
            self.report("MAP_FRAME_MISMATCH")
            return
        if self.home is None:
            self.save_home(pose, now)
        if self.mode == "IDLE":
            return
        if self.external_running:
            return
        if self.local_wall is None or time.monotonic() - self.local_wall > 15:
            self.report("WAITING_PROJECTED_COSTMAP")
            return
        if self.p["initial_spin"] and not self.spin_attempted:
            if not self.spin.server_is_ready():
                self.report("WAITING_NAV2_SPIN")
                return
            self.spin_attempted = True
            self.send("SPIN")
            self.report("INITIAL_OBSERVATION")
            return
        self.command(ControlExploration.Request.ACTION_START)
        self.report("FRONTIER_START_REQUESTED")


def main():
    rclpy.init()
    node = FrontierBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cancel_own()
            node.command(ControlExploration.Request.ACTION_STOP)
        end = time.monotonic() + 3.0
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.external_idle and node.goal_handle is None:
                break
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

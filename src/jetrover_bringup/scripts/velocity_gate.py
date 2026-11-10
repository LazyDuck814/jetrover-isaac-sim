#!/usr/bin/env python3
"""Single final Twist publisher; separate navigation and approach inputs."""
import time
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from mission_core import VelocitySelection


class Gate(Node):
    def __init__(self):
        super().__init__("mission_velocity_gate")
        values = {
            "output_topic": "/cmd_vel",
            "navigation_topic": "/jetrover_bringup/navigation_velocity",
            "approach_topic": "/jetrover_bringup/approach_velocity",
            "source_topic": "/mission/velocity_source",
            "timeout": 2.0,
        }
        p = {k: self.declare_parameter(k, v).value for k, v in values.items()}
        self.get_logger().info(
            "[VELOCITY ROUTER] Navigation/approach selection only; collision checking is handled by Nav2"
        )
        self.selection = VelocitySelection(p["timeout"])
        self.pub = self.create_publisher(Twist, p["output_topic"], 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, p["source_topic"], self.select, qos)
        for name in ("navigation", "approach"):
            self.create_subscription(
                Twist,
                p[name + "_topic"],
                lambda msg, source=name: self.selection.receive(
                    source, msg, time.monotonic()
                ),
                10,
            )
        self.timer = self.create_timer(
            0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def select(self, msg):
        self.selection.select(msg.data, time.monotonic())
        if self.selection.mode == "hold":
            self.pub.publish(Twist())

    def tick(self):
        self.pub.publish(self.selection.output(time.monotonic()) or Twist())


def main():
    rclpy.init()
    node = Gate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

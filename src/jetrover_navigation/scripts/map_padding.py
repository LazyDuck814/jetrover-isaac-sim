#!/usr/bin/env python3
"""Expand map extent with UNKNOWN cells; never invent observed free space."""
import copy, math
from array import array


def padded_map(msg, margin_m):
    i = msg.info
    if i.resolution <= 0 or margin_m < 0 or len(msg.data) != i.width * i.height:
        raise ValueError("Invalid map geometry")
    n = math.ceil(margin_m / i.resolution)
    if not n:
        return copy.deepcopy(msg)
    out = copy.deepcopy(msg)
    w = i.width + 2 * n
    h = i.height + 2 * n
    values = array("b", [-1]) * (w * h)
    for y in range(i.height):
        values[(y + n) * w + n : (y + n) * w + n + i.width] = array(
            "b", msg.data[y * i.width : (y + 1) * i.width]
        )
    q = i.origin.orientation
    if abs(q.x) + abs(q.y) > 1e-6:
        raise ValueError("Non-planar map origin")
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    d = n * i.resolution
    out.info.origin.position.x -= d * (math.cos(yaw) - math.sin(yaw))
    out.info.origin.position.y -= d * (math.sin(yaw) + math.cos(yaw))
    out.info.width = w
    out.info.height = h
    out.data = values
    return out


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid

    rclpy.init()

    class Padding(Node):
        def __init__(self):
            super().__init__("jetrover_map_padding")
            self.margin = self.declare_parameter("margin_m", 0.6).value
            qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.pub = self.create_publisher(OccupancyGrid, "/map", qos)
            self.create_subscription(
                OccupancyGrid, "/navigation/slam_map", self.update, qos
            )

        def update(self, msg):
            self.pub.publish(padded_map(msg, self.margin))

    node = Padding()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

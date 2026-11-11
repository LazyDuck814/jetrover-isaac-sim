#!/usr/bin/env python3
"""Add an unknown-cell border to the SLAM map."""

import copy
import math
from array import array


def padded_map(msg, margin_m):
    info = msg.info
    if info.resolution <= 0 or margin_m < 0 or len(msg.data) != info.width * info.height:
        raise ValueError("Invalid map geometry")
    margin_cells = math.ceil(margin_m / info.resolution)
    padded = copy.deepcopy(msg)
    if not margin_cells:
        return padded

    width = info.width + 2 * margin_cells
    height = info.height + 2 * margin_cells
    cells = array("b", [-1]) * (width * height)
    for row in range(info.height):
        source = row * info.width
        destination = (row + margin_cells) * width + margin_cells
        cells[destination : destination + info.width] = array(
            "b", msg.data[source : source + info.width]
        )
    q = info.origin.orientation
    if abs(q.x) + abs(q.y) > 1e-6:
        raise ValueError("Non-planar map origin")
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    # Shift along both grid axes, which may be rotated relative to the map frame.
    offset = margin_cells * info.resolution
    padded.info.origin.position.x -= offset * (math.cos(yaw) - math.sin(yaw))
    padded.info.origin.position.y -= offset * (math.sin(yaw) + math.cos(yaw))
    padded.info.width = width
    padded.info.height = height
    padded.data = cells
    return padded


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

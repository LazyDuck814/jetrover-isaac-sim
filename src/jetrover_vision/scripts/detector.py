#!/usr/bin/env python3
"""Detect the target in the latest RGB frame and estimate its 3D position."""
from collections import deque
from pathlib import Path
from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)
import json
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from geometry import register_depth, rigid_transform, sample_target


def default_model_path():
    try:
        share = Path(get_package_share_directory("jetrover_vision"))
    except PackageNotFoundError:
        # Support direct execution before the package is built.
        share = Path(__file__).resolve().parents[1]
    return str(share / "weights" / "best.pt")


def stamp_seconds(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Detector(Node):
    def __init__(self):
        super().__init__("jetrover_detector")
        defaults = {
            "model_path": default_model_path(),
            "target_class": "redblock",
            "device": "cpu",
            "confidence": 0.30,
            "imgsz": 640,
            "max_inference_hz": 10.0,
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_raw",
            "color_info_topic": "/camera/color/camera_info",
            "depth_info_topic": "/camera/depth/camera_info",
            "base_frame": "base_link",
            "use_depth": True,
            "sync_slop_sec": 0.05,
            "max_observation_age_sec": 0.5,
            "depth_wait_wall_sec": 0.75,
            "min_depth_m": 0.2,
            "max_depth_m": 2.5,
            "min_depth_pixels": 3,
            "jpeg_quality": 80,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if (
            self.p["max_inference_hz"] <= 0
            or self.p["sync_slop_sec"] < 0
            or self.p["depth_wait_wall_sec"] < 0
        ):
            raise ValueError("Invalid inference rate or synchronization tolerance")
        path = Path(self.p["model_path"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        names = self.model.names
        names = dict(enumerate(names)) if isinstance(names, list) else names
        target = self.p["target_class"].strip().lower()
        self.class_ids = [
            int(i) for i, name in names.items() if str(name).lower() == target
        ]
        if not self.class_ids:
            raise ValueError(f"Target {target!r} missing from model classes: {names}")
        self.bridge = CvBridge()
        self.tf = Buffer(node=self)
        self.listener = TransformListener(self.tf, self)
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.depths = deque(maxlen=30)
        self.infos = {}
        self.last_rgb_stamp = None
        self.generation = 0
        self.last_rgb_wall = None
        self.started = time.monotonic()
        self.status_pub = self.create_publisher(String, "/vision/target_status", 5)
        self.image_pub = self.create_publisher(Image, "/vision/image_result", 2)
        self.jpeg_pub = self.create_publisher(
            CompressedImage, "/vision/image_result/compressed", 2
        )
        self.camera_point_pub = self.create_publisher(
            PointStamped, "/vision/target_point_camera", 5
        )
        self.base_point_pub = self.create_publisher(
            PointStamped, "/vision/target_point_base", 5
        )
        self.create_subscription(
            Image, self.p["rgb_topic"], self.rgb_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.p["depth_topic"], self.depth_callback, qos_profile_sensor_data
        )
        for key, topic in [
            ("color", self.p["color_info_topic"]),
            ("depth", self.p["depth_info_topic"]),
        ]:
            self.create_subscription(
                CameraInfo,
                topic,
                lambda msg, key=key: self.info_callback(key, msg),
                qos_profile_sensor_data,
            )
        self.create_timer(5.0, self.watchdog)
        self.worker = threading.Thread(target=self.process_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Loaded {path}; classes={names}; device={self.p["device"]}'
        )

    def info_callback(self, key, msg):
        with self.lock:
            self.infos[key] = msg

    def depth_callback(self, msg):
        with self.lock:
            self.depths.append(msg)

    def rgb_callback(self, msg):
        stamp = stamp_seconds(msg)
        with self.lock:
            if self.last_rgb_stamp is not None and stamp < self.last_rgb_stamp:
                self.generation += 1
                self.depths.clear()  # Timeline reset: do not reuse previous episode observations.
                self.infos.clear()
            self.last_rgb_stamp = stamp
            self.last_rgb_wall = time.monotonic()
            generation = self.generation
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass
        self.pending.put_nowait((generation, msg))

    def watchdog(self):
        age = time.monotonic() - (self.last_rgb_wall or self.started)
        if age > 5:
            self.get_logger().warning(
                f"No new RGB for {age:.1f}s: check Play, ROS_DOMAIN_ID and topic"
            )

    def matrix(self, target, source, stamp):
        if not target or not source:
            raise ValueError("Image/CameraInfo frame_id is empty")
        if target == source:
            return np.eye(4)
        tr = self.tf.lookup_transform(target, source, Time.from_msg(stamp)).transform
        return rigid_transform(
            [tr.translation.x, tr.translation.y, tr.translation.z],
            [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w],
        )

    def process_loop(self):
        next_run = 0.0
        while not self.stop.is_set():
            if self.stop.wait(max(0, next_run - time.monotonic())):
                break
            try:
                generation, msg = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            next_run = time.monotonic() + 1 / self.p["max_inference_hz"]
            try:
                self.process(generation, msg)
            except Exception as exc:
                if self.stop.is_set() or not rclpy.ok():
                    break
                self.get_logger().error(f"Inference failed: {exc}")

    def process(self, generation, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self.model.predict(
            image,
            conf=self.p["confidence"],
            classes=self.class_ids,
            imgsz=self.p["imgsz"],
            device=self.p["device"],
            verbose=False,
        )[0]
        annotated = result.plot()
        age = (self.get_clock().now().nanoseconds * 1e-9) - stamp_seconds(msg)
        with self.lock:
            if generation != self.generation:
                return
        stale = (
            age < -self.p["sync_slop_sec"] or age > self.p["max_observation_age_sec"]
        )
        status = {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "base_frame": self.p["base_frame"],
            "target": self.p["target_class"],
            "width": msg.width,
            "height": msg.height,
            "found": False,
            "stale": bool(stale),
            "age_sec": float(age),
            "class_name": None,
            "class_id": None,
            "box": None,
            "confidence": None,
            "depth_valid": False,
            "base_valid": False,
            "target_xyz_camera": None,
            "target_xyz_base": None,
            "reason": "no_detection",
        }
        boxes = result.boxes
        if boxes is not None and len(boxes):
            index = int(boxes.conf.argmax().item())
            box = boxes.xyxy[index].cpu().numpy().tolist()
            status.update(
                found=True,
                box=box,
                confidence=float(boxes.conf[index].item()),
                class_id=int(boxes.cls[index].item()),
                class_name=self.p["target_class"],
                reason="stale_observation" if stale else "depth_disabled",
            )
            if self.p["use_depth"] and not stale:
                self.localize(msg, image.shape[:2], box, status)
        label = f'{self.p["target_class"]}: {status["reason"]}'
        if status["depth_valid"]:
            label += f' Z={status["target_xyz_camera"][2]:.3f}m'
        cv2.putText(
            annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1
        )
        with self.lock:
            if generation != self.generation:
                return
        self.publish_detection(msg.header, annotated, status)

    def publish_detection(self, header, annotated, status):
        output = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        output.header = header
        self.image_pub.publish(output)
        ok, jpeg = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.p["jpeg_quality"]]
        )
        if ok:
            compressed = CompressedImage()
            compressed.header = header
            compressed.format = "bgr8; jpeg compressed bgr8"
            compressed.data = jpeg.tobytes()
            self.jpeg_pub.publish(compressed)
        output_status = String()
        output_status.data = json.dumps(status, allow_nan=False)
        self.status_pub.publish(output_status)

    def localize(self, rgb, shape, box, status):
        try:
            # Wait in the worker so the ROS executor can receive depth frames.
            deadline = time.monotonic() + self.p["depth_wait_wall_sec"]
            rgb_stamp = stamp_seconds(rgb)
            while True:
                with self.lock:
                    snapshots = list(self.depths)
                if (
                    snapshots
                    and max(stamp_seconds(d) for d in snapshots) >= rgb_stamp - 1e-6
                ):
                    break
                if self.stop.wait(0.01) or time.monotonic() >= deadline:
                    break
            with self.lock:
                infos = dict(self.infos)
                depths = list(self.depths)
            if set(infos) != {"color", "depth"}:
                raise ValueError("Waiting for both CameraInfo topics")
            if not depths:
                raise ValueError("Waiting for depth image")
            depth_msg = min(
                depths, key=lambda d: abs(stamp_seconds(d) - stamp_seconds(rgb))
            )
            delta = abs(stamp_seconds(depth_msg) - stamp_seconds(rgb))
            status["depth_stamp_delta_sec"] = delta
            if delta > self.p["sync_slop_sec"] + 1e-6:
                raise ValueError("No depth image within timestamp tolerance")
            ci, di = infos["color"], infos["depth"]
            if (ci.height, ci.width) != shape or (di.height, di.width) != (
                depth_msg.height,
                depth_msg.width,
            ):
                raise ValueError("CameraInfo dimensions do not match images")
            if (
                ci.header.frame_id != rgb.header.frame_id
                or di.header.frame_id != depth_msg.header.frame_id
            ):
                raise ValueError("CameraInfo/image frame_id mismatch")
            if any(abs(float(x)) > 1e-8 for info in (ci, di) for x in info.d):
                raise ValueError(
                    "Nonzero lens distortion requires calibrated rectification"
                )
            depth = np.asarray(
                self.bridge.imgmsg_to_cv2(depth_msg, "passthrough"), dtype=float
            )
            if depth_msg.encoding == "16UC1":
                depth *= 0.001
            elif depth_msg.encoding != "32FC1":
                raise ValueError(f"Unsupported depth encoding: {depth_msg.encoding}")
            transform = self.matrix(
                rgb.header.frame_id, depth_msg.header.frame_id, rgb.header.stamp
            )
            registered = register_depth(
                depth,
                di.k,
                ci.k,
                transform,
                shape,
                self.p["min_depth_m"],
                self.p["max_depth_m"],
            )
            point, count = sample_target(
                registered, box, ci.k, self.p["min_depth_pixels"]
            )
            status.update(
                depth_valid=True,
                target_xyz_camera=point.tolist(),
                valid_depth_pixels=count,
                reason="camera_point_only",
            )
            self.publish_point(self.camera_point_pub, rgb.header, point)
            to_base = self.matrix(
                self.p["base_frame"], rgb.header.frame_id, rgb.header.stamp
            )
            point_base = to_base[:3, :3] @ point + to_base[:3, 3]
            # Depth and TF lookup can outlast the observation's age limit.
            age = self.get_clock().now().nanoseconds * 1e-9 - rgb_stamp
            status["age_sec"] = float(age)
            if (
                age < -self.p["sync_slop_sec"]
                or age > self.p["max_observation_age_sec"]
            ):
                status["stale"] = True
                raise ValueError("Observation became stale during depth localization")
            status.update(
                base_valid=True, target_xyz_base=point_base.tolist(), reason="ok"
            )
            self.publish_point(
                self.base_point_pub, rgb.header, point_base, self.p["base_frame"]
            )
        except Exception as exc:
            status["reason"] = str(exc)

    @staticmethod
    def publish_point(publisher, header, xyz, frame=None):
        msg = PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = frame or header.frame_id
        msg.point.x, msg.point.y, msg.point.z = map(float, xyz)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Detector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop.set()
            node.worker.join()  # Wait until inference stops before destroying publishers.
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

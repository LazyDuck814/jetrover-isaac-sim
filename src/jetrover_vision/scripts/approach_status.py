"""Translate Isaac detector output to the original real-robot status fields."""

import math


def adapt_status(
    source,
    distance_threshold=0.35,
    center_margin_ratio=0.02,
    target_x=None,
    base_frame="base_link",
):
    result = dict(source)
    result.update(
        direction="", reached=False, depth_valid=False, depth_from_memory=False
    )
    if not source.get("found") or source.get("stale", True):
        result["found"] = False
        return result
    box = source.get("box")
    width = source.get("width", 0)
    if not box or len(box) != 4 or width <= 0:
        result["found"] = False
        return result
    grasp_x = int(round((box[0] + box[2]) / 2))
    image_center = width // 2
    center_margin = int(width * center_margin_ratio)
    result["grasp_x"] = grasp_x
    if grasp_x < image_center - center_margin:
        result["direction"] = "left"
    elif grasp_x > image_center + center_margin:
        result["direction"] = "right"
    else:
        result["direction"] = "center"
    point = source.get("target_xyz_camera")
    if source.get("depth_valid") and point and len(point) == 3:
        distance = float(point[2])
        if math.isfinite(distance) and distance > 0:
            # Match the real-robot localizer's depth offset.
            distance += 0.03
            result.update(
                depth_valid=True,
                distance_m=distance,
                reached=distance <= distance_threshold
                and result["direction"] == "center",
            )
    if target_x is not None:
        # The grasp handover uses forward distance in the robot frame.
        base = source.get("target_xyz_base")
        valid = (
            source.get("base_valid") is True
            and source.get("base_frame") == base_frame
            and isinstance(base, (list, tuple))
            and len(base) == 3
        )
        if valid:
            valid = all(isinstance(v, (int, float)) and math.isfinite(v) for v in base)
        result["base_forward_m"] = float(base[0]) if valid else None
        result["reached"] = bool(
            valid
            and result["depth_valid"]
            and 0 < base[0] <= target_x
            and result["direction"] == "center"
        )
        if not valid:
            result["found"] = False
            result["depth_valid"] = False
    return result

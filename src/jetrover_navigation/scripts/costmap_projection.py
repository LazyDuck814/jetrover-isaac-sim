"""Project costmap cells onto the map-aligned grid."""

import math


def project_grid(data, width, height, resolution, origin, transform):
    if resolution <= 0 or len(data) != width * height or width <= 0 or height <= 0:
        raise ValueError("Invalid grid geometry")
    ox, oy, angle = origin
    tx, ty, yaw = transform
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = c * ox - s * oy + tx, s * ox + c * oy + ty
    a = angle + yaw
    c, s = math.cos(a), math.sin(a)

    def point(x, y):
        return (
            bx + c * x * resolution - s * y * resolution,
            by + s * x * resolution + c * y * resolution,
        )

    corners = [
        point(x, y) for x, y in ((0, 0), (width, 0), (0, height), (width, height))
    ]
    x0 = math.floor((min(x for x, y in corners) + 1e-8) / resolution) * resolution
    y0 = math.floor((min(y for x, y in corners) + 1e-8) / resolution) * resolution
    w = math.ceil((max(x for x, y in corners) - x0) / resolution - 1e-8)
    h = math.ceil((max(y for x, y in corners) - y0) / resolution - 1e-8)
    out = [-1] * (w * h)
    for y in range(height):
        for x in range(width):
            value = data[y * width + x]
            if value < 0:
                continue
            vertices = [
                point(x + dx, y + dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))
            ]
            lo_x = max(
                0, math.floor((min(a for a, b in vertices) - x0) / resolution + 1e-8)
            )
            hi_x = min(
                w - 1,
                math.ceil((max(a for a, b in vertices) - x0) / resolution - 1e-8) - 1,
            )
            lo_y = max(
                0, math.floor((min(b for a, b in vertices) - y0) / resolution + 1e-8)
            )
            hi_y = min(
                h - 1,
                math.ceil((max(b for a, b in vertices) - y0) / resolution - 1e-8) - 1,
            )
            for j in range(lo_y, hi_y + 1):
                for i in range(lo_x, hi_x + 1):
                    out[j * w + i] = max(out[j * w + i], value)
    return out, w, h, (x0, y0)

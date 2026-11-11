"""Mission transitions and velocity selection for each phase."""

import math


class TargetWindow:
    """Confirm detections across distinct frames before handing over to approach."""

    def __init__(self, margin=0.15, seconds=0.2, frames=3, top=0.1):
        if not 0 <= margin < 0.5 or not 0 <= top < 1 or seconds < 0 or frames < 1:
            raise ValueError("Invalid target handover window")
        self.margin, self.seconds, self.frames = margin, seconds, frames
        self.top = top
        self.reset()

    def reset(self):
        self.first = self.last = None
        self.count = 0
        self.ready = False

    def update(self, target, valid):
        try:
            width = float(target["width"])
            height = float(target["height"])
            left, top, right, bottom = map(float, target["box"])
            stamp = target["stamp"]
            now = int(stamp["sec"]) + int(stamp["nanosec"]) * 1e-9
            measurements = (width, height, left, top, right, bottom, now)
            inside = (
                valid
                and all(math.isfinite(value) for value in measurements)
                and width > 0
                and height > 0
                and self.margin * width <= left < right <= (1 - self.margin) * width
                and self.top * height <= top < bottom <= height
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            inside = False
        if not inside:
            self.reset()
            return False
        if self.last is not None and now == self.last:
            return self.ready
        if self.last is None or now < self.last or now - self.last > 0.5:
            self.reset()
            self.first = now
        self.last = now
        self.count += 1
        self.ready = (
            self.count >= self.frames and now - self.first >= self.seconds - 1e-8
        )
        return self.ready


class Mission:
    """Translate mission events into commands for the ROS orchestration layer."""

    def __init__(self, execute=False):
        self.state = "WAIT_READY" if execute else "IDLE"
        self.nav_revision = 0
        self.stop_revision = 0
        self.stop_ack = False
        self.idle_after_stop = False
        self.stop_purpose = "approach"
        self.reason = ""
        self.place_after_home = False

    def fail(self, reason):
        if self.state == "FAILED":
            return []
        self.state = "FAILED"
        self.reason = reason
        return ["hold", "kill_child", "stop_navigation"]

    def ready(self):
        if self.state != "WAIT_READY":
            return []
        self.state = "STARTING_EXPLORATION"
        return ["hold", "start_navigation"]

    def target(self, valid):
        if self.state != "EXPLORING" or not valid:
            return []
        return self.stop("approach")

    def approach_lost(self):
        if self.state != "APPROACHING":
            return []
        self.state = "STARTING_EXPLORATION"
        return ["hold", "start_navigation"]

    def stop(self, purpose):
        self.state = "STOPPING_NAVIGATION"
        self.stop_revision = self.nav_revision
        self.stop_ack = self.idle_after_stop = False
        self.stop_purpose = purpose
        return ["hold", "kill_child", "stop_navigation"]

    def handover(self):
        if self.state != "STOPPING_NAVIGATION" or not (
            self.stop_ack and self.idle_after_stop
        ):
            return []
        if self.stop_purpose == "home":
            self.state = "CARRYING_HOME"
            return ["hold", "spawn_carry"]
        self.state = "APPROACHING"
        return ["approach_velocity", "spawn_approach"]

    def service(self, kind, ok):
        expected = {
            "start_navigation": "STARTING_EXPLORATION",
            "stop_navigation": "STOPPING_NAVIGATION",
            "return_home": "REQUESTING_HOME",
        }
        if self.state != expected.get(kind):
            return []
        if not ok:
            return self.fail(kind + " rejected")
        if kind == "start_navigation":
            self.state = "EXPLORING"
            return ["navigation_velocity"]
        if kind == "stop_navigation":
            self.stop_ack = True
            # Accept only an IDLE received after the stop service reply.
            self.idle_after_stop = False
            return self.handover()
        self.state = "RETURNING_HOME"
        return ["navigation_velocity"]

    def navigation(self, text):
        self.nav_revision += 1
        if "CLOCK_RESET" in text:
            return self.fail(text)
        if (
            self.state == "STOPPING_NAVIGATION"
            and self.stop_ack
            and text == "IDLE"
            and self.nav_revision > self.stop_revision
        ):
            self.idle_after_stop = True
            return self.handover()
        if self.state == "RETURNING_HOME":
            if text == "HOME_REACHED":
                if self.place_after_home:
                    self.state = "PLACING"
                    return ["hold", "spawn_place"]
                self.state = "DONE"
                return ["hold"]
            if text in ("HOME_FAILED", "GOAL_REJECTED") or text.startswith(
                "GOAL_ERROR"
            ):
                return self.fail(text)
        return []

    def child_exit(self, kind, code, marker):
        expected = {
            "approach": "APPROACHING",
            "grasp": "GRASPING",
            "carry": "CARRYING_HOME",
            "place": "PLACING",
        }
        if self.state != expected.get(kind):
            return []
        if code != 0 or not marker:
            return self.fail(f"{kind} failed: exit={code}, completion={marker}")
        if kind == "approach":
            self.state = "GRASPING"
            return ["hold", "spawn_grasp"]
        if kind == "place":
            self.place_after_home = False
            self.state = "DONE"
            return ["hold"]
        if kind == "grasp":
            self.place_after_home = True
            # Keep the policy's final arm targets during the drive home.
            self.state = "REQUESTING_HOME"
            return ["hold", "return_home"]
        self.state = "REQUESTING_HOME"
        return ["hold", "return_home"]


class VelocitySelection:
    def __init__(self, timeout):
        self.timeout = timeout
        self.mode = "hold"
        self.heartbeat = None
        self.command = None

    def select(self, mode, now):
        if mode not in ("hold", "navigation", "approach"):
            mode = "hold"
        if mode != self.mode:
            self.command = None
        self.mode, self.heartbeat = mode, now

    def receive(self, source, msg, now):
        if source == self.mode and self.mode != "hold":
            self.command = (msg, now)

    def output(self, now):
        if (
            self.mode == "hold"
            or self.heartbeat is None
            or now - self.heartbeat > self.timeout
        ):
            return None
        if self.command is None or now - self.command[1] > self.timeout:
            return None
        return self.command[0]

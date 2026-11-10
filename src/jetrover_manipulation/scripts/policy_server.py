#!/usr/bin/env python3
"""Local single-request SmolVLA server for the ROS Jazzy Python client."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import traceback

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
DEFAULT_CHECKPOINT = Path(
    "/isaac-sim/workspace/jetrover_rl/logs/smolvla/baseline50_ver2_b64_s20000/checkpoints/020000/pretrained_model"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def load_policy(checkpoint_path):
    """Load the checkpoint and return its prediction function and metadata."""
    import numpy as np
    import torch
    from importlib.metadata import version
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    if version("lerobot") != "0.4.3":
        raise RuntimeError("Use the lerobot==0.4.3 training environment")
    checkpoint = checkpoint_path.resolve(strict=True)
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    if not isinstance(cfg, SmolVLAConfig):
        raise ValueError("Checkpoint is not a SmolVLA policy")
    if list(cfg.image_features) != ["observation.images.wrist"]:
        raise ValueError(f"Checkpoint camera mismatch: {cfg.image_features}")
    if tuple(cfg.input_features["observation.state"].shape) != (6,) or tuple(
        cfg.action_feature.shape
    ) != (6,):
        raise ValueError("Checkpoint must have six state and action dimensions")
    if cfg.chunk_size != 50 or cfg.n_obs_steps != 1 or cfg.adapt_to_pi_aloha:
        raise ValueError(
            "Expected 50-step chunks, one observation and native joint radians"
        )
    cfg.device = "cuda"
    cfg.load_vlm_weights = False  # All learned weights come from this checkpoint.
    policy = (
        SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg).to("cuda").eval()
    )
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    info = dict(
        model="smolvla",
        checkpoint=str(checkpoint),
        joint_names=JOINTS,
        action_shape=[50, 6],
        fps=30,
        action_semantics="absolute_joint_position_target_rad",
    )

    def predict(rgb, state, prompt, seed):
        if (
            rgb.shape != (360, 640, 3)
            or state.shape != (6,)
            or not np.isfinite(state).all()
        ):
            raise ValueError("Expected RGB 360x640x3 and six finite joint values")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt is empty")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        policy.reset()
        # Same [0,1] CHW image convention as the official dataset loader.
        batch = pre(
            {
                "observation.images.wrist": torch.from_numpy(rgb.copy())
                .permute(2, 0, 1)
                .float()
                / 255,
                "observation.state": torch.from_numpy(state.copy()).float(),
                "task": prompt,
            }
        )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch)
            actions = post(normalized).detach().cpu().numpy()
        if actions.shape != (1, 50, 6) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid policy output: {actions.shape}")
        return actions[0].tolist()

    return predict, info


def make_handler(predict, info):
    import numpy as np
    from PIL import Image

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(
                200 if self.path == "/health" else 404,
                info if self.path == "/health" else {},
            )

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                image = Image.open(
                    io.BytesIO(base64.b64decode(payload["image"], validate=True))
                )
                if image.size != (640, 360):
                    raise ValueError("Camera resolution must match training: 640x360")
                rgb = np.asarray(image.convert("RGB"))
                state = np.asarray(payload["state"], dtype=np.float32)
                prompt = payload["prompt"]
                seed = int(payload.get("seed", 42))
                actions = predict(rgb, state, prompt, seed)
                self.send_json(200, dict(actions=actions, prompt=prompt, **info))
            except Exception as exc:
                traceback.print_exc()
                self.send_json(500, {"error": str(exc)})

        def log_message(self, *args):
            pass

    return Handler


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np

    predict, info = load_policy(args.checkpoint)
    handler = make_handler(predict, info)
    # Bind first to detect an occupied port before warm-up.
    server = HTTPServer((args.host, args.port), handler)
    server.timeout = 1
    print(
        "[WARMUP] Loading saved normalization and running dummy inference; no robot commands",
        flush=True,
    )
    predict(
        np.zeros((360, 640, 3), dtype=np.uint8),
        np.zeros(6, dtype=np.float32),
        "Pick up the red cube, return to the initial arm pose, and hold it.",
        42,
    )
    print(f"[READY] http://{args.host}:{args.port} | {info['checkpoint']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Standalone VGGT / VGGT-Omega inference.

Reads RGB frames from a directory or video, runs the chosen model, and writes
per-frame color, depth, extrinsics, intrinsic, and a point cloud to the output
folder — matching the layout produced by `save_vggt_outputs` in main.py.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
from utils import predictions_to_pcd

# vggt-omega lives under submodules/, add it to sys.path on demand.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_VGGT_OMEGA_PATH = os.path.join(_REPO_ROOT, "submodules", "vggt-omega")


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
NON_RGB_IMAGE_SUFFIXES = ("depth.png", "normal.png", "mask.png", "semantic.png", "seg.png")


def _natural_sort_key(filename):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", filename)]


def _list_image_paths(input_path, max_frames, tmp_holder):
    """Return a sorted list of image file paths from a directory or video.

    `tmp_holder` is a list the caller keeps a reference to so the
    TemporaryDirectory used for ffmpeg-extracted frames survives until the
    images have been loaded.
    """
    if os.path.isdir(input_path):
        images = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
            and not f.lower().endswith(NON_RGB_IMAGE_SUFFIXES)
        ]
        rgb_png = [f for f in images if f.lower().endswith("rgb.png")]
        if rgb_png:
            images = rgb_png
        images = sorted(images, key=_natural_sort_key)
        if not images:
            raise ValueError(f"No image files found in directory: {input_path}")
        if max_frames > 0 and len(images) > max_frames:
            indices = np.linspace(0, len(images) - 1, max_frames).astype(int)
            images = [images[i] for i in indices]
        return [os.path.join(input_path, img) for img in images]

    tmp = tempfile.TemporaryDirectory()
    tmp_holder.append(tmp)
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vsync", "0", os.path.join(tmp.name, "frame_%04d.png")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return _list_image_paths(tmp.name, max_frames, tmp_holder)


def _save_outputs(output_path, predictions):
    os.makedirs(os.path.join(output_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "extrinsics"), exist_ok=True)

    np.savetxt(os.path.join(output_path, "intrinsic.txt"), predictions["intrinsic"])
    predictions["point_cloud_data"].export(os.path.join(output_path, "point_cloud.ply"))

    for i, image in enumerate(predictions["colors"]):
        cv2.imwrite(os.path.join(output_path, "color", f"{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    for i, depth in enumerate(predictions["depths"]):
        cv2.imwrite(os.path.join(output_path, "depth", f"{i}.png"), (depth * 1000).astype(np.uint16))
    for i, extrinsic in enumerate(predictions["extrinsics"]):
        np.savetxt(os.path.join(output_path, "extrinsics", f"{i}.txt"), extrinsic)


def _pad_extrinsics(extrinsic_3x4):
    extrinsics = np.pad(extrinsic_3x4, ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0)
    extrinsics[:, 3, 3] = 1
    return extrinsics


def run_vggt(image_paths, checkpoint, device, conf_thres):
    """Original VGGT inference path. Mirrors src/vggt_predict.py:vggt_predict."""
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Loaded {len(images)} frames; tensor shape {tuple(images.shape)}.")

    model = VGGT()
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images.cpu().numpy()

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions["pose_enc_list"] = None

    world_points = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

    point_cloud_data = predictions_to_pcd(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        prediction_mode="Depthmap and Camera Branch",
    )

    colors = (predictions["images"].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = predictions["depth"].squeeze(-1)
    extrinsics = _pad_extrinsics(predictions["extrinsic"])
    intrinsic_mean = np.mean(predictions["intrinsic"], axis=0)

    return {
        "point_cloud_data": point_cloud_data,
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsic": intrinsic_mean,
    }


def run_vggt_omega(image_paths, checkpoint, device, conf_thres, image_resolution):
    """VGGT-Omega inference path. Uses the model under submodules/vggt-omega."""
    if _VGGT_OMEGA_PATH not in sys.path:
        sys.path.insert(0, _VGGT_OMEGA_PATH)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images as load_omega_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = load_omega_images(image_paths, image_resolution=image_resolution).to(device)
    print(f"Loaded {len(images)} frames; tensor shape {tuple(images.shape)}.")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
        predictions_np[key] = value

    depth = predictions_np["depth"]
    num_frames, height, width, _ = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    d = depth[..., 0]
    fx = predictions_np["intrinsic"][:, 0, 0][:, None, None]
    fy = predictions_np["intrinsic"][:, 1, 1][:, None, None]
    cx = predictions_np["intrinsic"][:, 0, 2][:, None, None]
    cy = predictions_np["intrinsic"][:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * d, (y - cy) / fy * d, d], axis=-1)
    rotation = predictions_np["extrinsic"][:, :3, :3]
    translation = predictions_np["extrinsic"][:, :3, 3]
    world_points = np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )
    predictions_np["world_points_from_depth"] = world_points

    point_cloud_data = predictions_to_pcd(
        predictions_np,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        prediction_mode="Depthmap and Camera Branch",
    )

    colors = (predictions_np["images"].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = depth.squeeze(-1)
    extrinsics = _pad_extrinsics(predictions_np["extrinsic"])
    intrinsic_mean = np.mean(predictions_np["intrinsic"], axis=0)

    return {
        "point_cloud_data": point_cloud_data,
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsic": intrinsic_mean,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT or VGGT-Omega and dump depth + camera parameters.")
    parser.add_argument("--input-path", required=True, help="Directory of RGB images or a video file.")
    parser.add_argument("--output-path", required=True, help="Destination folder for outputs.")
    parser.add_argument("--max-frames", type=int, default=0, help="Uniformly subsample to at most this many frames (0 = use all).")
    parser.add_argument("--model", choices=["vggt", "vggt_omega"], default="vggt", help="Which model to load.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint. Required for vggt_omega; defaults to the path baked into src.models for vggt.",
    )
    parser.add_argument("--image-resolution", type=int, default=512, help="VGGT-Omega input resolution.")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="Confidence percentile for point cloud export.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_path, exist_ok=True)

    tmp_holder = []
    image_paths = _list_image_paths(args.input_path, args.max_frames, tmp_holder)
    print(f"Selected {len(image_paths)} images for {args.model}.")

    if args.model == "vggt":
        checkpoint = args.checkpoint or "/mnt/storage2/users/szxie_data/vggt/model_vggt.pt"
        predictions = run_vggt(image_paths, checkpoint, device, args.conf_thres)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --model vggt_omega.")
        predictions = run_vggt_omega(
            image_paths, args.checkpoint, device, args.conf_thres, args.image_resolution
        )

    _save_outputs(args.output_path, predictions)
    print(f"Saved outputs to {args.output_path}")


if __name__ == "__main__":
    main()

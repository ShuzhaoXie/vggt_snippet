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
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
NON_RGB_IMAGE_SUFFIXES = ("depth.png", "normal.png", "mask.png", "semantic.png", "seg.png")
COMMON_IMAGE_SUBDIRS = ("images", "rgb", "color", "colors", "frames", "sampled_images")
SPECIAL_IMAGE_SUBDIRS = {
    "holoscene": ("images",),
    "simrecon": ("images",),
}


def _natural_sort_key(filename):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", filename)]


def _dataset_scene_path(data_root, dataset_name, scene_name):
    dataset_parts = [part for part in dataset_name.replace("\\", "/").split("/") if part]
    return os.path.join(data_root, *dataset_parts, scene_name)


def _dedupe_paths(paths):
    seen = set()
    deduped = []
    for path in paths:
        normalized = os.path.normpath(path)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(path)
    return deduped


def _candidate_input_paths(input_path, dataset_name):
    if not os.path.isdir(input_path):
        return [input_path]

    dataset_root = dataset_name.replace("\\", "/").split("/", 1)[0] if dataset_name else ""
    candidates = []
    for subdir in SPECIAL_IMAGE_SUBDIRS.get(dataset_root, ()):
        candidates.append(os.path.join(input_path, subdir))
    candidates.append(input_path)
    for subdir in COMMON_IMAGE_SUBDIRS:
        candidates.append(os.path.join(input_path, subdir))
    return _dedupe_paths(candidates)


def _find_rgb_images(directory):
    if not os.path.isdir(directory):
        return []

    images = [
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
        and not f.lower().endswith(NON_RGB_IMAGE_SUFFIXES)
    ]
    rgb_png = [f for f in images if f.lower().endswith("rgb.png")]
    if rgb_png:
        images = rgb_png

    images = sorted(images, key=_natural_sort_key)
    return [os.path.join(directory, img) for img in images]


def _find_videos(directory):
    if not os.path.isdir(directory):
        return []

    videos = [
        f for f in os.listdir(directory)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]
    videos = sorted(videos, key=_natural_sort_key)
    return [os.path.join(directory, video) for video in videos]


def _subsample_paths(paths, max_frames):
    if max_frames > 0 and len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        return [paths[i] for i in indices]
    return paths


def _extract_video_frames(video_path, max_frames, tmp_holder):
    tmp = tempfile.TemporaryDirectory()
    tmp_holder.append(tmp)
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vsync", "0", os.path.join(tmp.name, "frame_%04d.png")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return _list_image_paths(tmp.name, max_frames, tmp_holder)


def _list_image_paths(input_path, max_frames, tmp_holder, dataset_name=None, scene_name=None):
    """Return a sorted list of RGB image paths for a scene directory or video.

    `tmp_holder` is a list the caller keeps a reference to so the
    TemporaryDirectory used for ffmpeg-extracted frames survives until the
    images have been loaded.
    """
    del scene_name
    input_path = os.path.expanduser(input_path)

    if os.path.isdir(input_path):
        candidate_paths = _candidate_input_paths(input_path, dataset_name)
        for candidate in candidate_paths:
            images = _find_rgb_images(candidate)
            if images:
                return _subsample_paths(images, max_frames)

        videos = []
        for candidate in candidate_paths:
            videos.extend(_find_videos(candidate))
        videos = _dedupe_paths(videos)
        if len(videos) == 1:
            return _extract_video_frames(videos[0], max_frames, tmp_holder)
        if len(videos) > 1:
            raise ValueError(
                "Multiple video files found; pass --input-path for the intended video: "
                + ", ".join(videos)
            )
        raise ValueError(
            "No RGB image or video files found. Checked: "
            + ", ".join(candidate_paths)
        )

    if os.path.isfile(input_path):
        lower_path = input_path.lower()
        if lower_path.endswith(IMAGE_EXTENSIONS):
            return [input_path]
        return _extract_video_frames(input_path, max_frames, tmp_holder)

    raise ValueError(f"Input path does not exist: {input_path}")


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
    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        default=None,
        help="Directory of RGB images or a video file.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        default=None,
        help="Destination folder for outputs.",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default="./data",
        help="Root containing dataset folders.",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        dest="dataset_name",
        default=None,
        help="Dataset path under --data-root, e.g. holoscene/replica.",
    )
    parser.add_argument(
        "--scene-name",
        "--scene_name",
        dest="scene_name",
        default=None,
        help="Scene name under the dataset folder.",
    )
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        "--n-frames",
        "--n_frames",
        dest="max_frames",
        type=int,
        default=0,
        help="Uniformly subsample to at most this many frames (0 = use all).",
    )
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


def _resolve_io_paths(args):
    if args.dataset_name or args.scene_name:
        if not args.dataset_name or not args.scene_name:
            raise SystemExit("--dataset-name and --scene-name must be provided together.")

        scene_path = _dataset_scene_path(args.data_root, args.dataset_name, args.scene_name)
        input_path = args.input_path or scene_path
        output_path = args.output_path or os.path.join(scene_path, args.model)
        return input_path, output_path

    if not args.input_path:
        raise SystemExit("Provide either --input-path or both --dataset-name and --scene-name.")
    if not args.output_path:
        raise SystemExit("--output-path is required when --dataset-name/--scene-name are not used.")
    return args.input_path, args.output_path


def main():
    args = parse_args()
    input_path, output_path = _resolve_io_paths(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmp_holder = []
    image_paths = _list_image_paths(
        input_path,
        args.max_frames,
        tmp_holder,
        dataset_name=args.dataset_name,
        scene_name=args.scene_name,
    )
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

    _save_outputs(output_path, predictions)
    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()

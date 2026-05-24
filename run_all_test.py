import argparse
import os
import shlex
import subprocess
import sys


dataset_name_to_scenes = {
    "dprecon/replica": [f"scan{i}" for i in range(1, 9)],
    "dprecon/scannetpp": [f"scan{i}" for i in range(1, 7)],
    "dprecon/youtube": [f"scan{i}" for i in range(1, 4)],
    "holoscene/custom": ["siebelgame"],
    "holoscene/gibson": ["Beechwood_0_int", "Beechwood_1_int"],
    "holoscene/replica": ["room_0", "room_1", "room_2"],
    "holoscene/scannetpp": ["67d702f2e8", "7831862f02", "acd69a1746"],
    # "hope-dataset/hope_image": ["test", "valid"],
    # "hope-dataset/hope-image-preview": [f"scene_{i:04d}" for i in range(10)],
    "hope-dataset/hope_video": [f"scene_{i:04d}" for i in range(10)],
    "phyrecon/replica": [f"scan{i}" for i in range(1, 9)],
    "phyrecon/scannetpp": [f"scan{i}" for i in range(1, 8)],
    "simrecon": ["room_4", "scene0000_00", "waldo_kitchen"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT over all configured test scenes.")
    parser.add_argument("--data-root", default="./data", help="Root containing dataset folders.")
    parser.add_argument("--model", choices=["vggt", "vggt_omega"], default="vggt")
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        "--n-frames",
        "--n_frames",
        dest="max_frames",
        type=int,
        default=30,
        help="Uniformly subsample each scene to at most this many frames (-1 = use all).",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint forwarded to run_vggt.py.")
    parser.add_argument("--device", default=None, help="Optional device forwarded to run_vggt.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=None,
        help="Destination folder for outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_name, scene_names in dataset_name_to_scenes.items():
        for scene_name in scene_names:
            command = [
                sys.executable,
                "run_vggt.py",
                "--data-root",
                args.data_root,
                "--dataset-name",
                dataset_name,
                "--scene-name",
                scene_name,
                "--model",
                args.model,
                "--max-frames",
                str(args.max_frames),
                "--output-dir",
                str(args.output_dir),
            ]
            if args.checkpoint:
                command.extend(["--checkpoint", args.checkpoint])
            if args.device:
                command.extend(["--device", args.device])

            print(f"Running command: {shlex.join(command)}")
            if not args.dry_run:
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.log")
                    already_logged = False
                    if os.path.exists(log_path):
                        with open(log_path, "r") as lf:
                            logs = lf.read()
                            if f"--dataset-name {dataset_name} --scene-name {scene_name}" in logs:
                                already_logged = True
                    
                    if not already_logged:
                        if e.returncode in (2, -9, 137):
                            with open(log_path, "a") as lf:
                                lf.write(f"OOM: --dataset-name {dataset_name} --scene-name {scene_name}\n")
                    
                    print(f"Command failed with exit code {e.returncode} on dataset={dataset_name}, scene={scene_name}. Continuing...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a per-identity teacher dataset using the existing pipeline.
This is the first step for compiling a fast identity-specific model.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import is_video


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def list_videos(path: Path):
    if path.is_file():
        return [path]
    videos = []
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    return videos


def save_identity(out_dir: Path, pipe: FasterLivePortraitPipeline):
    out_dir.mkdir(parents=True, exist_ok=True)
    # For single-image identity, use the first src_info entry.
    src_info = pipe.src_infos[0][0]
    x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = src_info
    if hasattr(mask_ori_float, "detach"):
        mask_ori = mask_ori_float.detach().cpu().numpy()
    else:
        mask_ori = mask_ori_float
    if hasattr(M, "detach"):
        M_c2o = M.detach().cpu().numpy()
    else:
        M_c2o = M

    np.savez_compressed(
        out_dir / "identity.npz",
        x_s_info=x_s_info,
        source_lmk=source_lmk,
        R_s=R_s,
        f_s=f_s,
        x_s=x_s,
        x_c_s=x_c_s,
        lip_delta_before_animation=lip_delta_before_animation,
        flag_lip_zero=flag_lip_zero,
        mask_ori=mask_ori,
        M_c2o=M_c2o,
        src_img=pipe.src_imgs[0],
    )


def main():
    parser = argparse.ArgumentParser(description="Generate teacher dataset for identity compilation")
    parser.add_argument("--source", required=True, help="source image path")
    parser.add_argument("--driving", required=True, help="driving video path or directory")
    parser.add_argument("--cfg", default="configs/trt_infer.yaml", help="inference config")
    parser.add_argument("--out_dir", required=True, help="output dataset directory")
    parser.add_argument("--max_frames", type=int, default=-1, help="max frames to export (all if -1)")
    parser.add_argument("--stride", type=int, default=1, help="frame stride")
    parser.add_argument("--motion_res", type=int, default=192, help="resolution for motion input frames")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = True
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)

    ok = pipe.prepare_source(args.source, realtime=False)
    if not ok:
        raise RuntimeError(f"no face detected in source image: {args.source}")

    save_identity(out_dir, pipe)

    driving_path = Path(args.driving)
    videos = list_videos(driving_path)
    if not videos:
        raise RuntimeError(f"no videos found at: {args.driving}")

    meta = {
        "source": args.source,
        "driving": [str(v) for v in videos],
        "cfg": args.cfg,
        "motion_res": args.motion_res,
        "stride": args.stride,
        "max_frames": args.max_frames,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    global_idx = 0
    for vid_path in videos:
        vcap = cv2.VideoCapture(str(vid_path))
        if not vcap.isOpened():
            print(f"warning: failed to open {vid_path}")
            continue
        frame_idx = 0
        first_frame = True
        while vcap.isOpened():
            ret, frame = vcap.read()
            if not ret:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue
            if args.max_frames > 0 and global_idx >= args.max_frames:
                break

            dri_crop, out_crop, out_org, dri_motion_info, details = pipe.run(
                frame,
                pipe.src_imgs[0],
                pipe.src_infos[0],
                first_frame=first_frame,
                realtime=False,
                return_details=True,
            )
            first_frame = False
            if out_crop is None:
                frame_idx += 1
                continue

            # downscale motion input frame
            dri_crop_resized = cv2.resize(dri_crop, (args.motion_res, args.motion_res))

            # pull motion info
            motion_dict = dri_motion_info[0]
            eye_ratio = dri_motion_info[1]
            lip_ratio = dri_motion_info[2]

            # detail info (per-face)
            detail = details[0]
            x_d_i_new = detail["x_d_i_new"]

            sample_path = frames_dir / f"sample_{global_idx:08d}.npz"
            np.savez_compressed(
                sample_path,
                dri_crop=dri_crop_resized.astype(np.uint8),
                out_crop=out_crop.astype(np.uint8),
                motion_dict=motion_dict,
                eye_ratio=eye_ratio,
                lip_ratio=lip_ratio,
                x_d_i_new=x_d_i_new.astype(np.float32),
            )

            global_idx += 1
            frame_idx += 1

        vcap.release()


if __name__ == "__main__":
    main()

import argparse
from itertools import groupby

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from model import TrackNet
from utils import extract_pos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def load_video(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def get_scale(orig_w, orig_h, model_w, model_h):
    return orig_w / (model_w * 2), orig_h / (model_h * 2)


def predict_trajectory(frames, model, device, orig_w, orig_h, width, height):
    dists = [-1] * 2
    traj = [(None, None)] * 2
    scale_x, scale_y = get_scale(orig_w, orig_h, width, height)

    for i in tqdm(range(2, len(frames))):
        cur = cv2.resize(frames[i], (width, height))
        prev = cv2.resize(frames[i - 1], (width, height))
        prev2 = cv2.resize(frames[i - 2], (width, height))

        inp = np.concatenate((cur, prev, prev2), axis=2)
        inp = inp.astype(np.float32) / 255.0
        inp = np.moveaxis(inp, 2, 0)
        inp = np.expand_dims(inp, 0)

        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device))
        pred = out.argmax(dim=1).cpu().numpy()
        x, y = extract_pos(pred)

        if x is not None and y is not None:
            x *= scale_x
            y *= scale_y

        traj.append((x, y))

        if traj[-1][0] and traj[-2][0]:
            dist = distance.euclidean(traj[-1], traj[-2])
        else:
            dist = -1
        dists.append(dist)

    return traj, dists


def remove_outliers(traj, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if dists[i + 1] > max_dist or dists[i + 1] == -1:
            traj[i] = (None, None)
            outliers.remove(i)
        elif dists[i - 1] == -1:
            traj[i - 1] = (None, None)
    return traj


def segment_trajectory(traj, max_gap=4, max_gap_dist=80, min_len=5):
    flags = [0 if x[0] else 1 for x in traj]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(flags)]

    pos = 0
    start = 0
    subtracks = []

    for i, (k, length) in enumerate(groups):
        if k == 1 and 0 < i < len(groups) - 1:
            dist = distance.euclidean(traj[pos - 1], traj[pos + length])
            if length >= max_gap or dist / length > max_gap_dist:
                if pos - start > min_len:
                    subtracks.append([start, pos])
                    start = pos + length - 1
        pos += length

    if len(flags) - start > min_len:
        subtracks.append([start, len(flags)])

    return subtracks


def interpolate_gaps(coords):
    def nan_helper(vals):
        return np.isnan(vals), lambda z: z.nonzero()[0]

    x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

    nans, idx_fn = nan_helper(x)
    x[nans] = np.interp(idx_fn(nans), idx_fn(~nans), x[~nans])

    nans, idx_fn = nan_helper(y)
    y[nans] = np.interp(idx_fn(nans), idx_fn(~nans), y[~nans])

    return list(zip(x, y))


def save_video(frames, traj, output, fps, trace_len=7):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in range(len(frames)):
        frame = frames[i].copy()
        for j in range(trace_len):
            if i - j > 0 and traj[i - j][0]:
                x = int(traj[i - j][0])
                y = int(traj[i - j][1])
                frame = cv2.circle(frame, (x, y), 0, (0, 0, 255), 10 - j)
            else:
                break
        writer.write(frame)

    writer.release()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()

    frames, fps = load_video(args.input)
    orig_w, orig_h = frames[0].shape[1], frames[0].shape[0]

    traj, dists = predict_trajectory(
        frames, model, device, orig_w, orig_h, args.width, args.height
    )

    traj = remove_outliers(traj, dists)

    if args.interpolate:
        subtracks = segment_trajectory(traj)
        for start, end in subtracks:
            subtrack = traj[start:end]
            interpolated = interpolate_gaps(subtrack)
            traj[start:end] = interpolated

    save_video(frames, traj, args.output, fps)
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance
from tqdm import tqdm


def train_epoch(model, loader, optimizer, device, epoch, max_steps=200):
    losses = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", total=min(max_steps, len(loader)))

    for i, batch in enumerate(pbar):
        model.train()
        optimizer.zero_grad()

        out = model(batch[0].float().to(device))
        target = batch[1].long().to(device)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        losses.append(loss.item())

        if i >= max_steps - 1:
            break

    return np.mean(losses)


def validate(model, loader, device, epoch, dist_thresh=5):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]

    criterion = nn.CrossEntropyLoss()
    model.eval()

    pbar = tqdm(loader, desc=f"Val {epoch}")

    for batch in pbar:
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            target = batch[1].long().to(device)
            loss = criterion(out, target)
            losses.append(loss.item())

            pred = out.argmax(dim=1).cpu().numpy()

            for i in range(len(pred)):
                x_pred, y_pred = extract_pos(pred[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]

                if x_pred:
                    if vis != 0:
                        dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dist < dist_thresh:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                else:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1

        pbar.set_postfix(
            loss=f"{np.mean(losses):.4f}",
            tp=sum(tp),
            tn=sum(tn),
            fp=sum(fp),
            fn=sum(fn),
        )

    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)

    total_vis = sum([tp[i] + fp[i] + tn[i] + fn[i] for i in [1, 2, 3]])
    recall = sum(tp) / (total_vis + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return np.mean(losses), precision, recall, f1


def extract_pos(heatmap, scale=2):
    heatmap = (heatmap * 255).reshape(360, 640).astype(np.uint8)
    _, binary = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )

    x, y = None, None
    if circles is not None and len(circles) == 1:
        x = circles[0][0][0] * scale
        y = circles[0][0][1] * scale

    return x, y

"""Sanity check: recompute MOT gt visibility from GT boxes alone using
depth order by box bottom edge (larger y2 = closer) + image-border cropping.
Occluder set: all annotated boxes in the frame (any class, any flag) OR pedestrians only.
Reports MAE and Spearman vs gt col 9 for class-1, flag-1 boxes."""
import numpy as np, sys, configparser, os
from scipy.stats import spearmanr

def vis_frame(boxes, W, H, cand_mask, occl_mask):
    # boxes: N x 4 (x,y,w,h) 1-based
    out = np.full(len(boxes), np.nan)
    x1 = boxes[:,0]-1; y1 = boxes[:,1]-1; x2 = x1+boxes[:,2]; y2 = y1+boxes[:,3]
    for i in np.where(cand_mask)[0]:
        bx1, by1, bx2, by2 = x1[i], y1[i], x2[i], y2[i]
        w = int(round(bx2-bx1)); h = int(round(by2-by1))
        if w <= 0 or h <= 0: out[i] = 0; continue
        m = np.zeros((h, w), bool)
        # image-border cropping: pixels outside image count as invisible
        xs = np.arange(w) + bx1; ys = np.arange(h) + by1
        inside = (ys[:,None] >= 0) & (ys[:,None] < H) & (xs[None,:] >= 0) & (xs[None,:] < W)
        m |= ~inside
        for j in np.where(occl_mask)[0]:
            if j == i or y2[j] <= y2[i]:  # occluder must be closer (bottom edge lower in image)
                continue
            ox1 = int(np.clip(round(x1[j]-bx1), 0, w)); ox2 = int(np.clip(round(x2[j]-bx1), 0, w))
            oy1 = int(np.clip(round(y1[j]-by1), 0, h)); oy2 = int(np.clip(round(y2[j]-by1), 0, h))
            if ox2 > ox1 and oy2 > oy1: m[oy1:oy2, ox1:ox2] = True
        out[i] = 1 - m.mean()
    return out

def run(seqdir, max_frames=None):
    ini = configparser.ConfigParser(); ini.read(os.path.join(seqdir, 'seqinfo.ini'))
    W = int(ini['Sequence']['imWidth']); H = int(ini['Sequence']['imHeight'])
    a = np.loadtxt(os.path.join(seqdir, 'gt/gt.txt'), delimiter=',')
    res = {}
    for name in ['all_classes', 'ped_only']:
        est, gt = [], []
        for f in np.unique(a[:,0])[:max_frames]:
            r = a[a[:,0]==f]
            cand = (r[:,7]==1) & (r[:,6]==1)
            occl = np.ones(len(r), bool) if name=='all_classes' else (r[:,7]==1)
            v = vis_frame(r[:,2:6], W, H, cand, occl)
            est.append(v[cand]); gt.append(r[cand,8])
        est = np.concatenate(est); gt = np.concatenate(gt)
        res[name] = (np.abs(est-gt).mean(), spearmanr(est, gt).correlation, len(gt))
    return res

if __name__ == '__main__':
    for s in sys.argv[2:]:
        print(s, run(s, int(sys.argv[1])))

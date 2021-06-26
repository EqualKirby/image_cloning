import numpy as np


def norm(a):
    return np.sqrt(np.sum(a * a, 1))


def calc_angles(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    cos_val = np.sum(a * b, 1) / norm(a) / norm(b)
    angle = np.arccos(np.clip(0, cos_val, 1))
    return angle


# ref: https://github.com/fafa1899/MVCImageBlend/blob/master/ImgViewer/qimageshowwidget.cpp
def mvc(src, dst, border_pts, inner_mask, offset):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    nz = np.argwhere(inner_mask > 0)
    L = np.zeros((len(nz), len(border_pts) - 1))

    for i, (r, c) in enumerate(nz):
        a = border_pts[0:-1, :]
        b = border_pts[1:, :]
        cur = np.array([c, r])

        angles = calc_angles(a - cur, b - cur)
        tan_val = np.tan(angles / 2)

        ta = np.hstack((tan_val[-1], tan_val[0:-1]))
        tb = tan_val
        w = (ta + tb) / norm(a - cur)
        w = w / np.sum(w)
        L[i, :] = w

    dx, dy = offset
    xs, ys = border_pts[:-1, 0], border_pts[:-1, 1]
    diff = dst[ys + dy, xs + dx, :] - src[ys, xs, :]

    val = L @ diff
    ys, xs = nz[:, 0], nz[:, 1]
    dst[ys + dy, xs + dx, :] = src[ys, xs, :] + val

    dst = np.clip(0, dst, 255)
    return dst.astype(np.uint8)

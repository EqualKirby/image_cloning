import numpy as np
import cv2
import triangle as tr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def norm(a):
    return np.sqrt(np.sum(a * a, 1))


def calc_angles(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos_val = np.sum(a * b, 1) / norm(a) / norm(b)
    angle = np.arccos(np.clip(0, cos_val, 1))
    return angle


def in_range(a, a_min, a_max):
    return np.logical_and.reduce((a_min[0] <= a[:, 0], a[:, 0] < a_max[0], a_min[1] <= a[:, 1], a[:, 1] < a_max[1]))


# ref: https://github.com/fafa1899/MVCImageBlend/blob/master/ImgViewer/qimageshowwidget.cpp
def mvc(src, dst, mask, offset, L=None, get_L=False):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    border_pts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    inner_mask = mask.copy()
    inner_mask[border_pts[:, 1], border_pts[:, 0]] = 0

    # calculate weight matrix
    nz = np.argwhere(inner_mask > 0)

    if L is None:
        L = np.zeros((len(nz), len(border_pts) - 1))

        a = border_pts[0:-1, :]
        b = border_pts[1:, :]

        for i, (r, c) in enumerate(nz):
            cur = np.array([c, r])
            angles = calc_angles(a - cur, b - cur)
            tan_val = np.tan(angles / 2)

            ta = np.hstack((tan_val[-1], tan_val[0:-1]))
            tb = tan_val
            w = (ta + tb) / norm(a - cur)
            w = w / np.sum(w)
            L[i, :] = w

    dx, dy = offset

    # calculate boundary difference
    border_idx = in_range(border_pts[:-1] + offset, (0, 0), (dst.shape[1], dst.shape[0]))
    x, y = border_pts[:-1, 0][border_idx], border_pts[:-1, 1][border_idx]
    diff = dst[y + dy, x + dx, :] - src[y, x, :]

    # calculate final result
    interior_idx = in_range(nz + [dy, dx], (0, 0), (dst.shape[0], dst.shape[1]))
    y, x = nz[:, 0][interior_idx], nz[:, 1][interior_idx]

    M = L[interior_idx, :][:, border_idx]
    M = M / np.sum(M, axis=1).reshape(-1, 1)
    dst[y + dy, x + dx, :] = src[y, x, :] + M @ diff

    dst = np.clip(0, dst, 255)

    if get_L:
        return dst.astype(np.uint8), L
    else:
        return dst.astype(np.uint8)


def mvc_mesh(src, dst, mask, offset, L=None, get_L=False):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    border_pts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    inner_mask = mask
    inner_mask[border_pts[:, 1], border_pts[:, 0]] = 0

    inner_border_pts = cv2.findContours(inner_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    mesh = generate_mesh(inner_border_pts, np.ones_like(src))
    vertices = mesh['vertices']

    # calculate weight matrix
    if L is None:
        L = np.zeros((len(vertices), len(border_pts) - 1))

        a = border_pts[0:-1, :]
        b = border_pts[1:, :]

        for i, (x, y) in enumerate(vertices):
            cur = np.array([x, y])
            angles = calc_angles(a - cur, b - cur)
            tan_val = np.tan(angles / 2)

            with np.printoptions(threshold=np.inf):
                if (norm(a - cur) == 0).any():
                    print(a, cur)
                    exit()

            ta = np.hstack((tan_val[-1], tan_val[0:-1]))
            tb = tan_val
            w = (ta + tb) / norm(a - cur)
            w = w / np.sum(w)
            L[i, :] = w

    dx, dy = offset

    # calculate boundary difference
    border_idx = in_range(border_pts[:-1] + offset, (0, 0), (dst.shape[1], dst.shape[0]))
    x, y = border_pts[:-1, 0][border_idx], border_pts[:-1, 1][border_idx]
    diff = dst[y + dy, x + dx, :] - src[y, x, :]

    base = L @ diff
    print(L)

    triang = mtri.Triangulation(mesh['vertices'][:, 1], mesh['vertices'][:, 0], triangles=mesh['triangles'])
    interp0 = mtri.LinearTriInterpolator(triang, base[:, 0])
    interp1 = mtri.LinearTriInterpolator(triang, base[:, 1])
    interp2 = mtri.LinearTriInterpolator(triang, base[:, 2])
    plt.triplot(triang)
    plt.show()

    nz = np.argwhere(inner_mask > 0)
    for r, c in nz:
        if 0 <= r + dy and r + dy < dst.shape[0] and 0 <= c + dx and c + dx < dst.shape[1]:
            a0 = float(interp0(r, c)) or 0
            a1 = float(interp1(r, c)) or 0
            a2 = float(interp2(r, c)) or 0
            dst[r + dy, c + dx] = src[r, c] + np.array([a0, a1, a2])

    dst = np.clip(0, dst, 255)

    if get_L:
        return dst.astype(np.uint8), 0
    else:
        return dst.astype(np.uint8)


def generate_mesh(pts, wireframe=None):
    poly = {}
    poly['vertices'] = pts
    poly['segments'] = [[i, i + 1] for i in range(len(pts) - 1)] + [[len(pts) - 1, 0]]

    res = tr.triangulate(poly, 'pqD')

    if wireframe is not None:
        idx = res['triangles'].flatten()
        pos = res['vertices'][idx]
        pos = np.round(pos).astype(np.int)
        tri = pos.reshape(len(idx) // 3, -1, 1, 2)
        cv2.polylines(wireframe, tri, True, (0, 0, 0))
        cv2.imshow('test', wireframe * 255)
        cv2.waitKey(0)

    return res

import numpy as np
import cv2
import triangle as tr
import matplotlib.pyplot as plt


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
def mvc(src, dst, mask, offset):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    border_pts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)    
    inner_mask = mask
    inner_mask[border_pts[:, 1], border_pts[:, 0]] = 0

    # calculate weight matrix
    nz = np.argwhere(inner_mask > 0)
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
    val = L[interior_idx, :][:, border_idx] @ diff
    dst[y + dy, x + dx, :] = src[y, x, :] + val

    dst = np.clip(0, dst, 255)
    return dst.astype(np.uint8)


def mvc_mesh(src, dst, mask, offset):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    border_pts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)    
    inner_mask = mask
    inner_mask[border_pts[:, 1], border_pts[:, 0]] = 0

    mesh = generate_mesh(border_pts, np.ones_like(src))

    dst = np.clip(0, dst, 255)
    return dst.astype(np.uint8)    


def generate_mesh(pts, wireframe=None):
    poly = {}
    poly['vertices'] = pts[:-1]
    poly['segments'] = [[i, i + 1] for i in range(len(pts) - 2)] + [[len(pts) - 2, 0]]

    res = tr.triangulate(poly, 'pqD')
    # del res['segment_markers']
    # print(res)

    if wireframe is not None:
        idx = res['triangles'].flatten()
        pos = res['vertices'][idx]
        pos = np.round(pos).astype(np.int)
        tri = pos.reshape(len(idx) // 3, -1, 1, 2)
        cv2.polylines(wireframe, tri, True, (0, 0, 0))
        cv2.imshow('test', wireframe * 255)
        cv2.waitKey(0)

    return res

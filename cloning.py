import numpy as np
import cv2
import triangle as tr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw


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
def mvc(src, dst, mask, offset, state=None, get_state=False):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    border_pts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    inner_mask = mask.copy()
    inner_mask[border_pts[:, 1], border_pts[:, 0]] = 0

    # calculate weight matrix
    nz = np.argwhere(inner_mask > 0)

    if state is None:
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
    else:
        L = state

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

    if get_state:
        return dst.astype(np.uint8), L
    else:
        return dst.astype(np.uint8)


def mvc_mesh(src, dst, mask, offset, state=None, get_state=False):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    border_pts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    inner_mask = mask.copy()
    inner_mask[border_pts[:, 1], border_pts[:, 0]] = 0

    # calculate weight matrix
    if state is None:
        inner_border_pts = cv2.findContours(inner_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
        mesh = generate_mesh(inner_border_pts, np.ones_like(src))
        vertices = mesh['vertices']

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
    else:
        L = state[0]
        mesh = state[1]

    dx, dy = offset

    # calculate boundary difference
    border_idx = in_range(border_pts[:-1] + offset, (0, 0), (dst.shape[1], dst.shape[0]))
    x, y = border_pts[:-1, 0][border_idx], border_pts[:-1, 1][border_idx]
    diff = dst[y + dy, x + dx, :] - src[y, x, :]

    base = L @ diff
    diff_map = interpolation(mesh, base, offset, dst.shape[1], dst.shape[0], base.min(), base.max() - base.min())

    nz = np.argwhere((diff_map != diff_map.max()).all(axis=2))
    for r, c in nz:
        if 0 <= r - dy and r - dy < src.shape[0] and 0 <= c - dx and c - dx < src.shape[1]:
            dst[r, c] = src[r - dy, c - dx] + diff_map[r, c]

    dst = np.clip(0, dst, 255)

    if get_state:
        return dst.astype(np.uint8), (L, mesh)
    else:
        return dst.astype(np.uint8)


def interpolation(mesh, base, offset, w, h, shift, scale):
    magic = 0.98

    glClearColor(1, 1, 1, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    vertices = mesh['vertices'] + offset + np.array([0.5, 0.5])
    base = (base - shift) / scale * magic

    glBegin(GL_TRIANGLES)
    for tri in mesh['triangles']:
        vs = vertices[tri]
        cs = base[tri]

        glColor3f(*cs[0])
        glVertex3f(*vs[0], 0)
        glColor3f(*cs[1])
        glVertex3f(*vs[1], 0)
        glColor3f(*cs[2])
        glVertex3f(*vs[2], 0)

    # glColor3f(1, 0, 0)
    # glVertex3f(0, 0, 0)
    # glColor3f(0, 1, 0)
    # glVertex3f(w, 0, 0)
    # glColor3f(0, 0, 1)
    # glVertex3f(0, h, 0)

    # glColor3f(1, 0, 0)
    # glVertex3f(w, 0, 0)
    # glColor3f(0, 1, 0)
    # glVertex3f(w, h, 0)
    # glColor3f(0, 0, 1)
    # glVertex3f(0, h, 0)
    glEnd()

    glReadBuffer(GL_COLOR_ATTACHMENT0)
    image_buffer = glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT)
    image = np.frombuffer(image_buffer, dtype=np.float32).reshape(h, w, 3)
    image = image * scale / magic + shift

    # cv2.imwrite('test.png', cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))
    # print(image * 255)

    return image


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

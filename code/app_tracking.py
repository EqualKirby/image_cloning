import numpy as np
import cv2
from skimage.draw import line
from objectTracking import *
from cloning import mvc, mvc_mesh, init_opengl
import glfw


def get_points_in_line(start, end):
    x, y = line(*start, *end)
    return x[1:], y[1:]


if __name__ == '__main__':
    wnd = 'Draw the region'
    skip_drawing = True

    src = cv2.imread('../image/source.jpg')
    h, w, c = src.shape

    if not skip_drawing:
        cv2.namedWindow(wnd)
        cv2.resizeWindow(wnd, w, h)
        cv2.imshow(wnd, src)

        canvas = np.zeros_like(src)
        pts = []
        drawing = False

        def draw_callback(event, x, y, flags, param):
            global drawing, canvas, pts

            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                canvas = np.zeros_like(src)
                pts = [(x, y)]
                cv2.imshow(wnd, (1 - canvas) * src)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    xs, ys = get_points_in_line(pts[-1], (x, y))
                    pts.extend(list(zip(xs, ys)))
                    canvas[ys, xs, :] = 1
                    cv2.imshow(wnd, (1 - canvas) * src)

            elif event == cv2.EVENT_LBUTTONUP:            
                drawing = False
                xs, ys = get_points_in_line((x, y), pts[0])
                pts.extend(list(zip(xs, ys)))

                canvas = np.zeros_like(src)
                cv2.fillPoly(canvas, [np.array(pts).reshape((-1, 1, 2))], (1, 1, 1))
                canvas[[p[1] for p in pts], [p[0] for p in pts], :] = 0
                cv2.imshow(wnd, canvas * src)

        cv2.setMouseCallback(wnd, draw_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        inner_mask = np.squeeze(canvas[:, :, 0])
        pts = np.array(pts)
        np.save('border', pts)
        np.save('inner', inner_mask)

    else:
        pts = np.load('border.npy')
        inner_mask = np.load('inner.npy')

    n_frame = 100
    video_path = "../video/easy.mp4"
    bboxs = objectTracking(video_path, n_frame, play_realtime=True, save_to_file=True)

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    
    ogl_wnd = init_opengl(w, h)
    L = None

    for frame_idx in range(n_frame):
        print("synthesis frame #", frame_idx)
        _, frames = cap.read()
        center = (int(np.mean(bboxs[frame_idx][:,0])), int(np.mean(bboxs[frame_idx][:,1])))
        offset = (center[0] - src.shape[1] // 2, center[1] - h // 2)
        if L is None:
            res, L = mvc_mesh(src, frames, inner_mask, offset, get_state=True)
        else:
            res = mvc_mesh(src, frames, inner_mask, offset, state=L)

        output_wnd = 'result'
        cv2.namedWindow(output_wnd)
        cv2.resizeWindow(output_wnd, w, h)
        cv2.imshow(output_wnd, res)
        cv2.waitKey(10)

    cap.release()
    glfw.destroy_window(ogl_wnd)
    glfw.terminate()

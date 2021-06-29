import numpy as np
import cv2
from skimage.draw import line
from cloning import mvc, mvc_mesh, init_opengl
import sys
import glfw


def get_points_in_line(start, end):
    x, y = line(*start, *end)
    return x[1:], y[1:]


if __name__ == '__main__':
    if len(sys.argv) == 3:
        src_name = sys.argv[1]
        dst_name = sys.argv[2]
    else:
        src_name = '../image/source.jpg'
        dst_name = '../image/target.jpg'

    wnd = 'Draw the region'
    skip_drawing = False

    src = cv2.imread(src_name)
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
        # np.save('border', pts)
        # np.save('inner', inner_mask)

    else:
        pts = np.load('border.npy')
        inner_mask = np.load('inner.npy')

    dst = cv2.imread(dst_name)
    h, w, c = dst.shape
    ogl_wnd = init_opengl(w, h)
    res, state = mvc_mesh(src, dst, inner_mask, (456 + 300 * 0, 326), get_state=True)

    output_wnd = 'result'    
    cv2.namedWindow(output_wnd)
    cv2.resizeWindow(output_wnd, w, h)
    cv2.imshow(output_wnd, res)

    def output_callback(event, x, y, flags, param):
        global state

        x = np.clip(x - src.shape[1] // 2, 0, w - 1)
        y = np.clip(y - src.shape[0] // 2, 0, h - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            res = mvc_mesh(src, dst, inner_mask, (x, y), state)
            cv2.imshow(output_wnd, res)

    cv2.setMouseCallback(output_wnd, output_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    glfw.destroy_window(ogl_wnd)
    glfw.terminate()

import numpy as np
import cv2
from skimage.draw import line
from cloning import *
from objectTracking import *

def get_points_in_line(start, end):
    x, y = line(*start, *end)
    return x[1:], y[1:]

if __name__ == '__main__':
    wnd = 'Draw the region'
    skip_drawing = False

    src = cv2.imread('../image/source.jpg')
    h, w, c = src.shape

    if not skip_drawing:
        cv2.namedWindow(wnd)
        cv2.resizeWindow(wnd, w, h)
        cv2.imshow(wnd, src)

        canvas = np.zeros_like(src)
        pts = []
        drawing = False

        def callback(event, x, y, flags, param):
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

                print(len(pts))
                print(canvas.sum().sum().sum() / 3)

        cv2.setMouseCallback(wnd, callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        inner_mask = np.squeeze(canvas[:, :, 0])
        pts = np.array(pts)
        np.save('border', pts)
        np.save('inner', inner_mask)

    else:
        pts = np.load('border.npy')
        inner_mask = np.load('inner.npy')

    # FIXME: out of boundary 
    # dst = cv2.imread('../image/target.jpg')
    # h, w, c = dst.shape
    inner_mask = np.dstack([inner_mask]*3) * 255
    # res = mvc(src, dst, pts, inner_mask, (456, 326))

    n_frame = 100
    video_path = "../video/easy.mp4"
    bboxs = objectTracking(video_path, n_frame, play_realtime=True, save_to_file=True)

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    for frame_idx in range(n_frame):
        print("synthesis frame #", frame_idx)
        _, frames = cap.read()
        center = (int(np.mean(bboxs[frame_idx][:,0])), int(np.mean(bboxs[frame_idx][:,1])))
        res = cv2.seamlessClone(src, frames, inner_mask, center, cv2.NORMAL_CLONE)

        output_wnd = 'result'
        cv2.namedWindow(output_wnd)
        cv2.resizeWindow(output_wnd, w, h)
        cv2.imshow(output_wnd, res)
        cv2.waitKey(10)

    cap.release()
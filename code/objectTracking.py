from estimateAllTranslation import *
import numpy as np
import cv2

def objectTracking(video_path, n_frame, play_realtime=False, save_to_file=False):
    # initilize
    rawVideo = cv2.VideoCapture(video_path)
    
    frames = np.empty((n_frame,),dtype=np.ndarray)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = rawVideo.read()

    # draw rectangle roi for target objects, or use default objects initilization
    bboxs[0] = np.empty((4,2), dtype=float)
    (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Area",frames[0])
    cv2.destroyWindow("Select Area")
    bboxs[0] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    
    if save_to_file:
        out = cv2.VideoWriter('output.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(frames[0].shape[1],frames[0].shape[0]))
    
    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY),bboxs[0],use_shi=False)
    for i in range(1,n_frame):
        print('Processing Frame',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])

        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_RGB2GRAY),bboxs[i])

        # draw bounding box and visualize feature point for each object
        frames_draw[i] = frames[i].copy()
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i].astype(int))
        frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
        for k in range(startXs.shape[0]):
            frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k]),int(startYs[k])),3,(0,0,255),thickness=2)
        
        # imshow if to play the result in real time
        if play_realtime:
            cv2.imshow("win",frames_draw[i])
            cv2.waitKey(10)
        if save_to_file:
            out.write(frames_draw[i])
    
    if save_to_file:
        out.release()
    
    rawVideo.release()

    return bboxs

if __name__ == "__main__":
    video_path = "../video/hard.mp4"
    bboxs = objectTracking(video_path, 100, play_realtime=True, save_to_file=True)
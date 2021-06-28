from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from skimage import transform as tf
from numpy.linalg import inv
from utils import interp2
from scipy import signal
import numpy as np
import cv2

def getFeatures(img,bbox,use_shi=False):
    (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox.astype(int))
    roi = img[ymin:ymin+boxh,xmin:xmin+boxw]
    if use_shi:
        corner_response = corner_shi_tomasi(roi)
    else:
        corner_response = corner_harris(roi)
    coordinates = peak_local_max(corner_response,num_peaks=20,exclude_border=2)
    coordinates[:,1] += xmin
    coordinates[:,0] += ymin

    x = coordinates[:,1].reshape(-1,1)
    y = coordinates[:,0].reshape(-1,1)
    return x,y

def estimateAllTranslation(startXs,startYs,img1,img2):
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I,(5,5),0.2)
    Iy, Ix = np.gradient(I.astype(float))

    startXs_flat = startXs.flatten()
    startYs_flat = startYs.flatten()
    newXs = np.full(startXs_flat.shape,-1,dtype=float)
    newYs = np.full(startYs_flat.shape,-1,dtype=float)
    for i in range(np.size(startXs)):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2)
    newXs = np.reshape(newXs, startXs.shape)
    newYs = np.reshape(newYs, startYs.shape)
    return newXs, newYs

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2, WINDOW_SIZE = 25):
    X=startX
    Y=startY
    mesh_x,mesh_y=np.meshgrid(np.arange(WINDOW_SIZE),np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mesh_x_flat_fix =mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix =mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix,mesh_y_flat_fix))
    I1_value = interp2(img1_gray, coor_fix[[0],:], coor_fix[[1],:])
    Ix_value = interp2(Ix, coor_fix[[0],:], coor_fix[[1],:])
    Iy_value = interp2(Iy, coor_fix[[0],:], coor_fix[[1],:])
    I=np.vstack((Ix_value,Iy_value))
    A=I.dot(I.T)

    for _ in range(15):
        mesh_x_flat=mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat=mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor=np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip=(I2_value-I1_value).reshape((-1,1))
        b=-I.dot(Ip)
        solution=inv(A).dot(b)
        X += solution[0,0]
        Y += solution[1,0]
    
    return X, Y

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    newbbox = np.zeros_like(bbox)
    Xs = newXs.copy()
    Ys = newYs.copy()
    
    desired_points = np.hstack((startXs,startYs))
    actual_points = np.hstack((newXs,newYs))
    t = tf.SimilarityTransform()
    t.estimate(dst=actual_points, src=desired_points)
    mat = t.params

    # estimate the new bounding box with all the feature points
    # coords = np.vstack((bbox[obj_idx,:,:].T,np.array([1,1,1,1])))
    # new_coords = mat.dot(coords)
    # newbbox[obj_idx,:,:] = new_coords[0:2,:].T

    # estimate the new bounding box with only the inliners (Added by Yongyi Wang)
    THRES = 1
    projected = mat.dot(np.vstack((desired_points.T.astype(float),np.ones([1,np.shape(desired_points)[0]]))))
    distance = np.square(projected[0:2,:].T - actual_points).sum(axis = 1)
    actual_inliers = actual_points[distance < THRES]
    desired_inliers = desired_points[distance < THRES]
    if np.shape(desired_inliers)[0]<4:
        print('too few points')
        actual_inliers = actual_points
        desired_inliers = desired_points
    t.estimate(dst=actual_inliers, src=desired_inliers)
    mat = t.params
    coords = np.concatenate((bbox, np.ones((4,1))), axis=1).T
    new_coords = mat.dot(coords)
    newbbox = new_coords[0:2,:].T
    Xs[distance >= THRES] = -1
    Ys[distance >= THRES] = -1

    return Xs, Ys, newbbox

if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    ret, frame1 = cap.read()  # get first frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    ret, frame2 = cap.read()  # get second frame
    frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)

    n_object = 1
    bbox = np.array([[[291,187],[405,187],[291,267],[405,267]]])

    startXs,startYs = getFeatures(frame1_gray,bbox)
    newXs, newYs =  estimateAllTranslation(startXs, startYs, frame1, frame2)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    diff = np.subtract(frame1_gray.astype(int),frame2_gray.astype(int))
    ax.imshow(diff,cmap='gray')
    ax.scatter(newXs[newXs!=-1],newYs[newYs!=-1],color=(0,1,0))
    ax.scatter(startXs[startXs!=-1],startYs[startYs!=-1],color=(1,0,0))
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:])
        patch = Rectangle((xmin,ymin),boxw,boxh,fill=False,color=(0,0,0),linewidth=1)
        ax.add_patch(patch)
    plt.show()
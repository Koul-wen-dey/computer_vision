import cv2
import numpy as np

def perspective(video:str,image:str):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.imread(image)
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters =  cv2.aruco.DetectorParameters_create()
        tmp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(tmp, dictionary, parameters=parameters)
        
        if len(markerIds) < 4:
            continue
        
        pst_src = np.array(
            [[0,0],
            [img.shape[1]-1,0],
            [img.shape[1]-1,img.shape[0]-1],
            [0,img.shape[0]-1]],dtype=float)
        markerCorners = np.asarray(markerCorners[:4]).reshape(16,2)
        points = np.zeros((4,2),dtype=np.float32)
        s = markerCorners.sum(axis=1)
        points[0] = markerCorners[np.argmin(s)]
        points[2] = markerCorners[np.argmax(s)]
        d = np.diff(markerCorners,axis=1)
        points[1] = markerCorners[np.argmin(d)]
        points[3] = markerCorners[np.argmax(d)]
        h, status = cv2.findHomography(pst_src,points)
        frame2 = frame.copy()
        tmp = cv2.warpPerspective(img,h,(frame.shape[1],frame.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
        cv2.fillConvexPoly(frame,points.astype(int),(0,0,0))
        all = cv2.add(frame,tmp)
        all = cv2.hconcat([frame2,all])
        all = cv2.resize(all,(1280,480))
        cv2.imshow('a',all)
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()
    cap.release()
    pass

if __name__ == '__main__':
    perspective('./Dataset_CvDl_Hw2/Q3_Image/video.mp4','./Dataset_CvDl_Hw2/Q3_Image/logo.png')
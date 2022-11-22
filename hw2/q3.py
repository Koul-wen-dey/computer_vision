import cv2
import numpy as np

def perspective(video:str,image:str):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return
        img = cv2.imread(image)
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters_create()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        pst_src = np.array(
            [[img.shape[1]-1,img.shape[0]-1],
            [img.shape[1]-1,0],
            [0,0],
            [0,img.shape[0]-1]],dtype=float)
        
        points = [
            (markerCorners[0][0,2,0],markerCorners[0][0,2,1]),#4
            (markerCorners[1][0,1,0],markerCorners[1][0,1,1]),#2
            (markerCorners[2][0,0,0],markerCorners[2][0,0,1]),#1
            (markerCorners[3][0,3,0],markerCorners[3][0,3,1]) #3
        ]
        points = np.asarray(points)
        h, status = cv2.findHomography(pst_src,points)
        tmp = cv2.warpPerspective(img,h,(frame.shape[1],frame.shape[0]))
        cv2.fillConvexPoly(frame,points.astype(np.int32),0)
        all = cv2.add(frame,tmp)
        cv2.imshow('a',all)
        cv2.waitKey(1000)
        
    cv2.destroyAllWindows()
    cap.release()
    pass

if __name__ == '__main__':
    perspective('./Dataset_CvDl_Hw2/Q3_Image/video.mp4','./Dataset_CvDl_Hw2/Q3_Image/logo.png')
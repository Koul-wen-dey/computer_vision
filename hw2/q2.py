import cv2
import numpy as np

def preprocessing(video:str):
    cap = cv2.VideoCapture(video)

    ret, frame = cap.read()
    if not ret:
        return 
    cap.release()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 35
    params.maxArea = 90
    params.minInertiaRatio = 0.46

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame)
    for k in keypoints:
        y1, x1 = int(k.pt[1]-6),int(k.pt[0]-6)
        y2, x2 = int(k.pt[1]+6),int(k.pt[0]+6)
        frame = cv2.rectangle(frame,(x1,y1),(x2, y2),(0,0,255))
        frame = cv2.line(frame,(x1+6,y1),(x1+6,y2),(0,0,255))
        frame = cv2.line(frame,(x1,y1+6),(x2,y1+6),(0,0,255))
    # img = cv2.drawKeypoints(frame,keypoints,np.array([]),(255,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # for k in keypoints:
    
    # img = cv2.rectangle(frame,keypoints,np.array([]),(0,0,255))
    cv2.imshow('1',frame)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

def tracking(video:str):

    pass

if __name__ == '__main__':
    preprocessing('./Dataset_CvDl_Hw2/Q2_Image/optical_flow.mp4')
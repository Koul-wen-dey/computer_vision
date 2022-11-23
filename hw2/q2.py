import cv2
import numpy as np



def preprocessing(video:str):
    global keypoints
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
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def tracking(video:str):
    global keypoints
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    params = {'winSize':(10,10),'maxLevel':2}
    flag = True
    coordinate = np.array([k.pt for k in keypoints]).astype(np.float32)
    coor_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not flag:
            coordinate, state, error = cv2.calcOpticalFlowPyrLK(gray_past,gray,coor_past,None,**params)
        gray_past = gray.copy()
        coor_past = coordinate
        coor_list.append(coordinate)
        curve_array = np.asarray(coor_list)
        
        for i in range(7):
            cv2.circle(frame,np.int32(coordinate[i]),4,(0,255,255))
            cv2.polylines(frame,np.int32([curve_array[:,i,:]]),False,(0,255,255),2)
        flag = False
        cv2.imshow('a',frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tracking('./Dataset_CvDl_Hw2/Q2_Image/optical_flow.mp4')
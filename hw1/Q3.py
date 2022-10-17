import cv2
import numpy as np

def click_event(event,x,y,flags, params):
    global image_right,depth
    if event == cv2.EVENT_LBUTTONDOWN:
        if depth[y,x] <= 0:
            return
        img = image_right.copy()
        cv2.circle(img,(x-(depth[y,x]), y), 10,(0,255,0),-1)
        cv2.imshow('ImageR', img)
    
def stereo_disparity_map(imgL:str='',imgR:str=''):
    global image_right,depth

    image_left = cv2.imread(imgL,cv2.IMREAD_COLOR)
    image_right = cv2.imread(imgR,cv2.IMREAD_COLOR)
    gray_left = cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
    left_match = cv2.StereoBM_create(256,25)
    disp = left_match.compute(gray_left,gray_right)
    depth = np.clip(disp/16.0,0,255).astype(np.uint8)
    

    cv2.namedWindow('ImageL',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ImageL',720,480)
    cv2.setMouseCallback('ImageL',click_event)
    cv2.namedWindow('ImageR',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ImageR',720,480)
    cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disparity',720,480)
    cv2.imshow('ImageL',image_left)
    cv2.imshow('ImageR',image_right)
    cv2.imshow('disparity',depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


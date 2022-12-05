import cv2
import numpy as np
def background_subtract(video:str):
    cap = cv2.VideoCapture(video)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if count == 24:
            break
        count += 1

    frames = np.asarray(frames,dtype=np.uint8)
    mean = np.mean(frames[:25,:,:],axis=0)
    dev = np.std(frames[:25,:,:],axis=0)
    dev[np.where(dev<5)] = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmp =np.zeros(gray.shape,dtype=np.uint8)
        tmp = np.where(np.abs(gray-mean)>dev*5,255,0).astype(np.uint8)
        frame[np.where(tmp==0)] = 0
        tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
        all = cv2.hconcat([frame2,tmp,frame])
        cv2.imshow('Q1',all)
        cv2.waitKey(20)
    cv2.destroyAllWindows()
    cap.release()

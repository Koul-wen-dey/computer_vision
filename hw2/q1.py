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
    cap.release()

    frames = np.asarray(frames,dtype=np.uint8)
    dev = np.zeros((frames.shape[1],frames.shape[2]))
    mean = np.zeros((frames.shape[1],frames.shape[2]))
    for y in range(frames.shape[1]):
        for x in range(frames.shape[2]):
            dev[y,x] = np.std(frames[:,y,x])
            mean[y,x] = np.mean(frames[:,y,x])
    dev[np.where(dev<5)] = 5

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmp =np.zeros(gray.shape,dtype=np.uint8)
        tmp = np.where(gray-mean>dev*5,255,0).astype(np.uint8)
        frame[np.where(tmp==0)] = 0
        tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
        all = cv2.hconcat([frame2,tmp,frame])
        cv2.imshow('a',all)
        cv2.waitKey(5)
    cv2.destroyAllWindows()
    cap.release()

import cv2
import numpy as np

def on_board(folder:str,text:str):
    if len(text) > 6:
        return
    
    width = 11
    height = 8
    cordinate_bias = [62,59,56,29,26,23]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objects = np.zeros((width*height, 3), np.float32)
    objects[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    object_points = []
    image_points = []
    folder += '/'
    ch = []
    fs = cv2.FileStorage(folder+'Q2_lib/alphabet_lib_onboard.txt',cv2.FILE_STORAGE_READ)

    for t in range(len(text)):
        word = fs.getNode(text[t]).mat()
        for c in range(len(word)):
            for i in range(len(word[c])):
                tmp = cordinate_bias[t] + word[c][i][0] + word[c][i][1] * width
                ch.append(tmp)
    
    for i in range(1,6):
        img = cv2.imread(folder+str(i)+'.bmp', cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray_img, (width, height))
        if ret:
            corner = cv2.cornerSubPix(gray_img, corner, (11, 11), (-1, -1), criteria)
            object_points.append(objects)
            image_points.append(corner)
        ret, inmtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_img.shape[::-1], None, None)
    
        for j in range(0,len(ch),2):
            x1, y1 = corner[ch[j]][0]
            x2, y2 = corner[ch[j+1]][0]
            cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),10)
        cv2.namedWindow('2.1',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('2.1',1280,960)
        cv2.imshow('2.1',img)
        cv2.waitKey(1200)
    cv2.destroyAllWindows()
    # exmtx = []
    # for i in range(5):
    #     rmtx, _ = cv2.Rodrigues(rvecs[i])
    #     tmp = np.append(rmtx, tvecs[i], axis=1)
    #     exmtx.append(tmp)
    # cv2.projectPoints([i], rvecs[i], tvecs[i], inmtx, dist)
    

def vertical(folder:str,text:str):
    if len(text) > 6:
        return
    
    width = 11
    height = 8
    cordinate_bias = [62,59,56,29,26,23]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objects = np.zeros((width*height, 3), np.float32)
    objects[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    folder += '/'
    fs = cv2.FileStorage(folder+'Q2_lib/alphabet_lib_vertical.txt',cv2.FILE_STORAGE_READ)
    
    for i in range(1,6):
        object_points = []
        image_points = []
        projected_points = []
        ch = []
        img = cv2.imread(folder+str(i)+'.bmp', cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray_img, (width, height))
        corner = cv2.cornerSubPix(gray_img, corner, (11, 11), (-1, -1), criteria)
        object_points.append(objects)
        image_points.append(corner)
        ret, inmtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_img.shape[::-1], None, None)
    
        for t in range(len(text)):
            word = fs.getNode(text[t]).mat()
            for c in range(len(word)):
                for j in range(len(word[c])):
                    tmp = object_points[0][cordinate_bias[t]] + word[c][j]
                    ch.append(tmp)

        for j in range(len(ch)):
            p, _=cv2.projectPoints(ch[j],np.float32(rvecs),np.float32(tvecs),inmtx,dist)
            projected_points.append(p)

        for j in range(0,len(ch),2):
            x1, y1 = projected_points[j][0][0]
            x2, y2 = projected_points[j+1][0][0]
            cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),10)
        cv2.namedWindow('2.2',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('2.2',1280,960)
        cv2.imshow('2.2',img)
        cv2.waitKey(1200)
    cv2.destroyAllWindows()

    
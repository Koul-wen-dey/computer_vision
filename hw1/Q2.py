import cv2
import numpy as np

def on_board(folder:str,text:str):
    if len(text) > 6:
        return
    '''
    folder += '/'
    img = cv2.imread(folder+'1.bmp', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1280, 960))
    tmp = cv2.findChessboardCorners(img, (11, 8))

    c1 = [7,5,0]
    c2 = [4,5,0]
    c3 = [1,5,0]
    c4 = [7,2,0]
    c5 = [4,2,0]
    c6 = [1,2,0]

    '''
    cordinate_bias = np.array(
        [
            [7,5,0],
            [4,5,0],
            [1,5,0],
            [7,2,0],
            [4,2,0],
            [1,2,0]
        ]
    )
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ob = np.zeros((11*8, 3), np.float32)
    ob[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    object_points = []
    image_points = []
    folder += '/'
    ch = []
    fs = cv2.FileStorage(folder+'Q2_lib/alphabet_lib_onboard.txt',cv2.FILE_STORAGE_READ)
    for t in text:
        ch.append(fs.getNode(t).mat())
    for c in ch:
        print(c)
    for i in range(1,6):
        img = cv2.imread(folder+str(i)+'.bmp', cv2.IMREAD_GRAYSCALE)
        ret, corner = cv2.findChessboardCorners(img, (11, 8))
        if ret:
            corner = cv2.cornerSubPix(img, corner, (11, 11), (-1, -1), criteria)
            object_points.append(ob)
            image_points.append(corner)

    ret, inmtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img.shape[::-1], None, None)
    
    exmtx = []
    for i in range(5):
        rmtx, _ = cv2.Rodrigues(rvecs[i])
        tmp = np.append(rmtx, tvecs[i], axis=1)
        exmtx.append(tmp)
    cv2.projectPoints([i], rvecs[i], tvecs[i], inmtx, dist)
    

def vertical(folder:str,text:str):
    if len(text) > 6:
        return
    folder += '/'
    pass
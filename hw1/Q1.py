import cv2
import numpy as np

global dist, inmtx, rvecs, tvecs


def find_corner(folder: str):
    folder += '/'
    for i in range(1, 16):
        img = cv2.imread(folder+str(i)+'.bmp',
                         cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1280, 960))
        tmp = cv2.findChessboardCorners(img, (11, 8))
        cv2.drawChessboardCorners(img, (11, 8), tmp[1], True)
        cv2.imshow('1.1', img)
        cv2.waitKey(1200)
        cv2.destroyAllWindows()


def find_intrinsic(folder: str):
    global dist, inmtx, rvecs, tvecs
    folder += '/'
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ob = np.zeros((11*8, 3), np.float32)
    ob[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    object_points = []
    image_points = []
    for i in range(1, 16):
        img = cv2.imread(folder+str(i)+'.bmp',
                         cv2.IMREAD_GRAYSCALE)
        ret, corner = cv2.findChessboardCorners(img, (11, 8))
        if ret:
            corner = cv2.cornerSubPix(
                img, corner, (11, 11), (-1, -1), criteria)
            object_points.append(ob)
            image_points.append(corner)

    ret, inmtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img.shape[::-1], None, None)

    print('Intrinsic:')
    print(inmtx)


def find_extrinsic(folder: str, index: int):
    global rvecs, tvecs
    rmtx, _ = cv2.Rodrigues(rvecs[index-1])
    exmtx = np.append(rmtx, tvecs[index-1], axis=1)
    print('Extrinsic:')
    print(exmtx)


def find_distortion():
    global dist
    print('Distortion:')
    print(dist)


def show_result(folder: str):
    global inmtx, dist
    folder += '/'
    for i in range(1, 16):
        img = cv2.imread(folder+str(i)+'.bmp',
                         cv2.IMREAD_COLOR)
        uimg = cv2.undistort(img, inmtx, dist)
        uimg = cv2.resize(uimg, (640, 720))
        img = cv2.resize(img, (640, 720))
        both = np.concatenate((img, uimg), axis=1)
        cv2.imshow('1.5', both)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

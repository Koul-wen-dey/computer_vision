import cv2


def p(a):
    print(a)


def find_corner(folder):
    folder += '/Q1_Image/'
    img = cv2.imread(
        'C:/Users/a2722/OneDrive/文件/computer_vision/hw1/Dataset_CvDl_Hw1/Q1_Image', cv2.IMREAD_GRAYSCALE)
    print(img)
    # for i in range(1, 16):
    # print(folder+str(i)+'.bmp')
    # img = cv2.imread(folder+str(i)+'.bmp', cv2.IMREAD_GRAYSCALE)
    # tmp = cv2.findChessboardCorners(img, (12, 9))
    # print(img)
    # pass


def find_intrinsic(folder: str):
    pass


def find_extrinsic(folder: str):
    pass


def find_distortion(folder: str):
    pass


def show_result(folder: str):
    pass

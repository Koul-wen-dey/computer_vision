from sklearn.decomposition import PCA
from glob import glob
import cv2
import numpy as np


def reconstruction(folder:str):
    images = glob(folder+'*')
    pca = PCA()
    print(images)



if __name__ == '__main__':
    reconstruction('./Dataset_CvDl_Hw2/Q4_Image/')
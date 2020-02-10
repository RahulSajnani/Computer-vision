import cv2
import matplotlib.pyplot as plt
import numpy as np


def DLT(pixels_2D, points_3D):
    '''
    Function find camera parameters using Direct Linear Transform algorithm

    Input:
        pixels_2D - 2 x N - Pixel values
        points_3D - 3 x N - World points
    
    Returns:
        projectionMatrix - 3 x 4 - Camera projection matrix 
    '''

    for i in range(pixels_2D.shape[0]):
        
        # Extracting image pixels
        u = pixels_2D[i, 0]
        v = pixels_2D[i, 1]

        # Extracting world points for corresponding image pixels
        X = points_3D[i, 0]
        Y = points_3D[i, 1]
        Z = points_3D[i, 2]

        Camarray = np.array([[X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u],
                             [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]])

        if i == 0:
            pointMatrix = Camarray
        else: 
            pointMatrix = np.vstack((pointMatrix, Camarray))


    U, D, Vt = np.linalg.svd(pointMatrix)
    # print(Vt.shape)
    P = Vt[-1, :]
    P = np.reshape(P, (3, 4))
    
    return P


if __name__ == "__main__":
    imageDir = './Camera_calibration_data/'
    calibObject = 'calib-object.jpg'
    calibObjectLegend = 'calib-object-legend.jpg'
    calibObjectImage = plt.imread(imageDir + calibObject)
    calibObjectLegendImage = plt.imread(imageDir + calibObjectLegend)
    

    # plt.imshow(calibObjectImage)
    # plt.show()

    pixels_2D = np.array([[615 , 1700],
                          [1132, 1640],
                          [1930, 1640],
                          [2418, 1706],
                          [1544, 2090],
                          [1544, 2555],
                          [1277, 1802],
                          [1792, 1806]])

    points_3D = np.array([[   0,   0, 168],
                          [   0,   0,  84],
                          [  84,   0,   0],
                          [ 168,   0,   0],
                          [   0,  84,   0],
                          [   0, 168,   0],
                          [   0,  28,  56],
                          [  28,   0,  56]])

    P = DLT(pixels_2D, points_3D)
    print(P)          

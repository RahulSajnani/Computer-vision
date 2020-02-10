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
    # P = P / P[2,3]
    return P

def decompositionRQ(H):
    '''
    Function to perform RQ decomposition
    Input:
        H - 3 x 3
    Returns:
        R - 3 x 3 - Upper triangular matrix
        Q - 3 x 3 - Orthogonal matrix
    '''
    Q, R = np.linalg.qr(np.flipud(H).T)
    Q = Q.T
    R = np.flipud(R.T)
    R = R[:, ::-1]
    Q = Q[ ::-1, :]
    return R, Q

def decompositionProjectionMatrix(P):

    '''
    Function to decompose projection matrix

    Input:
        P - 3 x 4 - Camera projection matrix
    Returns:
        K - 3 x 3 - Camera intrinsic matrix
        R - 3 x 3 - Rotation matrix
        C - 3 x 1 - Camera center
    '''

    KRMatrix = P[:,:3]
    C = - np.linalg.inv(KRMatrix) @ P[:,3]
    K, R = decompositionRQ(KRMatrix)
    K = K / K[2,2]
    return K, R, C

if __name__ == "__main__":
    imageDir = './Camera_calibration_data/'
    calibObject = 'calib-object.jpg'
    calibObjectLegend = 'calib-object-legend.jpg'
    calibObjectImage = plt.imread(imageDir + calibObject)
    calibObjectLegendImage = plt.imread(imageDir + calibObjectLegend)
    

    

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
                          [  56,  28,   0]])

    P = DLT(pixels_2D, points_3D)
    print('Projection matrix \n', P)          
    K, R, C = decompositionProjectionMatrix(P)
    print('Camera intrinsic matrix \n', K)
    print('Rotation matrix \n', R)
    print('Camera center \n', C)
    
    plt.imshow(calibObjectImage)
    plt.plot(pixels_2D[:,0],pixels_2D[:,1],'ro')
    plt.show()
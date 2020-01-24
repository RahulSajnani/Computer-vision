import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def Vid2Image(vid, directoryName):
    '''
    function to convert video to images
    Input:
        vid - video sequence to convert to images
        directoryName - name of direcotry to output images 
    Output:
        none
    '''
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)

    
    isFrame, image = vid.read()
    i = 0
    while(isFrame):
        imageName = '%06d.jpg' % i
        imagePath = directoryName + '/' + imageName
        cv2.imwrite(imagePath, image)
        isFrame, image = vid.read()
        i = i + 1

def Image2Vid(dirName, fps, outputDir, vidName):

    '''
    function to conver images to video with given fps
    Input:
        dirName - directory name with images
        fps - output fps
        outputDir - output directory name to save video
    Output:
        none
    '''
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    
    imageArray = []
    files = [images for images in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, images))]
    files.sort()
    

    for i in range(len(files)):
        fileName = dirName + '/'+ files[i]
        image = cv2.imread(fileName)
        imageArray.append(image)
    
    height, width, layers = imageArray[0].shape
    size = (width, height)

    video = cv2.VideoWriter(outputDir + '/' + vidName,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(imageArray)):
        video.write(imageArray[i])
    video.release()    

def CaptureImagesWebcam(outputDir):

    '''
    Function to capture images from webcam
    Press Q to stop capturing images
    Input:
        outputDir - output directory name
    Output:
        none
    '''

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    cam = cv2.VideoCapture(0)
    i = 0
    while True:

        isFrame, image = cam.read()
        imageName = '%06d.jpg' % i
        imagePath = outputDir + '/' + imageName
        cv2.imshow('webcam',image)
        cv2.imwrite(imagePath, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        isFrame, image = cam.read()
        i = i + 1

def ChromaKey(image1, image2):

    '''
    Chroma keying function to replace background of image1 
    Input:
        image1 - foreground image
        image2 - background image
    Output:
        imageOut - image with foreground of image 1 and background of image 2    
    '''

    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    lower_green = np.array([0,205,0], dtype=np.uint8)
    upper_green = np.array([100,255,150], dtype=np.uint8)
    mask = cv2.inRange(hsv1, lower_green, upper_green)
    res = cv2.bitwise_and(image1,image1, mask= mask)
    res = hsv1 - res
    background = image2[:mask.shape[0],:mask.shape[1],:]
    background[mask == 0] = [0, 0, 0]    
    imageOut = res + background
    # imageOut = cv2.cvtColor(imageOut, cv2.COLOR_BGR2RGB)
    # plt.imshow(imageOut)
    # plt.show()
    
    return imageOut


def ReplaceBackgroundGreen(inputDir1, inputDir2, outputDir):

    '''
    Function to change background of directory
    Input:
        inputDir - input directory containing images
        outputDir - directory to store output images
    Output:
        none
    '''
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    imageArray = []
    files1 = [images for images in os.listdir(inputDir1) if os.path.isfile(os.path.join(inputDir1, images))]
    files1.sort()

    imageArray2 = []
    files2 = [images for images in os.listdir(inputDir2) if os.path.isfile(os.path.join(inputDir2, images))]
    files2.sort()
    numFiles = len(files1)

    if len(files2) < len(files1):
        numFiles = len(files2)


    for i in range(numFiles):
        fileName = inputDir1 + '/'+ files1[i]
        image = cv2.imread(fileName)
        imageArray.append(image)

    for i in range(numFiles):
        fileName = inputDir2 + '/'+ files2[i]
        image = cv2.imread(fileName)
        imageArray2.append(image)
    # print(len(imageArray))
    # print(len(imageArray2))
    i = 0
    
    # print(numFiles)
    for j in range(numFiles):
        # print(images.shape)
        imageName = '%06d.jpg' % i
        imagePath = outputDir + '/' + imageName
        i = i + 1
        print(i)
        outputImage = ChromaKey(imageArray[j], imageArray2[j])
        # time.sleep(0.5)
        cv2.imwrite(imagePath, outputImage)

if __name__ == "__main__":
    
    dirName = 'testFramesHelicopter'
    dirBackground = 'testFrames'
    videoName = 'GreenHelicopter.mp4'
    videoOutputDir = 'videoDirHelicopter'
    fps = 30
    webcamOutputDir = 'webOut'
    videoOutputName = 'testHelicopter.avi'
    greenScreen = 'greenscreen.jpg'
    image = 'testFramesHelicopter/000128.jpg'
    outputGreen = 'greenOut'
    ChromaKeyOutputDir = 'ChromaResultHelicopter'
    image = cv2.imread(image)
    greenScreen = cv2.imread(greenScreen)
    # test = cv2.imread('testFrames/000001.jpg')
    # ChromaKey(image, test)
    # vid = cv2.VideoCapture(videoName)
    # CaptureImagesWebcam(webcamOutputDir)
    # Vid2Image(vid, dirName)
    # Image2Vid(ChromaKeyOutputDir, fps, videoOutputDir, videoOutputName)
    # ReplaceBackground(webcamOutputDir, outputGreen,greenScreen)
    ReplaceBackgroundGreen(dirName, dirBackground, ChromaKeyOutputDir)

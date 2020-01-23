import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([255,255,40], dtype=np.uint8)

    mask = cv2.inRange(hsv1, lower_white, upper_white)
    res = cv2.bitwise_and(image1,image1, mask= mask)

    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    background = image2[:res.shape[0],:res.shape[1]]
    background[0 != mask] = [0, 0, 0]
    
    
    imageOut = res + background
    return imageOut



if __name__ == "__main__":
    
    dirName = 'testFrames'
    videoName = 'test.mp4'
    videoOutputDir = 'videoDir'
    fps = 24
    webcamOutputDir = 'webOut'
    videoOutputName = 'test.avi'
    greenScreen = 'greenscreen.jpg'
    image = 'webOut/000000.jpg'
    image = cv2.imread(image)
    greenScreen = cv2.imread(greenScreen)
    ChromaKey(image, greenScreen)
    # vid = cv2.VideoCapture(videoName)
    # CaptureImagesWebcam(webcamOutputDir)
    # Vid2Image(vid, dirName)
    # Image2Vid(dirName, fps, videoOutputDir, videoOutputName)


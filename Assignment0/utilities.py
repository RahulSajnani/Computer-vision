import cv2
import numpy as np
import os

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


if __name__ == "__main__":
    
    dirName = 'testFrames'
    videoName = 'test.mp4'
    videoOutputDir = 'videoDir'
    fps = 24
    webcamOutputDir = 'webOut'
    vid = cv2.VideoCapture(videoName)
    videoOutputName = 'test.avi'
    CaptureImagesWebcam(webcamOutputDir)
    # Vid2Image(vid, dirName)
    # Image2Vid(dirName, fps, videoOutputDir, videoOutputName)

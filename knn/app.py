import cv2
import numpy as np
import math
import time
import operator
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
cap = cv2.VideoCapture(0)
MIN_CONTOUR_AREA = 25
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
gettime=0
take=""
adding=40
get=0
getstring=""
taketime=""
strCurrentChar=""
getit=""

class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0
    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA:
            return False
        return True

while True:
    ret, img = cap.read()
    npaClassifications = np.loadtxt("classify.txt", np.float32)
    npaFlattenedImages = np.loadtxt("feat.txt", np.float32)
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    
    crop_img = img[100:300, 100:300]
    kulaygray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35,35)
    #BLURRED PARA MAS KITA ANG BLACK AND WHITE PAG NAG THRESHOLDING
    blurred = cv2.GaussianBlur(kulaygray, value, 0)
    cv2.imshow('blurred',blurred)
    #TRESHOLDING PARA BLACK AND WHITE.
    _,thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                            
    cv2.imshow('Thresholded', thresh1)
    
    allContoursWithData = []
    validContoursWithData = []


    
    
        ###############
        
        ########
   
    imgThreshCopy = thresh1.copy()
    npaContours, _ = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:
        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)

    
    validContoursWithData.sort(key = operator.attrgetter("intRectX"))

    strFinalString = ""
    for contourWithData in validContoursWithData:
        cv2.rectangle(crop_img,(contourWithData.intRectX, contourWithData.intRectY),(contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),(0, 255, 0),2)

        imgROI = thresh1[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
        strCurrentChar = str(chr(int(npaResults[0][0])))

        strFinalString = strFinalString + strCurrentChar
    
    
    
            
    getit=strFinalString+getit  
    getstring=strFinalString
    
    gettime=gettime+1
    if gettime==15:
        take=strCurrentChar
    
    
    if gettime>=20:
        
        if gettime==adding:
            
            taketime=adding
            adding=adding+40
            take= take+strCurrentChar
    cv2.putText(crop_img,take,(0,190),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    if len(take)==13:
        take=""
    #dapat nasa thresh1 to yung copy
    cv2.putText(crop_img,strCurrentChar,(0,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    
    cv2.rectangle(crop_img,(400,190),(200,150),(0,255,0),1)
    cv2.imshow('ALL IMAGE', crop_img)
    ###
    

    k = cv2.waitKey(10)
    if k == 37:
        break
    

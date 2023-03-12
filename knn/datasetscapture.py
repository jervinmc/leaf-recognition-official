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
while True:
    ret, img = cap.read()
    crop_img = img[100:300, 100:300]
    cv2.imshow("asd",crop_img)
    k = cv2.waitKey(10)
    if k == 37:
        break

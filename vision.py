import cv2 as cv
import numpy as np 
import math

img = cv.imread('left.jpeg')

w, h = img.shape[1], img.shape[0]
sectionW = math.ceil(w/3)
sectionH = math.ceil(h/3)
darkestJunctions = (0, 167, 85)
lightestJunctions = (255, 255, 128)
smallestArea = 4000
junctionNumAttr = 0
junctionPointAttr = (0, 0)
junctionDistanceAttr = 0
propAreaAttr = 0

imgCrop1 = img[sectionH:h, :sectionW]
imgCrop2 = img[sectionH:h, sectionW:sectionW*2]
imgCrop3 = img[sectionH:h, 2*sectionW:w]


hsv1 = cv.cvtColor(imgCrop1, cv.COLOR_BGR2HSV)
hsv2 = cv.cvtColor(imgCrop2, cv.COLOR_BGR2HSV)
hsv3 = cv.cvtColor(imgCrop3, cv.COLOR_BGR2HSV)
BLUE_MIN = np.array([110,50,50], np.uint8)
BLUE_MAX = np.array([130,255,255], np.uint8)
dst1 = cv.inRange(hsv1, BLUE_MIN, BLUE_MAX)
dst2 = cv.inRange(hsv2, BLUE_MIN, BLUE_MAX)
dst3 = cv.inRange(hsv3, BLUE_MIN, BLUE_MAX)
quantity1 = cv.countNonZero(dst1)
quantity2 = cv.countNonZero(dst2)
quantity3 = cv.countNonZero(dst3)

print(quantity1, quantity2, quantity3)

gray = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
grayb = cv.GaussianBlur(gray, (15, 15), 0)
thresholded = cv.inRange(grayb, darkestJunctions, lightestJunctions)
contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

bigContours = []
if contours:
    biggestContour = max(contours, key=cv.contourArea)
    bigContours.append(biggestContour)
    # Find centroid
    moments = cv.moments(biggestContour)
    if moments['m00'] != 0:
        junctionPoint = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    else:
        junctionPoint = (0, 0)
    # Assign attributes
    junctionNumAttr = len(contours)
    propAreaAttr = cv.contourArea(biggestContour)
    junctionDistanceAttr = 240000 / cv.contourArea(biggestContour)
    junctionPointAttr = junctionPoint
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv.drawContours(img, bigContours, -1, (255, 0, 0), 3)


if quantity1>quantity2 and quantity1>quantity3:
    print('prop is on left')
    output = cv.bitwise_and(imgCrop1, imgCrop1, mask = dst1)
    cv.imshow("images", np.hstack([imgCrop1, output]))
    gray = cv.cvtColor(imgCrop1, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 30, 200)
    cv.imshow('canny', canny)
elif quantity2>quantity1 and quantity2>quantity3:
    print('prop is on center')
    output = cv.bitwise_and(imgCrop2, imgCrop2, mask = dst2)
    cv.imshow("images", np.hstack([imgCrop2, output]))
    gray = cv.cvtColor(imgCrop2, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 30, 200)
    cv.imshow('canny', canny)
else:
    print('prop is on right')
    output = cv.bitwise_and(imgCrop3, imgCrop3, mask = dst3)
    cv.imshow("images", np.hstack([imgCrop3, output]))
    gray = cv.cvtColor(imgCrop3, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 30, 200)
    cv.imshow('canny', canny)

#output = cv.bitwise_and(imgCrop2, imgCrop2, mask = dst2)
#cv.imshow("images", np.hstack([imgCrop2, output]))

#cv.imshow('cropped', imgCrop1)
cv.imshow('original', img)
cv.waitKey(0)
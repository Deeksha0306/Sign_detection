import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

vid = cv.VideoCapture(0)
while(1):
    _,img = vid.read()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    red_l = np.array([135,85,115], np.uint8)
    red_u = np.array([185,250,250], np.uint8)
    redmask = cv.inRange(hsv, red_l, red_u)
    
    green_l = np.array([40,75,75], np.uint8)
    green_u = np.array([80,255,255], np.uint8)
    greenmask = cv.inRange(hsv, green_l, green_u)

    blue_l = np.array([100, 150, 50], np.uint8)
    blue_u = np.array([130,255,255 ], np.uint8)
    bluemask = cv.inRange(hsv, blue_l, blue_u)

    kernel = np.ones((5,5), 'uint8')
    redmask = cv.dilate(redmask, kernel)
    red = cv.bitwise_and(img,img,mask=redmask)

    greenmask = cv.dilate(greenmask, kernel)
    green = cv.bitwise_and(img,img,mask=greenmask)

    bluemask = cv.dilate(bluemask, kernel)
    blue = cv.bitwise_and(img,img,mask=bluemask)

    contour, heirarchy = cv.findContours(redmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contour):
        area = cv.contourArea(contour)
        if area>300 :
            x, y, w, h = cv.boundingRect(contour)
            img = cv.rectangle(img, (x,y),(x+w, y+h), (0,0,255), thickness = 2)
            cv.putText(img,"Red color", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    

    contour, heirarchy = cv.findContours(greenmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contour):
        area = cv.contourArea(contour)
        if area>300 :
            x, y, w, h = cv.boundingRect(contour)
            img = cv.rectangle(img, (x,y),(x+w, y+h), (0,255,0), thickness = 2)
            cv.putText(img,"Green color", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    contour, heirarchy = cv.findContours(bluemask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contour):
        area = cv.contourArea(contour)
        if area>300 :
            x, y, w, h = cv.boundingRect(contour)
            img = cv.rectangle(img, (x,y),(x+w, y+h), (255,0,0), thickness = 2)
            cv.putText(img,"Blue color", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    cv.imshow("blue color",img)

    if cv.waitKey(10) & 0xff==ord('d'):
        vid.release()
        cv.destroyAllWindows()
        break

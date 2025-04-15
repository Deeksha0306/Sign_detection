import cv2 as cv
face_cascade = cv.CascadeClassifier("OpenCV-journey\images\haarcascade_frontalface_default.xml")
vid = cv.VideoCapture(0)
while True:
    _, img = vid.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y),(x+w,y+h), (255,0,0),4)
    cv.imshow('image',img)
    k=cv.waitKey(30) & 0xff
    if k==27:
        break
vid.release()
cv.destroyAllWindows()
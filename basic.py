import cv2 as cv
capture = cv.VideoCapture("../images/vid.mp4")
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.putText(frame,"journey to jammu", (20,100),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv.imshow('frame',frame)
    cv.imshow("grayframe",frame_gray)
    cv.imshow("rgbframe",frame_rgb)
    if cv.waitKey(3) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()


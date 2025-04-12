import cv2 as cv
# img = cv.imread("../images/girl.jpeg")
# cv.imshow("image",img)
# imgresi = cv.resize(img,(200,599))
# cv.imshow("resized",imgresi)
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow('frame',frame)
    cv.imshow("grayframe",frame_gray)
    cv.imshow("rgbframe",frame_rgb)
    if cv.waitKey(3) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()


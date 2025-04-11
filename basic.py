import cv2 as cv
img = cv.imread("../images/girl.jpeg")
cv.imshow("image",img)
imgresi = cv.resize(img,(200,599))
cv.imshow("resized",imgresi)
cv.waitKey(0)
cv.destroyAllWindows()
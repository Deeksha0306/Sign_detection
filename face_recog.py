import cv2 as cv
import os

def create_user(f_id, name):
    cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier("../images/haarcascade_frontalface_default.xml")

    dataset_dir = "dataset"
    user_dir = os.path.join(dataset_dir, name)
    if not os.path.isdir(user_dir):
        os.makedirs(user_dir)

    counter = 0
    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            counter += 1
            face_img = gray[y:y+h, x:x+w]

            file_path = os.path.join(user_dir, f"{name}.{f_id}.{counter}.jpg")
            cv.imwrite(file_path, face_img)

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.imshow("Face Capture", frame)

        k = cv.waitKey(100) & 0xFF
        if k == 27:
            break
        elif counter >= 40:
            print("Collected 40 face images.")
            break

    cap.release()
    cv.destroyAllWindows()

def train():
    database = 'dataset'
    img_dir = [x[0] for x in os.walk(database)][1::]
    recogniser = cv.face.LBPHFaceRecognizer_create()


#create_user(f_id=1, name="maya")  


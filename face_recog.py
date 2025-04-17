import cv2 as cv
import numpy as np
from PIL import Image
import os

def create_user(f_id, name):
    cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


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
#create_user(f_id=1, name="Maya")  

def train():
    database = 'dataset'
    img_dir = [x[0] for x in os.walk(database)][1::]
    recognizer = cv.face.LBPHFaceRecognizer_create()
    detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    faceSamples = []
    ids = []

    for path in img_dir:
        path = str(path)
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split('.')[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

    recognizer.train(faceSamples,np.array(ids))
    recognizer.write('trainer.yml')
    print('\n[INFO] {0} faces trained. Existing Program'.format(len(np.unique(ids))))
    return len(np.unique(ids))
#train()

def recognize(names):
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    cascadePath = "../images/haarcascade_frontalface_default.xml"  # Make sure the path is correct
    faceCascade = cv.CascadeClassifier(cascadePath)

    font = cv.FONT_HERSHEY_SIMPLEX
    id = 0
    name = ""
    face_count = 0

    cam = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not cam.isOpened():
        print("Error: Camera could not be opened.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        
        # Check if the frame was captured successfully
        if not ret:
            print("Failed to capture image from the camera. Retrying...")
            continue  # Skip the rest of the loop if frame capture failed

        img = cv.flip(img, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 70:
                if id in names:
                    id = names[id]
                else:
                    id = 'Unknown'
            else:
                id = 'Unknown'

            confidence = " {0}%".format(round(100 - confidence))

            cv.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv.imshow("Camera", img)

        k = cv.waitKey(30) & 0xff
        if k == 27: 
            break

    cam.release()  
    cv.destroyAllWindows()  
recognize({1: "Maya"})

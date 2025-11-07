import cv2 as cv
import os

from cv2 import waitKey

camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Camera not opened, terminating..")
    exit()

Labels = ["bird", "boar", "dog", "dragon", "hare", "horse", "monkey", "ox", "ram", "rat", "serpent", "tiger"]

for label in Labels:
    folder_path = f'./data/drive_data/{label}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

for folder in Labels:

    count = 0

    print("Press 'x' for " + folder)
    userinput = input()
    if userinput != 'x':
        print("Wrong Input, terminating...")
        exit()
    
    print("Preview mode - Press 'p' to start capturing, 'q' to quit preview")
    while True:
        status, frame = camera.read()
        if not status:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (600,400))
        cv.imshow("Preview - Press 'p' to start, 'q' to quit", gray)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('p'):
            break
        elif key == ord('q'):
            print("Preview cancelled")
            exit()

    cv.destroyWindow("Preview - Press 'p' to start, 'q' to quit")
    
    waitKey(3000)
    
<<<<<<< HEAD
    while count < 50:
=======
    while count < 10: # pics per animal
>>>>>>> 133981086cc31cbac3870ebdaeaa4d05b21cc1e7

        status, frame = camera.read()

        if not status:
            print("Frame is not been captured, terminating...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("Video Window",gray)
        gray = cv.resize(gray, (600,400))
        cv.imwrite('data/drive_data/' + folder + '/' + folder + str(count) + '.jpg', gray)
        count = count + 1
        waitKey(100)
        if cv.waitKey(1) == ord('q'):
            break

camera.release()
cv.destroyAllWindows()

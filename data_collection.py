import cv2
import numpy as np
import os

path = os.getcwd()
imageDir = "%s/yogesh"%path
testDir = "%s/test"%imageDir
trainDir = "%s/train"%imageDir
'''
if not os.path.isdir(imageDir):
    os.system("mkdir %s"%imageDir)
    os.system("mkdir %s"%testDir)
    os.system("mkdir %s"%trainDir)
'''

counter = 0
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    status, image = cap.read()
    if status:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cor = face_classifier.detectMultiScale(gray_image, 1.3, 5)
        if len(face_cor) > 0:
            counter += 1
            for (x, y, w, h) in face_cor:
                cropped_image = image[y-15:y+h+15, x-20:x+w+20]
                if cropped_image.any():
                    cv2.imshow("image", cv2.resize(cropped_image, (200, 200)))
                    if counter < 400:
                        cv2.imwrite("%s/yogesh_%d.jpg"%(trainDir, counter), cropped_image)
                    elif counter >= 400 and counter <= 600:
                        cv2.imwrite("%s/yogesh_%d.jpg"%(testDir, counter), cropped_image)

        k = cv2.waitKey(1)
        if k == 27:
            break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")

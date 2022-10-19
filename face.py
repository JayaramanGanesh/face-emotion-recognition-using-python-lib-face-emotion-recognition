import cv2
from facial_emotion_recognition import EmotionRecognition

er = EmotionRecognition(device="cpu")
cam=cv2.VideoCapture(0)

while True:
    red,frame = cam.read()
    frame =er.recognition_emotion(frame, return_type="BGR")
    cv2.imshow("frame",frame)
    key =cv2.waitKey(1)
    if key == 27:
        break


cam.release()
cv2.destroyAllWindows()

"""
versions
-------------------
python - 3.7.8
torch - 1.3.1
torchvision - 0.4.2
facial-emotion-recognition -0.3.4


devices types
-------------------
pc or computer mode was cpu or gpu

pc processor was cpu 4 line u put device configure cpu
pc processor was gpu 4 linr u put devices configure gpu



camera setup 
--------------------
inbelt camera u put 0 in 5 line code videocapture(0)
external camera u put 1 in 5 line code videocapture(1) [external camera based on value change]



"""
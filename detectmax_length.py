import cv2
import numpy as np
import time
import mediapipe as mp
import math

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize mediapipe hands model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils  # To draw hand landmarks
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky finger tip landmarks

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Find the position of hand landmarks."""
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def findDistance(self, p1, p2, img, draw=True):
        """Find the distance between two points (landmarks)."""
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        length = math.hypot(x2 - x1, y2 - y1)  # Euclidean distance
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        return length

wcam, hcam = 640, 480  # Set webcam resolution

cap = cv2.VideoCapture(0)  # Use the primary camera (index 0)
cap.set(3, wcam)
cap.set(4, hcam)

pTime = 0
detector = handDetector(detectionCon=0.7)  # Initialize the hand detector
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
vol=0
volPer=0
volBar = 400
minvol=volRange[0]
maxvol=volRange[1]
while True:
    success, img = cap.read()  # Capture frame from webcam
    if not success:
        break

    img = detector.findHands(img)  # Find and draw hands in the frame
    lmList = detector.findPosition(img,draw=False)  # Get landmark positions

    if len(lmList) != 0:
        #print(lmList[4],lmList[8])
        
        x1, y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+y1)//2,(x2+y2)//2
        
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        length = math.hypot(x2-x1,y2-y1)
        print(length)
        
        
        # hand range 50 300
        # volume range -65 0
        
        vol= np.interp(length,[50,300],[minvol,maxvol])
        volBar= np.interp(length,[50,300],[400,150])
        volPer= np.interp(length,[50,300],[0,100])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(0, None)
        
        
        if length <50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
            
    cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, ( 0,255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # Calculate FPS
    pTime = cTime

    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)  # Display the resulting image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop when 'q' is pressed
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close the OpenCV window

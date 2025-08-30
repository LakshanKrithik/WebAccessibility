import cv2
import numpy as np
import time
import autopy
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera and frame settings
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()
smoothening = 7

# Mouth landmarks for upper and lower lips
MOUTH_UPPER_LIP = 13
MOUTH_LOWER_LIP = 14

# Clicking threshold
MOUTH_OPEN_THRESHOLD = 15

# Chin landmark index
CHIN_LANDMARK = 152

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get chin landmark
            chin = face_landmarks.landmark[CHIN_LANDMARK]
            chin_x = int(chin.x * wCam)
            chin_y = int(chin.y * hCam)

            # Draw a circle on the chin
            cv2.circle(img, (chin_x, chin_y), 10, (255, 0, 255), cv2.FILLED)

            # Map chin position to screen coordinates
            x3 = np.interp(chin_x, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(chin_y, (frameR, hCam - frameR), (0, hScr))

            # Smooth the movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

            # Detect mouth opening for clicking
            upper_lip = face_landmarks.landmark[MOUTH_UPPER_LIP]
            lower_lip = face_landmarks.landmark[MOUTH_LOWER_LIP]
            mouth_distance = abs(upper_lip.y - lower_lip.y) * hCam

            if mouth_distance > MOUTH_OPEN_THRESHOLD:
                cv2.putText(img, "Click!", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Chin Mouse Control", img)
    cv2.waitKey(1)

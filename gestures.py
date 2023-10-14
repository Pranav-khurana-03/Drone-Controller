import cv2
import mediapipe as mp

modelPath = '/home/pranavk/Drone-Controller/HandDetectionModel/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=modelPath),
    #running_mode=VisionRunningMode.LIVE_STREAM
    running_mode=VisionRunningMode.IMAGE
)

landmarker = HandLandmarker.create_from_options(options)

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read() # ret is whether the read was sucessful or not
                            # frame is the matrix of pixels that make up the captured image
    if ret is False:
        break
    cv2.imshow('Frame',frame) # 'Frame' is the name of the window. It can be anything you want.




# cap = cv2.VideoCapture(0)
# hand_detector = mediapipe.solutions.hands.Hands()

# while True:
#     # Read the video frame
#     success, frame = cap.read()

#     # If the video frame is not successful, break the loop
#     if not success:
#         break

#     # Detect hands in the video frame
#     hands = hand_detector.process(frame)

#     # If hands are detected, track their movement
#     if hands.detections:
#         for hand in hands.detections:
#             # Print the hand's center point
#             print(hand.center)

#     # Display the video frame
#     cv2.imshow('Hand Gesture Detection', frame)

#     # Check if the user has pressed the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object
# cap.release()

# # Close all open windows
# cv2.destroyAllWindows()


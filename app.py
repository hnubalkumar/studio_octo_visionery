import streamlit as st
import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2, av
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from tensorflow.keras.models import load_model

st.title("Emotion & Gesture Detection")
st.divider()

tab1, tab2 = st.tabs(["Emotion_Detector","Gesture_Detector"])
with tab1:
    # loading emotion_model
    emotion_model = load_model("Emotion_Detection.keras")
    emotion_labels = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprise',
        'Neutral'
    ]

    # Initialising Haarcascades

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    print("Loaded:", face_detector) 
    print("Cascade Loaded:", face_detector)

    # Making a class for emotion_detection
    class EmotionProcessor():
        def __init__(self):
            print("Emotion Processor Initialised")

        def recv(self,frame):
            print("Frame received")
        
            frm = frame.to_ndarray(format="bgr24")
            print(frm.shape)

            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60,60)
            )
            
            for x,y,w,h in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face,(48,48))
                face = face.astype("float32")/255
                face = face.reshape(1,48,48,1)

                prediction = emotion_model.predict(face, verbose=0)
                print("Prediction:", prediction)                
                label = emotion_labels[np.argmax(prediction)]
                print("Label:", label)
                cv2.rectangle(
                    frm,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    3
                )
              
                cv2.putText(
                    frm,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                # Code for calling the api and predicting our emotions
                                
                api_url = f"http://127.0.0.1:8000/predict?img1={emotion_labels[0]}&img2={emotion_labels[1]}&img3={emotion_labels[2]}&img4={emotion_labels[3]}&img5={emotion_labels[4]}&img6={emotion_labels[5]}&img7={emotion_labels[6]}"
                print("API URL:", api_url)
                # Using the requests library to send a GET request to the API
                response = requests.get(api_url)
                print("Response:", response.json())

                if response.status_code == 200:
                    prediction_result = response.json().get("Prediction")
                    if prediction_result == 1:
                        st.success("The camera detected the emotion.")
                    else:
                        st.error("The camera couldn't function properly. Please check the camera and try again.")
                else:
                    st.error("Server not playing!")
                
                return av.VideoFrame.from_ndarray(frm,format="bgr24") 

st.write("Creating webrtc streamer...")
ctx = webrtc_streamer(
    key="emotion",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
         "video":True,
         "audio":False
    }
)

st.write("WebRTC playing:", ctx.state.playing)

with tab2:

    # loading gesture_model
    gesture_model = load_model("Gesture_Detection.keras")

    gestures = [
        "Fist",
        "Five",
        "Okay",
        "Peace",
        "Rad",
        "Thumbs"
    ]
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

#     # Setup the hand tracking model
#     hands = mp_hands.Hands(
#         static_image_mode=False,        # False treats input as a continuous video stream
#         max_num_hands=1,                # Limit detection to 1 hand for simplicity
#         min_detection_confidence=0.7,   # Threshold for initial hand detection
#         min_tracking_confidence=0.7     # Threshold for tracking landmarks
#     )

#     class GestureProcessor():
#         def __init__(self):
#             print("Gesture Processor Initialised")

#         def recv(self, frame):
#             frm = frame.to_ndarray(format="bgr24")
#             frm = cv2.flip(frm, 1)
                
# #           # MediaPipe requires RGB images, but OpenCV reads frames in BGR
            
#             rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                
# #           # Process the frame and find hands
#             results = hands.process(rgb_frame)

# #           # 3. Check if any hands are detected
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Draw the skeleton connections on the screen
#                     mp_draw.draw_landmarks(frm, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 api_url = f"http://127.0.0.1:8000/predict?Fist_img={gestures[0]}&Five_img={gestures[1]}&Okay_img={gestures[2]}&Peace_img={gestures[3]}&Rad_img={gestures[4]}&Thumbs_img={gestures[5]}"
#                 # Using the requests library to send a GET request to the API
#                 response = requests.get(api_url)
                            
#                 if response.status_code == 200:
#                     prediction_result = response.json().get("Prediction")
#                     if prediction_result == 1:
#                         st.success("The camera detected the emotion.")
#                     else:
#                         st.error("The camera couldn't function properly. Please check the camera and try again.")
#                 else:
#                     st.error("Server not playing!")
                
#                 return av.VideoFrame.from_ndarray(frm,format="bgr24")

            
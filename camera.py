import cv2
from Themodel import FacialExpressionModel
import numpy as np
import pandas as pd
import time

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel("model.json", "model_weights.h5")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.emotion_data = pd.DataFrame(columns=['Name', 'Emotion', 'Time'])
        self.start_time = time.time()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces):
            face_region = gray_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_region, (48, 48))
            emotion_prediction = self.model.predict_emotion(resized_face[np.newaxis, :, :, np.newaxis])

            # Draw square box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Add person number at the top of the box
            cv2.putText(frame, f"Person {i+1}", (x, y-10), self.font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            # Add emotion at the bottom of the box
            cv2.putText(frame, emotion_prediction, (x, y+h+20), self.font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            # Record emotion data every 10 seconds
            current_time = time.time()
            if current_time - self.start_time >= 10:
                self.emotion_data = self.emotion_data.append({'Name': f'Person {i+1}', 'Emotion': emotion_prediction, 'Time': time.strftime('%Y-%m-%d %H:%M:%S')}, ignore_index=True)
                self.start_time = current_time
                self.save_emotion_data()

        _, jpeg_frame = cv2.imencode('.jpg', frame)
        return jpeg_frame.tobytes()

    def save_emotion_data(self, filename='emotion_data.csv'):
        self.emotion_data.to_csv(filename, index=False, mode='a', header=False)

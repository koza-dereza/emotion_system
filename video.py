import cv as cv
import numpy as np
import moviepy as mp
import cv2
import altair as alt
import speech_recognition as sr
import base64
import tempfile
import pandas as pd
import streamlit as st
import streamlit_webrtc
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from keras.models import model_from_json,load_model
from datetime import datetime
import os


emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("emotion_model1.h5")



class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # load face
        try:
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        except Exception:
            st.write("Error loading cascade classifiers")
        img = frame.to_ndarray(format="bgr24")
        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def save_video(file):
        if file.size > 400000000000000:
            return 1
        # if not os.path.exists("audio"):
        #     os.makedirs("audio")
        folder = "video1"
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # clear the folder to avoid storage overload
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        try:
            with open("log0.txt", "a") as f:
                f.write(f"{file.name} - {file.size} - {datetoday};\n")
        except:
            pass

        with open(os.path.join(folder, file.name), "wb") as f:
            f.write(file.getbuffer())
        return 0
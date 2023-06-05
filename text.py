#text

import joblib
import pickle

pipe_lr = joblib.load(open("emotion_text.pkl", "rb"))

def predict_emotion(text):
    result = pipe_lr.predict([text])

    return result[0]

def get_prediction_proba(text):
    result = pipe_lr.predict_proba([text])

    return result

emotions_emoji_dict = {"anger", "disgust", "fear", "happy", "joy",
                               "neutral", "sad", "sadness", "shame", "surprise"}
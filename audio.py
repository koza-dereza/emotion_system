import numpy as np
import cv2

import librosa

import matplotlib.pyplot as plt
import os

from datetime import datetime



class AudioTransformer():
    CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
    COLOR_DICT = {"neutral": "grey",
                  "positive": "green",
                  "happy": "green",
                  "surprise": "orange",
                  "fear": "purple",
                  "negative": "red",
                  "angry": "red",
                  "sad": "lightblue",
                  "disgust": "brown"}

    TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
    TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])


    def get_melspec(audio):
        y, sr = librosa.load(audio, sr=44100)
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        img = np.stack((Xdb,) * 3, -1)
        img = img.astype(np.uint8)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.resize(grayImage, (224, 224))
        rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
        return (rgbImage, Xdb)



    def color_dict(coldict=COLOR_DICT):
        return AudioTransformer.COLOR_DICT


    def get_title(predictions, categories=CAT7):
        title = f"Эмоции {categories[predictions.argmax()]} \
        - {predictions.max() * 100:.2f}%"
        return title



    def plot_polar(fig, predictions=TEST_PRED, categories=TEST_CAT,
                   title="TEST", colors=COLOR_DICT):
        # color_sector = "grey"

        N = len(predictions)
        ind = predictions.argmax()

        COLOR = color_sector = colors[categories[ind]]
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = np.zeros_like(predictions)
        radii[predictions.argmax()] = predictions.max() * 10
        width = np.pi / 1.8 * predictions
        fig.set_facecolor("#d1d1e0")
        ax = plt.subplot(111, polar="True")
        ax.bar(theta, radii, width=width, bottom=0.0, color=color_sector, alpha=0.25)

        angles = [i / float(N) * 2 * np.pi for i in range(N)]
        angles += angles[:1]

        data = list(predictions)
        data += data[:1]
        plt.polar(angles, data, color=COLOR, linewidth=2)
        plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

        ax.spines['polar'].set_color('lightgrey')
        ax.set_theta_offset(np.pi / 3)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
        plt.suptitle(title, color="darkblue", size=12)
        plt.title(f"BIG {N}\n", color=COLOR)
        plt.ylim(0, 1)
        plt.subplots_adjust(top=0.75)

    def save_audio(file):
        if file.size > 400000000:
            return 1
        # if not os.path.exists("audio"):
        #     os.makedirs("audio")
        folder = "audio"
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


    def get_mfccs(audio, limit):
        y, sr = librosa.load(audio)
        a = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if a.shape[1] > limit:
            mfccs = a[:, :limit]
        elif a.shape[1] < limit:
            mfccs = np.zeros((a.shape[0], limit))
            mfccs[:, :a.shape[1]] = a
        return mfccs

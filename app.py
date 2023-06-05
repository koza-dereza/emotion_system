from joblib import load
import pandas as pd
import altair as alt
from librosa.feature import melspectrogram
import plotly
import moviepy
from moviepy.editor import *
import tempfile
from streamlit_webrtc import  VideoTransformerBase
import base64

from audio import *
from video import *
from text import *


emotions = {0:'angry', 1:'calm', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprise'}


# Define the classes
emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
classifier.load_weights("emotion_model1.h5")


# Define a function to extract audio features
def extract_audio_features(video_path):
    # Load the video file
    video = VideoFileClip(video_path)
    # Extract the audio from the video
    audio = video.audio
    # Convert the audio to a NumPy array
    audio_array = audio.to_soundarray()
    # Extract audio features using Librosa
    features = librosa.feature.mfcc(y=np.transpose(audio_array), sr=audio.fps, n_mfcc=13)
    return features

# Define a function to extract visual features
def extract_visual_features(video_path):
    # Load the video file
    video = VideoFileClip(video_path)
    # Extract visual features using MoviePy
    frames = np.array(list(video.iter_frames()))
    # Reshape the frames to match the input shape of the model
    features = np.reshape(frames, (frames.shape[0], frames.shape[1], frames.shape[2], 3))
    return features

# Define a function to extract text features


# Define the Streamlit app


def main():
    # Face Analysis Application #
    st.title("Распознавание эмоций")
    activiteis = ["Распознавание эмоций по выражению лица с веб-камеры", "Распознавание эмоций по загруженному видео","Распознавание эмоций по аудиофайлам", "Распознавание эмоций по тексту","multi"]
    choice = st.sidebar.selectbox("Выберите вариант", activiteis)
    if choice == "Распознавание эмоций по выражению лица с веб-камеры":
        st.header("Распознавание эмоций по выражению лица с веб-камеры")
        model_video = load_model('video_r.hdf5')
        st.header("Распознавание эмоций по выражению лица с веб-камеры")
    st.header("Распознавание эмоций по выражению лица с веб-камеры")
    run = st.checkbox('Run')
    if run:
        # Main loop for capturing and processing frames
        for frame in st.camera():
            # Process the frame
             img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             faces = haar_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)

             for (x, y, w, h) in faces:
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
                roi_gray = img_gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = np.expand_dims(roi, axis=0)
                    prediction = model.predict(roi)[0]
                    maxindex = int(np.argmax(prediction))
                    finalout = emotion_dict[maxindex]
                    output = str(finalout)

                label_position = (x, y)
                cv2.putText(frame, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                st.image(frame, channels="BGR")

    if choice == "Распознавание эмоций по загруженному видео":
        m = []
        st.markdown("Загрузите видеофаил")
        with st.container():

            FRAME_WINDOW = st.image([])
            col1, col2 = st.columns(2)
            with col1:
                uploaded_video = st.file_uploader("Загрузите видео в формате mp4", type=['mp4'])
                if uploaded_video is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_video.read())
                    cap = cv2.VideoCapture(tfile.name)
                    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    frames = []
                    count = 0
                    skip = 3
                    if st.button('Эмоция по видео целиком'):
                        # Loop through all frames
                        while True:
                            # Capture frame
                            ret, frame = cap.read()
                            if (count % skip == 0 and count > 20):
                                if not ret:
                                    break
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                                if len(faces) != 1:
                                    continue
                                for (x, y, w, h) in faces:
                                    face = frame[y:y + h, x:x + w]

                                face = cv2.resize(face, (122, 122))
                                face = face[5:-5, 5:-5]
                                face = face / 255.
                                frames.append(face)
                            count += 1
                        model_video = load_model('video_r.hdf5')
                        frames = np.array(frames)
                        num_frames = len(frames)
                        pred = model_video.predict(frames)
                        pred_video = np.mean(pred, axis=0)
                        with col1:
                            st.success(f"Эмоция_видео: {emotions[pred_video.argmax()]}")
                    if st.button('Эмоция по видео по кадрам'):
                        try:
                            while cap.isOpened():
                                _, frame = cap.read()
                                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                                faces = haar_cascade.detectMultiScale(
                                    image=img_gray, scaleFactor=1.3, minNeighbors=5)
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(img=frame, pt1=(x, y), pt2=(
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
                                    cv2.putText(frame, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    FRAME_WINDOW.image(frame)
                        except:
                            pass
    elif choice == "Распознавание эмоций по аудиофайлам":
        st.markdown("Загрузите аудиофаил")
        with st.container():
            col1, col2 = st.columns(2)
            # audio_file = None
            # path = None
            with col1:
                audio_file = st.file_uploader("Загрузите аудиофаил в формате wav,mp3", type=['wav', 'mp3', 'ogg'])
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    if_save_audio = AudioTransformer.save_audio(audio_file)
                    if if_save_audio == 1:
                        st.warning("Слишком большой фаил")
                    elif if_save_audio == 0:
                        try:
                            wav, sr = librosa.load(path, sr=44100)
                            # # display audio
                            st.audio(audio_file, format='audio/wav', start_time=0)
                            model_ = load_model("model4.h5")
                            mfccs_ = AudioTransformer.get_mfccs(path, model_.input_shape[-2])
                            mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
                            pred_ = model_.predict(mfccs_)[0]
                            txt = "MFCC\n" + AudioTransformer.get_title(pred_, AudioTransformer.CAT7)
                            fig3 = plt.figure(figsize=(5, 5))
                            COLORS = AudioTransformer.color_dict(AudioTransformer.COLOR_DICT)
                            AudioTransformer.plot_polar(fig3, predictions=pred_, categories=AudioTransformer.CAT7,
                                       title=txt, colors=COLORS)

                            st.write(fig3)
                        except Exception as e:
                            audio_file = None
                            st.error(f"Ошибка")
                    else:
                        st.error("Unknown error")
    elif choice == "Распознавание эмоций по тексту":
        emotions_emoji_dict = {"anger", "disgust", "fear", "happy", "joy",
                               "neutral", "sad", "sadness", "shame", "surprise"}

        with st.form(key='emotion_clf_form'):
            text = st.text_area("Напиши текст")
            submit = st.form_submit_button(label='Определить эмоции')

        if submit:

            if text:
                st.write(f"{text}")
                col1, col2 = st.columns(2)
                # output prediction and proba
                prediction = predict_emotion(text)
                datePrediction = datetime.now()
                probability = get_prediction_proba(text)

                with col1:
                    st.success(f"Вероятность: {np.max(probability) * 100}%")




                if 'texts' and 'probas' and 'predictions' and 'date' not in st.session_state:
                    st.session_state.texts = []
                    st.session_state.predictions = []
                    st.session_state.probas = []
                    st.session_state.date = []


                # store text
                # st.write("User input")
                st.session_state.texts.append(text)
                # st.write(st.session_state.texts)

                # store predictions
                # st.write("Classified emotions")
                st.session_state.predictions.append(prediction.upper())
                # st.write(st.session_state.predictions)

                # store probabilities
                st.session_state.probas.append(np.max(probability) * 100)

                # store date
                st.session_state.date.append(datePrediction)

                prdcts = st.session_state.predictions
                txts = st.session_state.texts
                probas = st.session_state.probas
                dateUser = st.session_state.date

                def get_table_download_link(df):

                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
                    st.markdown(href, unsafe_allow_html=True)

                if 'emotions' and 'occurence' not in st.session_state:
                    st.session_state.emotions = ["ANGER", "DISGUST", "FEAR", "JOY", "NEUTRAL", "SADNESS", "SHAME",
                                                 "SURPRISE"]
                    st.session_state.occurence = [0, 0, 0, 0, 0, 0, 0, 0]

                # Create data frame
                if prdcts and txts and probas:
                    st.write("Data Frame")
                    d = {'Text': txts, 'Emotion': prdcts, 'Probability': probas, 'Date': dateUser}
                    df = pd.DataFrame(d)
                    st.write(df)
                    get_table_download_link(df)

                    ## emotions occurences

                    index_emotion = st.session_state.emotions.index(prediction.upper())
                    st.session_state.occurence[index_emotion] += 1

                    d_pie = {'Emotion': st.session_state.emotions, 'Occurence': st.session_state.occurence}
                    df_pie = pd.DataFrame(d_pie)
                    # st.write("Emotion Occurence")
                    # st.write(df_pie)

                    # df_occur = {'Emotion': prdcts, 'Occurence': occur['Emotion']}
                    # st.write(df_occur)

                    # Line chart
                    # c = alt.Chart(df).mark_line().encode(x='Date',y='Probability')
                    # st.altair_chart(c)

                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("Emotion Occurence")
                        st.write(df_pie)
                    with col4:
                        chart = alt.Chart(df).mark_line().encode(
                            x=alt.X('Date'),
                            y=alt.Y('Probability'),
                            color=alt.Color("Emotion")
                        ).properties(title="Emotions evolution by time")
                        st.altair_chart(chart, use_container_width=True)

                    # Pie chart
                    import plotly.express as px
                    st.write("Probabily of total predicted emotions")
                    fig = px.pie(df_pie, values='Occurence', names='Emotion')
                    st.write(fig)

            else:
                st.write("No text has been submitted!")
    elif choice=="multi":
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                uploaded_video = st.file_uploader("Загрузите видео в формате mp4", type=['mp4'])
                if uploaded_video is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_video.read())
                    cap = cv2.VideoCapture(tfile.name)
                    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    frames = []
                    count = 0
                    skip = 3

                    # Loop through all frames
                    while True:
                        # Capture frame
                        ret, frame = cap.read()
                        if (count % skip == 0 and count > 20):
                            if not ret:
                                break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                            if len(faces) != 1:
                                continue
                            for (x, y, w, h) in faces:
                                face = frame[y:y + h, x:x + w]

                            face = cv2.resize(face, (122, 122))
                            face = face[5:-5, 5:-5]
                            face = face / 255.
                            frames.append(face)
                        count += 1
                    model_video = load_model('video_r.hdf5')
                    frames = np.array(frames)
                    num_frames = len(frames)
                    pred = model_video.predict(frames)
                    pred_video = np.mean(pred, axis=0)


                    print('shape frames:', frames.shape)
                    audiofile = moviepy.editor.AudioFileClip(tfile.name).set_fps(48000)
                    audio = audiofile.to_soundarray()
                    audio = audio[int(48000 / 2):int(48000 / 2 + 48000 * 3)]
                    audio = np.array([elem[0] for elem in audio])
                    mel = librosa.power_to_db(
                        melspectrogram(y=audio))

                    scaler = load('std_scaler.bin')
                    mel = scaler.transform(mel)

                    mel = np.expand_dims(mel, axis=2)
                    mel = np.expand_dims(mel, axis=0)
                    mel.shape





                    model_audio = load_model('audio_r.h5')
                    pred = model_audio.predict(mel)
                    pred_audio = np.mean(pred, axis=0)
                    pred_global = pred_video + pred_audio
                    a = emotions[pred_global.argmax()]
                    b = emotions[pred_audio.argmax()]
                    c = emotions[pred_video.argmax()]
                    with col1:
                        st.success(f"Эмоция_общая: {emotions[pred_global.argmax()]}")
                        st.success(f"Эмоция_аудио: {emotions[pred_audio.argmax()]}")
                        st.success(f"Эмоция_видео: {emotions[pred_video.argmax()]}")
if __name__ == "__main__":
    main()

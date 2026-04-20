import os
import cv2
import tempfile
import numpy as np
from moviepy import VideoFileClip
import speech_recognition as sr
import scipy.io.wavfile as wav


def split_video_into_1sec_chunks(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)

    clip = VideoFileClip(video_path)
    recognizer = sr.Recognizer()

    chunks = []

    for sec in range(duration):
        #FER Split -----------------------------------------------------------
        ferdata = []

        start_frame = sec * fps
        end_frame = (sec + 1) * fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            ferdata.append(frame)

        #SER Split ----------------------------------------------------------
        '''audio_segment = clip.audio.subclip(sec, sec + 1)

        temp_wav = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        audio_segment.write_audiofile(
            temp_wav.name,
            fps=16000,
            verbose=False,
            logger=None
        )

        sample_rate, waveform = wav.read(temp_wav.name)

        serdata = waveform

        #TER Split -------------------------------------------------------------
        try:
            with sr.AudioFile(temp_wav.name) as source:
                audio = recognizer.record(source)

            terdata = recognizer.recognize_google(audio)

        except:
            terdata = ""

        os.remove(temp_wav.name)
        '''
        chunks.append({
            "ferdata": ferdata,
            #"serdata": serdata,
            #"terdata": terdata
        })

    cap.release()
    clip.close()

    return chunks

_face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


def fer_prep(frames):
    
    if _face_cascade.empty():
        raise IOError("Failed to load Haar Cascade. Check the file path.")
    
    n = len(frames)
    k = min(10, n)
    
    idxs = np.linspace(0, n - 1, k, dtype=int)  # 10 evenly spaced frames

    prepped = []

    for i in idxs:
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        
        prepped.append(face.reshape(48, 48, 1))
        
    return np.array(prepped)
import os
import cv2
import numpy as np
import tempfile
import subprocess
import librosa
import soundfile as sf
import whisper

def split_video_into_1sec_chunks(video_path):
    #rebuild of the pipeline using whisper, as i was having difficulties with google speech recognition. could end up being slower but more accurate.
    

    # ---------------------------------------------------------
    # Load video info
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)

    chunks = []

    # ---------------------------------------------------------
    # Extract FULL audio once using ffmpeg
    # ---------------------------------------------------------
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio.close()

    #this is the one part of the pipeline that i have no idea how it works.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            temp_audio.name
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # ---------------------------------------------------------
    # Load Whisper model once
    # ---------------------------------------------------------
    model = whisper.load_model("base")

    # ---------------------------------------------------------
    # Process each second
    # ---------------------------------------------------------
    for sec in range(duration):

        # =====================================================
        # FER SECTION
        # =====================================================
        ferdata = []

        start_frame = sec * fps
        end_frame = (sec + 1) * fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        sections = range(start_frame, end_frame)

        for i in sections:
            print(f"Loading section: {i} of {len(sections)}", flush=True)
            ret, frame = cap.read()
            if not ret:
                break
            ferdata.append(frame)

        # =====================================================
        # SER SECTION
        # =====================================================
        serdata, sample_rate = librosa.load(
            temp_audio.name,
            sr=16000,
            offset=sec,
            duration=3.0,
            mono=True
        )

        # =====================================================
        # TER SECTION
        # =====================================================
        ter_audio, ter_sample_rate = librosa.load(temp_audio.name, sr=16000, offset=max(0, sec-1), duration=3.0)
        
        temp_chunk = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        temp_chunk.close()

        sf.write(temp_chunk.name, ter_audio, ter_sample_rate)

        try:
            result = model.transcribe(
                temp_chunk.name,
                fp16=False,
                language="en"
            )

            terdata = result["text"].strip()

        except Exception as e:
            print("Whisper Error:", e, flush=True)
            terdata = ""

        os.remove(temp_chunk.name)

        # =====================================================
        # Save Chunk
        # =====================================================
        chunks.append({
            "ferdata": ferdata,
            "serdata": serdata,
            "terdata": terdata
        })

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    cap.release()
    os.remove(temp_audio.name)

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



def ser_prep(audio, sample_rate, max_pad_len=174, n_mfcc=40):
    """
    Extracts MFCC features from an audio column.

    Returns a 2D array of shape (n_mfcc, max_pad_len) —
    this becomes one "time-series" sample for the LSTM.
    """
    try:

        # Extract MFCCs — shape: (n_mfcc, time_steps)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Pad or truncate to a fixed length so all samples are the same shape
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        # mfccs  # shape: (40, 174)


        data_t = np.array([mfccs]).transpose(0,2,1)

        return data_t  # shape: (1, 174, 40)



    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

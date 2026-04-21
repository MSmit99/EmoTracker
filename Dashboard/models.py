import cv2
import joblib
import numpy as np
import tensorflow as tf
import pickle
#from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from preprocess import fer_prep, ser_prep

EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']
FER_EMOTIONS = ['Angry','Fearful','Happy','Sad', 'Neutral']
SER_EMOTIONS = ['Angry', 'Fearful', 'Happy', 'Neutral', 'Sad']

'''
The app expects the data from the models in this format:
overall = {
    'model':'name',
    'emotion':'emotion',
    'accuracy':90,
    'all_probs': {"Angry": 30, "Fearful": 10, "Happy": 10, "Neutral": 30, "Sad": 20}
    'timeline': [{
            'second':0,
            'emotion':'emotion',
            'accuracy':90,
            'all_probs':{"Angry": 30, "Fearful": 10, "Happy": 10, "Neutral": 30, "Sad": 20},
        },{
            'second':1,
            ...
        },...]
}

'''

EMPTY_DATA = {
    'emotion':'Neutral',
    'accuracy':0,
    'all_probs': {"Angry": 0, "Fearful": 0, "Happy": 0, "Neutral": 0, "Sad": 0},
    'timeline': [{
            'second':0,
            'emotion':'Neutral',
            'accuracy':0,
            'all_probs':{"Angry": 0, "Fearful": 0, "Happy": 0, "Neutral": 0, "Sad": 0},
        }]
    }

# -------------------------------------------------
# LOAD TRAINED MODELS
# -------------------------------------------------

fer_model = tf.saved_model.load("models/fer_model")

serModel = tf.saved_model.load("models/ser_model")

#terModel = 

def run_fer(chunks):
    timeline = []
    overall = {}
    for second, chunk in enumerate(chunks):
        frames = chunk["ferdata"]
        
        fer_data = fer_prep(frames)
    
        predictions = []
        for face in fer_data:
            face = np.array(face).reshape(1,48,48,1)
            infer = fer_model.signatures["serving_default"]
            
            output = infer(tf.constant(face, dtype=tf.float32))
            probs = list(output.values())[0].numpy()[0]
            
            predictions.append(probs)
        
        if not predictions:
            return {"second": second, "emotion": "Neutral", "accuracy": 0.0, "all_probs": {e: 0.0 for e in FER_EMOTIONS.values()}}
        
        avg_probs = np.mean(predictions, axis=0)
        best_idx = int(np.argmax(avg_probs))
        emotion = FER_EMOTIONS[best_idx]
        print(f"{avg_probs} : {best_idx} : {emotion}",flush=True)
        timeline.append({
            'second':    second,
            'emotion':   emotion,
            'accuracy':  round(float(avg_probs[best_idx]) * 100, 1), #currently this is just the accuracy of the highest value
            'all_probs': {FER_EMOTIONS[i]: round(float(p) * 100, 1) for i, p in enumerate(avg_probs)},
        })
        
        overall = {
            'model': "FER Model (Facial Expression)",
            'emotion': emotion,
            'accuracy': round(float(avg_probs[best_idx]) * 100, 1),
            'all_probs': {FER_EMOTIONS[i]: round(float(p) * 100, 1) for i, p in enumerate(avg_probs)},
            'timeline':timeline
        }
        
    overall['timeline'] = timeline
    
    return overall

def run_ser(chunks):
    timeline = []
    overall = {}
    import pickle
    with open("data.pkl", "wb") as f:
        pickle.dump(chunks, f)
    for second, chunk in enumerate(chunks):
        frames = chunk["serdata"]
        
        ser_data = ser_prep(frames, 16000)
    
        predictions = []

        infer = serModel.signatures["serving_default"]
        
        output = infer(tf.constant(ser_data, dtype=tf.float32))
        probs = list(output.values())[0].numpy()[0]
        
        predictions.append(probs)
        
        if not predictions:
            return {"second": second, "emotion": "Neutral", "accuracy": 0.0, "all_probs": {e: 0.0 for e in SER_EMOTIONS.values()}}
        
        avg_probs = np.mean(predictions, axis=0)
        best_idx = int(np.argmax(avg_probs))
        emotion = SER_EMOTIONS[best_idx]
        print(f"{avg_probs} : {best_idx} : {emotion}",flush=True)
        timeline.append({
            'second':    second,
            'emotion':   emotion,
            'accuracy':  round(float(avg_probs[best_idx]) * 100, 1), #currently this is just the accuracy of the highest value
            'all_probs': {SER_EMOTIONS[i]: round(float(p) * 100, 1) for i, p in enumerate(avg_probs)},
        })
        
        overall = {
            'model': "SER Model (Speech Emotion Recognition)",
            'emotion': emotion,
            'accuracy': round(float(avg_probs[best_idx]) * 100, 1),
            'all_probs': {SER_EMOTIONS[i]: round(float(p) * 100, 1) for i, p in enumerate(avg_probs)},
            'timeline':timeline
        }
        
    overall['timeline'] = timeline
    
    return overall

def run_ter(chunks):
    # BERTweet pipeline
    '''with open("models/bertweet_pipeline_e6.pkl", "rb") as f:
        emotion_pipe = pickle.load(f)
    print(emotion_pipe(["i feel so happy today"]),flush=True)
    '''
    overall = {}
    return overall

def run_models(chunks):
    result = []
    
    fempty = EMPTY_DATA
    sempty = EMPTY_DATA
    tempty = EMPTY_DATA
    fempty['model'] = "FER Model (Facial Expression)"
    sempty['model'] = "SER Model (Speech)"
    tempty['model'] = "TER Model (Textual)"
    
    result.append(run_fer(chunks) or fempty)
    result.append(run_ser(chunks) or sempty)
    result.append(run_ter(chunks) or tempty)
    
    return result
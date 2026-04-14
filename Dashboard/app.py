from flask import Flask, render_template, request, jsonify
import random
import time
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']

EMOTION_COLORS = {
    'Happy':     '#F59E0B',
    'Sad':       '#3B82F6',
    'Angry':     '#EF4444',
    'Fearful':   '#8B5CF6',
    'Neutral':   '#6B7280',
}

#---------------- Plug in section for the models:

def _probs_at(seed: int):
    random.seed(seed)
    scores = [random.random() for _ in EMOTIONS]
    total  = sum(scores)
    return [round(s / total * 100, 1) for s in scores]


def run_model(model_name: str, seed: int, duration: int = 60):
    time.sleep(random.uniform(0.1, 0.3))
    probs    = _probs_at(seed)
    best_idx = probs.index(max(probs))
    overall  = {
        'model':     model_name,
        'emotion':   EMOTIONS[best_idx],
        'accuracy':  probs[best_idx],
        'color':     EMOTION_COLORS[EMOTIONS[best_idx]],
        'all_probs': dict(zip(EMOTIONS, probs)),
    }
    timeline = []
    for t in range(duration):
        p  = _probs_at(seed + t * 7)
        bi = p.index(max(p))
        timeline.append({
            'second':    t,
            'emotion':   EMOTIONS[bi],
            'accuracy':  p[bi],
            'all_probs': dict(zip(EMOTIONS, p)),
        })
    overall['timeline'] = timeline
    return overall

#---------------- End plug in section

#---------------- Section for Ensemble Model:

def combine_at(results: list, second: int = None):
    combined = {e: 0.0 for e in EMOTIONS}
    for r in results:
        src = r['all_probs']
        if second is not None and r.get('timeline') and second < len(r['timeline']):
            src = r['timeline'][second]['all_probs']
        for emotion, prob in src.items():
            combined[emotion] += prob / len(results)
    combined = {k: round(v, 1) for k, v in combined.items()}
    best     = max(combined, key=combined.get)
    return {
        'emotion':   best,
        'accuracy':  combined[best],
        'color':     EMOTION_COLORS[best],
        'all_probs': combined,
    }
    
#--------------- End Ensemble Model Section


def build_ensemble_timeline(results: list):
    duration = min(len(r['timeline']) for r in results)
    return [combine_at(results, second=t) for t in range(duration)]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    video.seek(0, os.SEEK_END)
    seed = video.tell() % 10_000
    video.seek(0)
    models = [
        ('SER Model (Speech)',            seed + 1),
        ('FER Model (Facial Expression)', seed + 2),
        ('TER Model (Textual)',           seed + 3),
    ]
    results           = [run_model(name, s, duration=60) for name, s in models]
    ensemble_overall  = combine_at(results)
    ensemble_timeline = build_ensemble_timeline(results)
    return jsonify({
        'filename': video.filename,
        'models':   results,
        'ensemble': {**ensemble_overall, 'timeline': ensemble_timeline},
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, render_template, request, jsonify
import os
import tempfile

from preprocess import split_video_into_1sec_chunks
from models import run_models
from utils import combine_at, average_timeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files['video']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video.save(temp.name)
        path = temp.name
        print("saved", flush=True)
    try:
        chunks = split_video_into_1sec_chunks(path)
        print('chunked',flush=True)

        results = run_models(chunks)
        
        print("Completed",flush=True)

        ensemble_overall = combine_at(results)
        ensemble_timeline = average_timeline(results)

        return jsonify({
            'filename':video.filename,
            'models': results,
            'ensemble':{**ensemble_overall, "timeline":ensemble_timeline},
            })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(path)

if __name__ == '__main__':
    app.run(debug=True)
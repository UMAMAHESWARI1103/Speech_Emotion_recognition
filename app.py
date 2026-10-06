from flask import Flask, render_template, request, send_from_directory, url_for
import joblib
import numpy as np
import librosa
import soundfile as sf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")

model = joblib.load("emotion_model.pkl")

# Emotion → Emoji
EMOJI_MAP = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

def extract_features(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def generate_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    img_path = "static/waveform.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    emoji = ""
    audio_url = None
    waveform_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html")

        filepath = "static/upload.wav"
        file.save(filepath)

        features = extract_features(filepath).reshape(1, -1)
        prediction = model.predict(features)[0]

        emoji = EMOJI_MAP.get(prediction, "")

        waveform_path = generate_waveform(filepath)

        audio_url = url_for('static', filename='upload.wav')
        waveform_url = url_for('static', filename='waveform.png')

    return render_template(
        "index.html",
        prediction=prediction,
        emoji=emoji,
        audio_url=audio_url,
        waveform_url=waveform_url
    )

if __name__ == "__main__":
    app.run(debug=True)

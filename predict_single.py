import joblib
import numpy as np
import librosa
import soundfile as sf

# Load your trained model
model = joblib.load("emotion_model.pkl")

# Function to extract MFCC features
def extract_features(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert stereo to mono
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

#  Path to the WAV file you want to test
file_path = "D:\Mini\mini duplicate\speech_emotion_recognition\audio"


# Extract features and reshape for prediction
features = extract_features(file_path).reshape(1, -1)

# Predict emotion
prediction = model.predict(features)

print(f" Predicted emotion: {predicted_emotion} {emoji}")

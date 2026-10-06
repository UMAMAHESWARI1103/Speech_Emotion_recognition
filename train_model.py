import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Emotion label mapping (RAVDESS)
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Feature extraction
def extract_features(file_path):
    try:
        audio, sample_rate = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Path to audio folder
audio_dir = os.path.join(os.getcwd(), "audio")
print("Searching in:", audio_dir)

features = []
labels = []

# Iterate over WAV files
for root, _, files in os.walk(audio_dir):
    for file_name in files:
        if file_name.endswith(".wav"):
            print(f"📁 Processing: {file_name}")
            file_path = os.path.join(root, file_name)

            # FIXED — remove .wav before splitting
            name_only = file_name.replace(".wav", "")
            parts = name_only.split("-")

            # RAVDESS format: XX-XX-EMOTION-XX-XX-XX-XX
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code)

                if emotion:
                    mfcc = extract_features(file_path)
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(emotion)
                else:
                    print(f"⚠ Unknown emotion code: {emotion_code} in file {file_name}")
            else:
                print(f"❌ Invalid filename format: {file_name}")

# Convert lists to arrays
X = np.array(features)
y = np.array(labels)

print(f"📊 Total samples loaded: {len(X)}")

# Training
if len(X) == 0:
    print("❌ No data found. Please check audio folder or filename format.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {acc * 100:.2f}%")

    # Save model
    joblib.dump(model, "emotion_model.pkl")
    print("✅ Model saved as emotion_model.pkl")

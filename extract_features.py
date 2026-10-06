import soundfile as sf
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:  # If stereo, convert to mono
            audio = np.mean(audio, axis=1)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

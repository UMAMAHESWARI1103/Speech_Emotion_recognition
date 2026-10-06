def extract_features(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert stereo to mono
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs
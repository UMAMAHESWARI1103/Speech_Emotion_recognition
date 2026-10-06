# prepare_data.py
import os
import numpy as np
import soundfile as sf
import librosa

def extract_features(file_path):
    try:
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

# --- locate audio folder relative to this script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(script_dir, "audio")

print("🔎 Looking for audio folder at:", audio_dir)
print("Exists?:", os.path.exists(audio_dir))

if os.path.exists(audio_dir):
    try:
        listing = os.listdir(audio_dir)
        print("📁 Files in audio folder (first 50 shown):", listing[:50])
    except Exception as e:
        print("❌ Could not list directory contents:", e)
else:
    print("❌ Audio folder not found. Make sure `audio/` exists next to this script.")
    raise SystemExit(1)

features = []
labels = []
processed = 0
skipped = 0

# If your files are nested in subfolders, os.walk will find them.
for root, _, files in os.walk(audio_dir):
    print("\n📂 Entering folder:", root)
    for fname in files:
        print("  - Found:", fname)
        if not fname.lower().endswith(".wav"):
            print("    ↳ skipping (not .wav)")
            skipped += 1
            continue

        # full path
        fpath = os.path.join(root, fname)

        # remove extension safely
        name_only = fname[:-4] if fname.lower().endswith(".wav") else fname
        parts = name_only.split("-")

        # Try RAVDESS-style parsing first (emotion code is at index 2)
        emotion = None
        if len(parts) >= 3:
            emotion_code = parts[2]
            # map expected codes like "01","02",...
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
            emotion = emotion_map.get(emotion_code)
            if emotion:
                print(f"    ↳ parsed RAVDESS emotion code {emotion_code} -> {emotion}")
        # If not found, try a fallback: maybe filename is like 'happy_01.wav' or 'happy.wav'
        if emotion is None:
            # fallback: try to find any of the emotion names in filename
            fallback_map = {
                "neutral":"neutral","calm":"calm","happy":"happy","sad":"sad",
                "angry":"angry","fearful":"fearful","disgust":"disgust","surprised":"surprised"
            }
            lower_name = name_only.lower()
            for k,v in fallback_map.items():
                if k in lower_name:
                    emotion = v
                    print(f"    ↳ fallback parsed by substring -> {emotion}")
                    break

        if emotion is None:
            print("    ↳ SKIP: could not determine emotion from filename:", fname)
            skipped += 1
            continue

        # extract features
        feats = extract_features(fpath)
        if feats is None:
            print("    ↳ SKIP: feature extraction failed for", fname)
            skipped += 1
            continue

        features.append(feats)
        labels.append(emotion)
        processed += 1

# Summary
print("\n================ SUMMARY ================")
print("Total files processed (features collected):", processed)
print("Total files skipped:", skipped)
print("Total features length:", len(features))

if len(features) == 0:
    print("❌ No features were extracted. Check the debug output above.")
    raise SystemExit(1)

# convert and save
X = np.array(features)
y = np.array(labels)
np.save("features.npy", X)
np.save("labels.npy", y)
print("✅ Saved features.npy and labels.npy")
print("features shape:", X.shape)
print("labels shape:", y.shape)

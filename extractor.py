import pandas as pd
import librosa
import numpy as np
def extract_features(audio_file):
    # Function to extract features from an audio file
    y, sr = librosa.load(audio_file)
    features = {}

    # Get the length of the audio signal
    signal_length = len(y)

    # Set n_fft dynamically (at least 256 and at most the signal length)
    n_fft = min(1024, signal_length) if signal_length >= 256 else signal_length

    # Temporal Features
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['rms_energy'] = np.mean(librosa.feature.rms(y=y))

    # Spectral Features (with dynamic n_fft)
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft))
    features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))

    # MFCC Features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
    for i in range(1, 14):
        features[f'mfcc_{i}'] = np.mean(mfccs[i-1])
    # Chroma Features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
    for i in range(1, 13):
        features[f'chroma_{i}'] = np.mean(chroma_stft[i-1])

    # Chroma CENS
    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i in range(1, 13):
        features[f'cens_{i}'] = np.mean(cens[i-1])

    # Tempo and Beat Features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # Pitch and Pitch Confidence
    pitch, pitch_confidence = librosa.core.piptrack(y=y, sr=sr)
    features['pitch'] = np.mean(pitch[pitch > 0])
    features['pitch_confidence'] = np.mean(pitch_confidence[pitch > 0])

    # Mel-Spectrogram and Log Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    features['mel_spectrogram'] = np.mean(mel_spectrogram)
    extract_formants=[]
    # Formant Features
    formants = extract_formants(y, sr)
    for i, formant in enumerate(formants):
        features[f'formant_{i+1}'] = formant

    return features
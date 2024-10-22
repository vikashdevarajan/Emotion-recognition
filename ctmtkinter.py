import sounddevice as sd
import wave
import threading
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import customtkinter as ctk
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.compat.v1 import ConfigProto  # type: ignore
from tensorflow.compat.v1 import InteractiveSession  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import StandardScaler
import time
import librosa
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.models import load_model # type: ignore
warnings.filterwarnings("ignore")
fs = 16000 
is_recording = False
audio_data = None
duration = 8  
filename = "output.wav"
countdown_label = None
extracted_features = {} 
features_display = "" 
def record_audio():
    global is_recording, audio_data
    is_recording = True
    audio_data = np.zeros((0, 1), dtype=np.float32)
    def countdown():
        for i in range(duration, 0, -1):
            if not is_recording:
                break
            update_countdown(i) 
            time.sleep(1) 
    countdown_thread = threading.Thread(target=countdown)
    countdown_thread.start()
    def callback(indata, frames, time, status):
        global audio_data
        if is_recording:
            audio_data = np.append(audio_data, indata, axis=0) 
    with sd.InputStream(samplerate=fs, channels=1, callback=callback, dtype=np.float32):  
        sd.sleep(duration * 1000)  
        stop_recording()  
def stop_recording():
    global is_recording
    is_recording = False
    update_countdown(0) 
    print("Recording stopped.")
def upload_audio():
    global features_display
    filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if filepath:
        messagebox.showinfo("File Selected", f"Selected Audio File: {filepath}")
        extracted_features = extract_features(filepath)
        features_display = format_features(extracted_features)
        update_features_display()
        emotion_prediction, probabilities, accuracy = predict_emotion4(extracted_features)
        a = f'Predicted emotion: {emotion_prediction}\n\nAccuracy: {accuracy * 100:.10f}%\n'
        for prob in probabilities:
            a += f"{prob}\n"
        prediction_result.delete(1.0, tk.END) 
        prediction_result.insert(tk.END, a)
        messagebox.showinfo("Prediction", f"Predicted Emotion: {emotion_prediction}")
        
    else:
        messagebox.showerror("Error", "No file selected!")
def save_audio():
    global filename, audio_data, extracted_features, features_display
    if audio_data is not None and len(audio_data) > 0:
        directory = "D:/audio"
        if not os.path.exists(directory):
            os.makedirs(directory)
        existing_files = [f for f in os.listdir(directory) if f.startswith('output') and f.endswith('.wav')]
        next_number = len(existing_files) + 1
        filename = f"output_{next_number}.wav"
        filepath = os.path.join(directory, filename)
        write_wav_file(filepath, audio_data)
        messagebox.showinfo("Success", f"Recording saved as {filename} in {directory}")
        extracted_features = extract_features(filepath)
        features_display = format_features(extracted_features) 
        update_features_display()
        emotion_prediction, probabilities, accuracy = predict_emotion4(extracted_features)
        a = f'Predicted emotion: {emotion_prediction}\n\nAccuracy: {accuracy * 100:.10f}%\n'
        for prob in probabilities:
            a += f"{prob}\n"
        prediction_result.delete(1.0, tk.END) 
        prediction_result.insert(tk.END, a) 
        messagebox.showinfo("Prediction", f"Predicted Emotion: {emotion_prediction}")

    else:
        messagebox.showerror("Error", "No recording to save!")
def write_wav_file(filename, data):
    """Function to save audio data to .wav format"""
    scaled_data = np.int16(data * 32767)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1) 
        wf.setsampwidth(2) 
        wf.setframerate(fs)
        wf.writeframes(scaled_data.tobytes())
def start_recording_thread():
    """Start the recording in a new thread to avoid blocking the UI"""
    thread = threading.Thread(target=record_audio)
    thread.start()
def start_recording_thread():
    """Start the recording in a new thread to avoid blocking the UI"""
    thread = threading.Thread(target=record_audio)
    thread.start()
def stop_recording_thread():
    """Stop the recording thread"""
    stop_recording()
def update_countdown(value):
    """Update the countdown timer"""
    countdown_label.configure(text=f"Time remaining: {value} seconds")
def update_features_display():
    """Update the features display area with extracted features"""
    features_text.delete(1.0, tk.END)
    features_text.insert(tk.END, features_display)
def format_features(features):
    """Format extracted features for display"""
    formatted_features = ""
    for feature, value in features.items():
        formatted_features += f"{feature}: {value}\n"
    return formatted_features
def extract_formants(y, sr):
    formants = []
    order = 2 + sr // 1000 
    lpc_coeffs = librosa.lpc(y, order=order)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    frequencies = sorted(angles * (sr / (2 * np.pi)))
    formants = frequencies[:3] if len(frequencies) >= 3 else frequencies
    return formants
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    features = {}
    signal_length = len(y)
    n_fft = min(1024, signal_length) if signal_length >= 256 else signal_length
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft))
    features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
    for i in range(1, 14):
        features[f'mfcc_{i}'] = np.mean(mfccs[i-1])
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
    for i in range(1, 13):
        features[f'chroma_{i}'] = np.mean(chroma_stft[i-1])
    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i in range(1, 13):
        features[f'cens_{i}'] = np.mean(cens[i-1])
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    pitch, pitch_confidence = librosa.core.piptrack(y=y, sr=sr)
    features['pitch'] = np.mean(pitch[pitch > 0])
    features['pitch_confidence'] = np.mean(pitch_confidence[pitch > 0])
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
    features['mel_spectrogram'] = np.mean(mel_spectrogram)
    formants = extract_formants(y, sr)
    for i, formant in enumerate(formants):
        features[f'formant_{i+1}'] = formant
    return features
def predict_emotion4(extracted_features):
    data = pd.read_csv('D:\\ML PROJECT\\dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target'] 
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = load_model('D:\\ML PROJECT\\emotion_classification_model.keras')
    results = model.evaluate(X_test, y_test)
    print(f"Loaded model test accuracy: {results[1]}")
    extracted_features_df = pd.DataFrame([extracted_features])
    scaled_input = scaler.transform(extracted_features_df)  
    prediction = model.predict(scaled_input)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    emotions = {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgust',
        7: 'surprised'
    }
    predicted_emotion = emotions.get(predicted_class_index, 'unknown')
    print(f"The predicted emotion is: {predicted_emotion}")
    probs=[]
    for idx, prob in enumerate(prediction[0]):
        emotion_label = emotions.get(idx, 'unknown')
        probs_printer=f"Probability of {emotion_label}: {prob:.10f}"
        probs.append(probs_printer)
        print(probs_printer)
    return predicted_emotion, probs, results[1]
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("Audio Recorder & Emotion Predictor")
root.geometry("1200x900")
frame_buttons = ctk.CTkFrame(root, width=300, height=350)
frame_buttons.pack(side=ctk.TOP, padx=0, pady=0, fill=ctk.X)
frame_features = ctk.CTkFrame(root, width=300, height=350)
frame_features.pack(side=ctk.TOP, padx=0, pady=0, fill=ctk.X)
frame_prediction = ctk.CTkFrame(root, width=300, height=350)
frame_prediction.pack(side=ctk.TOP, padx=0, pady=0, fill=ctk.X)
record_button = ctk.CTkButton(frame_buttons, text="Record", command=start_recording_thread, width=150)
record_button.pack(padx=10, pady=10)
stop_button = ctk.CTkButton(frame_buttons, text="Stop", command=stop_recording_thread, width=150)
stop_button.pack(padx=10, pady=10)
upload_button = ctk.CTkButton(frame_buttons, text="Upload Audio", command=upload_audio, width=150)
upload_button.pack(padx=10, pady=10)
save_button = ctk.CTkButton(frame_buttons, text="Save & Predict", command=save_audio, width=150)
save_button.pack(padx=10, pady=10)
countdown_label = ctk.CTkLabel(frame_buttons, text="Time remaining: 8 seconds", font=("Helvetica", 12))
countdown_label.pack(padx=10, pady=10)
features_text = ctk.CTkTextbox(frame_features, height=200, width=1200)
features_text.pack(padx=10, pady=10)
prediction_result = ctk.CTkTextbox(frame_prediction, height=200, width=1200)
prediction_result.pack(padx=10, pady=10)
root.mainloop()